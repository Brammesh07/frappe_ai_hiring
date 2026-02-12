# Copyright (c) 2025, Your Company and contributors
# For license information, please see license.txt

"""
Resume Parser Service
Parses resume text using AI and creates structured candidate profiles
"""

import frappe
import json
from typing import Dict, Any, Optional
from frappe_ai_hiring.ai_hiring.utils.llm_client import LLMClient
from frappe_ai_hiring.ai_hiring.utils.pii_redactor import PIIRedactor
from frappe_ai_hiring.ai_hiring.utils.audit_logger import AIAuditLogger


# Prompt version for tracking
RESUME_PARSING_PROMPT_VERSION = "1.0.0"


def get_resume_parsing_prompt(resume_text: str) -> tuple[str, str]:
	"""
	Get system and user prompts for resume parsing

	Args:
		resume_text: Resume text to parse

	Returns:
		Tuple of (system_prompt, user_prompt)
	"""
	system_prompt = """You are an expert resume parser and HR analyst.
Your task is to extract structured information from resumes accurately.

CRITICAL RULES:
- Extract only factual information present in the resume
- Do not infer or guess information not explicitly stated
- Normalize skill names (e.g., "JS" → "JavaScript")
- Calculate total experience from dates
- Respond ONLY with valid JSON matching the schema
- Do not include any text outside the JSON object

EDUCATION RELEVANCE CLASSIFICATION RULES (STRICT):
- You must assign EXACTLY ONE value to education_relevance.
- Choose ONLY from: Highly Relevant, Relevant, Somewhat Relevant, Not Relevant.
- Use the MOST relevant degree only if multiple degrees are present.
- Decision order (first match wins):
  1. Highly Relevant:
     Degree field exactly matches the job role or is a direct prerequisite
     (e.g., Computer Science for Software Developer).
  2. Relevant:
     Degree is closely related and commonly accepted for the role
     (e.g., Information Technology, Software Engineering).
  3. Somewhat Relevant:
     Degree is indirectly related or provides transferable skills
     (e.g., Electronics, Mechanical for software roles).
  4. Not Relevant:
     Degree has no meaningful relationship to the job role.
- When uncertain, choose the LOWER relevance category.
- Do NOT explain the choice. Return the label only."""

	user_prompt = f"""Parse this resume and extract structured information.

Resume Text:
{resume_text}

OUTPUT SCHEMA (respond with valid JSON only):
{{
  "skills": ["skill1", "skill2", ...],
  "experience_years": <float>,
  "education": [
    {{
      "degree": "string",
      "institution": "string",
      "year": "string",
      "field": "string"
    }}
  ],
  "experience": [
    {{
      "title": "string",
      "company": "string",
      "duration": "string",
      "responsibilities": ["resp1", "resp2"]
    }}
  ],
  "projects": [
    {{
      "title": "string",
      "candidate_contribution": "string",
      "skills": ["skill1", "skill2"]
    }}
  ],
  "certifications": ["cert1", "cert2"],
  "education_relevance": "Highly Relevant| Relevant| Somewhat Relevant| Not Relevant",
  "summary": "Brief 2-3 sentence professional summary",
  "confidence_score": <0.0-1.0>
}}

Parse the resume now:"""

	return system_prompt, user_prompt


def parse_resume(applicant_name: str, job_opening: str) -> Optional[str]:
	"""
	Parse resume and create AI Candidate Profile

	Args:
		applicant_name: Job Applicant name
		job_opening: Job Opening name

	Returns:
		AI Candidate Profile name or None
	"""
	try:
		frappe.logger("ai_hiring").info(f"Parsing resume for: {applicant_name}")

		# Step 1: Extract resume text
		from frappe_ai_hiring.ai_hiring.services.resume_extractor import extract_resume_text

		resume_text = extract_resume_text(applicant_name)

		if not resume_text:
			frappe.throw("No resume text found")

		# Step 2: Redact PII
		settings = frappe.get_cached_doc("AI Settings", "AI Settings")
		redacted_text = resume_text
		token_map = {}

		if settings.enable_pii_redaction:
			redactor = PIIRedactor()
			redacted_text, token_map = redactor.redact_text(resume_text)

		# Step 3: Call LLM for parsing
		client = LLMClient()
		system_prompt, user_prompt = get_resume_parsing_prompt(redacted_text)

		parsed_data = client.call_llm(
			prompt=user_prompt,
			system_prompt=system_prompt,
			operation="Resume Parsing",
			metadata={"doctype": "Job Applicant", "docname": applicant_name, "job_opening": job_opening},
		)

		# Step 4: Validate parsed data
		required_fields = ["skills", "experience_years", "education", "experience", "projects"]
		for field in required_fields:
			if field not in parsed_data:
				frappe.throw(f"Missing required field in parsed data: {field}")

		# Step 5: Create AI Candidate Profile
		profile = frappe.get_doc(
			{
				"doctype": "AI Candidate Profile",
				"applicant": applicant_name,
				"job_opening": job_opening,
				"pii_redacted": 1 if settings.enable_pii_redaction else 0,
			}
		)

		# Set parsed data
		config = settings.get_api_config()
		profile.set_parsed_data(
			data=parsed_data,
			model=config.get("model"),
			confidence=parsed_data.get("confidence_score", 0.0),
		)

		# Store PII token map (encrypted)
		if token_map:
			profile.pii_token_map = json.dumps(token_map)

		profile.insert(ignore_permissions=True)
		frappe.db.commit()

		frappe.logger("ai_hiring").info(
			f"Created AI Candidate Profile: {profile.name}"
		)

		return profile.name

	except Exception as e:
		error_msg = f"Resume parsing failed for {applicant_name}: {str(e)}"
		frappe.logger("ai_hiring").error(error_msg)
		AIAuditLogger.log_error(
			operation="Resume Parsing",
			error_message=error_msg,
			metadata={"doctype": "Job Applicant", "docname": applicant_name, "job_opening": job_opening},
		)
		raise


def create_candidate_profile(job_applicant: str, job_opening: Optional[str] = None) -> Optional[str]:
	"""Public wrapper to parse resume and create AI Candidate Profile."""
	if not job_opening:
		job_opening = frappe.db.get_value("Job Applicant", job_applicant, "job_title")
		if not job_opening:
			frappe.throw(f"Job Opening not found for applicant {job_applicant}")

	return parse_resume(job_applicant, job_opening)


def validate_resume_schema(data: Dict[str, Any]) -> bool:
	"""
	Validate parsed resume data against schema

	Args:
		data: Parsed resume data

	Returns:
		True if valid, raises exception otherwise
	"""
	required_fields = [
		"skills",
		"experience_years",
		"education",
		"experience",
		"projects",
		"education_relevance",
		"summary",
	]

	for field in required_fields:
		if field not in data:
			frappe.throw(f"Missing required field: {field}")

	# Validate types
	if not isinstance(data["skills"], list):
		frappe.throw("skills must be a list")

	if not isinstance(data["experience_years"], (int, float)):
		frappe.throw("experience_years must be a number")

	if not isinstance(data["education"], list):
		frappe.throw("education must be a list")

	if not isinstance(data["experience"], list):
		frappe.throw("experience must be a list")

	# Validate education relevance
	valid_relevance = ["Highly Relevant", "Relevant", "Somewhat Relevant", "Not Relevant"]
	if data["education_relevance"] not in valid_relevance:
		frappe.throw(f"education_relevance must be one of: {', '.join(valid_relevance)}")

	return True


@frappe.whitelist()
def reparse_resume(applicant_name: str, job_opening: str):
	"""
	Reparse resume for an applicant

	Args:
		applicant_name: Job Applicant name
		job_opening: Job Opening name

	Returns:
		Result dictionary
	"""
	try:
		# Delete existing profile if any
		existing = frappe.db.get_value(
			"AI Candidate Profile", {"applicant": applicant_name, "job_opening": job_opening}
		)
		if existing:
			frappe.delete_doc("AI Candidate Profile", existing, ignore_permissions=True)

		# Parse resume
		profile_name = parse_resume(applicant_name, job_opening)

		return {
			"success": True,
			"message": "Resume reparsed successfully",
			"profile": profile_name,
		}

	except Exception as e:
		return {"success": False, "message": str(e)}
