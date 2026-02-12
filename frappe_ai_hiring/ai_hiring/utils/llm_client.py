# Copyright (c) 2025, Your Company and contributors
# For license information, please see license.txt

"""
LLM Client
Handles communication with OpenAI-compatible APIs
"""

import frappe
import json
import requests
from typing import Dict, Any, Optional
from frappe_ai_hiring.ai_hiring.utils.audit_logger import AIAuditLogger


class LLMClient:
	"""Client for interacting with LLM APIs"""

	def __init__(self):
		"""Initialize LLM client with settings"""
		self.settings = self._get_settings()

	def _get_settings(self):
		"""Get AI settings"""
		from frappe_ai_hiring.ai_hiring.doctype.ai_settings.ai_settings import get_ai_settings

		settings = get_ai_settings()
		if not settings:
			frappe.throw("AI Settings not configured. Please configure AI Settings first.")

		if not settings.enable_ai_processing:
			frappe.throw("AI Processing is disabled in settings")

		return settings

	def call_llm(
		self,
		prompt: str,
		system_prompt: Optional[str] = None,
		operation: str = "generic",
		metadata: Optional[Dict[str, Any]] = None,
		temperature: Optional[float] = None,
		max_tokens: Optional[int] = None,
	) -> Dict[str, Any]:
		"""
		Make an LLM API call

		Args:
			prompt: User prompt
			system_prompt: System prompt (optional)
			operation: Operation type for logging
			metadata: Additional metadata for logging
			temperature: Override default temperature
			max_tokens: Override default max tokens

		Returns:
			Parsed JSON response from LLM
		"""
		try:
			# Prepare request
			config = self.settings.get_api_config()
			headers = self._get_headers(config)
			payload = self._build_payload(
				prompt=prompt,
				system_prompt=system_prompt,
				config=config,
				temperature=temperature,
				max_tokens=max_tokens,
			)
			print(json.dumps(payload, indent=2))
			# Make API call
			url = self._get_endpoint_url(config)
			response = requests.post(
				url, headers=headers, json=payload, timeout=config.get("timeout", 60)
			)

			response.raise_for_status()
			result = response.json()

			# Extract content
			content = self._extract_content(result, config)

			# Parse JSON from content
			parsed_response = self._parse_json_response(content)

			# Log successful call
			AIAuditLogger.log_llm_call(
				operation=operation,
				prompt=prompt[:500],  # Truncate for storage
				response=json.dumps(parsed_response)[:1000],
				model=config.get("model"),
				metadata=metadata,
				success=True,
			)

			return parsed_response

		except requests.exceptions.RequestException as e:
			error_msg = f"API request failed: {str(e)}"
			AIAuditLogger.log_error(operation, error_msg, metadata)
			frappe.throw(error_msg)

		except json.JSONDecodeError as e:
			error_msg = f"Failed to parse JSON response: {str(e)}"
			AIAuditLogger.log_error(operation, error_msg, metadata)
			frappe.throw(error_msg)

		except Exception as e:
			error_msg = f"LLM call failed: {str(e)}"
			AIAuditLogger.log_error(operation, error_msg, metadata)
			frappe.throw(error_msg)

	def _get_headers(self, config: Dict[str, Any]) -> Dict[str, str]:
		"""Build request headers"""
		headers = {"Content-Type": "application/json"}

		if config.get("api_key"):
			headers["Authorization"] = f"Bearer {config['api_key']}"

		return headers

	def _get_endpoint_url(self, config: Dict[str, Any]) -> str:
		"""Get API endpoint URL"""
		base_url = config.get("api_base_url", "https://api.openai.com/v1")

		# Remove trailing slash
		base_url = base_url.rstrip("/")

		# For OpenAI-compatible APIs
		if "/api/chat" not in base_url:
			return f"{base_url}/api/chat"

		return base_url

	def _build_payload(
		self,
		prompt: str,
		system_prompt: Optional[str],
		config: Dict[str, Any],
		temperature: Optional[float] = None,
		max_tokens: Optional[int] = None,
	) -> Dict[str, Any]:
		"""Build API request payload"""
		messages = []

		if system_prompt:
			messages.append({"role": "system", "content": system_prompt})

		messages.append({"role": "user", "content": prompt})

		payload = {
			"model": config.get("model"),
			"messages": messages,
			"temperature": temperature or config.get("temperature", 0.2),
			"max_tokens": max_tokens or config.get("max_tokens", 2000),
			"stream": False,
		}

		return payload

	def _extract_content(self, response: Dict[str, Any], config: Dict[str, Any]) -> str:
		"""Extract content from API response"""
		provider = config.get("provider", "OpenAI")

		# Standard OpenAI format
		if "choices" in response and len(response["choices"]) > 0:
			return response["choices"][0]["message"]["content"]

		# Ollama /api/chat format (non-streaming)
		if "message" in response and "content" in response["message"]:
			return response["message"]["content"]

		# Fallback
		if "content" in response:
			return response["content"]

		frappe.throw(f"Unable to extract content from API response. Response keys: {list(response.keys())}")

	def _parse_json_response(self, content: str) -> Dict[str, Any]:
		"""
		Parse JSON from LLM response
		Handles markdown code blocks and other formatting
		"""
		# Try direct parse first
		try:
			return json.loads(content)
		except json.JSONDecodeError:
			pass

		# Try to extract JSON from markdown code block
		import re

		json_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
		if json_match:
			try:
				return json.loads(json_match.group(1))
			except json.JSONDecodeError:
				pass

		# Try to find JSON object in text
		json_match = re.search(r"\{.*\}", content, re.DOTALL)
		if json_match:
			try:
				return json.loads(json_match.group(0))
			except json.JSONDecodeError:
				pass

		frappe.throw(f"Unable to parse JSON from response: {content[:200]}")

	def test_connection(self) -> Dict[str, Any]:
		"""Test API connection"""
		try:
			response = self.call_llm(
				prompt="Respond with: {\"status\": \"ok\", \"message\": \"Connection successful\"}",
				system_prompt="You are a test assistant. Respond only with valid JSON.",
				operation="Other",
			)
			return response
		except Exception as e:
			return {"status": "error", "message": str(e)}


# Convenience wrapper used by services that import call_llm directly
def call_llm(
	system_prompt: Optional[str],
	user_prompt: str,
	operation_type: str = "generic",
	reference_doctype: Optional[str] = None,
	metadata: Optional[Dict[str, Any]] = None,
	temperature: Optional[float] = None,
	max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
	"""Invoke the LLM using shared settings."""
	client = LLMClient()
	return client.call_llm(
		prompt=user_prompt,
		system_prompt=system_prompt,
		operation=operation_type,
		metadata=metadata or {"doctype": reference_doctype} if reference_doctype else metadata,
		temperature=temperature,
		max_tokens=max_tokens,
	)


@frappe.whitelist()
def test_llm_client():
	"""Test LLM client (for debugging)"""
	try:
		client = LLMClient()
		result = client.test_connection()
		return {"success": True, "result": result}
	except Exception as e:
		return {"success": False, "error": str(e)}
