"""
Gemini API Client
=================

Client for Google's Gemini API that powers PA's reasoning.
Supports text and image inputs.
"""

import base64
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional

from .telemetry import get_telemetry


class GeminiAPIError(Exception):
    """Exception for Gemini API errors with retry metadata."""

    def __init__(
        self,
        message: str,
        error_type: str,
        status_code: Optional[int] = None,
        retryable: bool = True,
    ):
        """
        Initialize Gemini API error.

        Args:
            message: Error message.
            error_type: Type of error (http, network, parse, etc).
            status_code: HTTP status code if applicable.
            retryable: Whether this error should be retried.
        """
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        self.retryable = retryable

    @property
    def is_client_error(self) -> bool:
        """Return True if this is a client error (4xx) that shouldn't be retried."""
        return self.status_code is not None and 400 <= self.status_code < 500

    def to_error_string(self) -> str:
        """Format as error string for parsing detection."""
        if self.status_code:
            return f"[GEMINI_ERROR:{self.error_type}:{self.status_code}:{self.message}]"
        return f"[GEMINI_ERROR:{self.error_type}:{self.message}]"

    def __str__(self) -> str:
        return self.message


class GeminiClient:
    """
    Client for Gemini API that powers PA's reasoning.
    
    Supports:
    - Text prompts (system + user)
    - Image inputs (inline base64 encoding)
    
    Usage:
        client = GeminiClient()
        response = client.call("You are helpful", "What is 2+2?")
    """
    
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    
    # Supported image MIME types
    MIME_TYPES = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
    }
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
            
        Raises:
            ValueError: If no API key found.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
    
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_retries: int = 3,
    ) -> str:
        """
        Make API call to Gemini with optional images and retry logic.

        Args:
            system_prompt: System instruction for the model.
            user_prompt: User prompt/question.
            image_paths: Optional list of image file paths to include.
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum output tokens.
            max_retries: Maximum number of retry attempts (default: 3).

        Returns:
            Model's text response.
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return self._call_once(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_paths=image_paths,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    attempt=attempt,
                )
            except GeminiAPIError as e:
                last_error = e

                # Don't retry on client errors (4xx) or non-retryable errors
                if e.is_client_error or not e.retryable:
                    return e.to_error_string()

                # Retry on server errors (5xx) and network errors
                if attempt < max_retries - 1:
                    delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"[PA] Gemini API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    # Final attempt failed
                    return e.to_error_string()
            except Exception as e:
                # Unexpected errors - don't retry
                return f"[Gemini Error: {str(e)[:100]}]"

        # Should not reach here, but just in case
        if last_error:
            return last_error.to_error_string()
        return "[Gemini Error: Unknown error]"

    def _call_once(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: Optional[List[str]],
        temperature: float,
        max_tokens: int,
        attempt: int,
    ) -> str:
        """
        Single API call attempt (internal method).

        Raises:
            GeminiAPIError: On API or network errors.
        """
        telemetry = get_telemetry()

        # Start OTEL span for Gemini API call if enabled
        if telemetry.enabled and telemetry.tracer:
            gemini_span = telemetry.tracer.start_span(
                "gemini.api.call",
                attributes={
                    "gemini.model": "gemini-2.5-flash",
                    "gemini.has_images": bool(image_paths),
                    "gemini.num_images": len(image_paths) if image_paths else 0,
                    "gemini.temperature": temperature,
                    "gemini.max_tokens": max_tokens,
                    "gemini.attempt": attempt + 1,
                }
            )
            start_time = time.time()
        else:
            gemini_span = None

        try:
            # Build request parts: text prompts first, then images
            parts = [
                {"text": system_prompt},
                {"text": user_prompt},
            ]

            # Add images as inline base64 data
            if image_paths:
                for img_path in image_paths:
                    image_part = self._encode_image(img_path)
                    if image_part:
                        parts.append(image_part)

            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                }
            }

            req = urllib.request.Request(
                self.API_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                },
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))

                # Safely extract response text with validation
                if not isinstance(result, dict):
                    raise GeminiAPIError(
                        message="Invalid response format: expected dict",
                        error_type="parse",
                        retryable=False,
                    )

                candidates = result.get("candidates")
                if not candidates or not isinstance(candidates, list) or len(candidates) == 0:
                    raise GeminiAPIError(
                        message="No candidates in response",
                        error_type="parse",
                        retryable=False,
                    )

                content = candidates[0].get("content")
                if not content or not isinstance(content, dict):
                    raise GeminiAPIError(
                        message="No content in candidate",
                        error_type="parse",
                        retryable=False,
                    )

                parts = content.get("parts")
                if not parts or not isinstance(parts, list) or len(parts) == 0:
                    raise GeminiAPIError(
                        message="No parts in content",
                        error_type="parse",
                        retryable=False,
                    )

                text = parts[0].get("text")
                if text is None:
                    raise GeminiAPIError(
                        message="No text in response part",
                        error_type="parse",
                        retryable=False,
                    )

                response_text = str(text)

                # Record OTEL metrics
                if gemini_span:
                    duration = time.time() - start_time
                    telemetry.gemini_api_duration.record(duration)
                    gemini_span.set_attribute("gemini.response_length", len(response_text))
                    gemini_span.set_attribute("gemini.success", True)
                    gemini_span.end()

                return response_text

        except urllib.error.HTTPError as e:
            if gemini_span:
                try:
                    from opentelemetry import trace as otel_trace
                    gemini_span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, f"HTTP {e.code}"))
                    gemini_span.set_attribute("gemini.success", False)
                    gemini_span.set_attribute("gemini.error_code", e.code)
                except ImportError:
                    pass
                gemini_span.end()
            raise GeminiAPIError(
                message=f"HTTP {e.code}: {e.reason}",
                error_type="http",
                status_code=e.code,
                retryable=500 <= e.code < 600,  # Retry server errors, not client errors
            )
        except urllib.error.URLError as e:
            if gemini_span:
                try:
                    from opentelemetry import trace as otel_trace
                    gemini_span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, "Network error"))
                    gemini_span.set_attribute("gemini.success", False)
                except ImportError:
                    pass
                gemini_span.end()
            raise GeminiAPIError(
                message=f"Network error: {e.reason}",
                error_type="network",
                retryable=True,
            )
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if gemini_span:
                try:
                    from opentelemetry import trace as otel_trace
                    gemini_span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, "Parse error"))
                    gemini_span.set_attribute("gemini.success", False)
                except ImportError:
                    pass
                gemini_span.end()
            raise GeminiAPIError(
                message=f"Failed to parse Gemini response: {str(e)[:100]}",
                error_type="parse",
                retryable=False,  # Parse errors usually indicate API format changes
            )
        except Exception as e:
            if gemini_span:
                try:
                    from opentelemetry import trace as otel_trace
                    gemini_span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, str(e)))
                    gemini_span.record_exception(e)
                    gemini_span.set_attribute("gemini.success", False)
                except ImportError:
                    pass
                gemini_span.end()
            # Re-raise unexpected errors
            raise
    
    def _encode_image(self, img_path: str) -> Optional[dict]:
        """
        Encode image file as base64 inline_data for Gemini.
        
        Args:
            img_path: Path to image file.
            
        Returns:
            Dict with inline_data format, or None if encoding fails.
        """
        try:
            path = Path(img_path)
            if not path.exists():
                return None
            
            # Get MIME type from extension
            ext = path.suffix.lower().lstrip(".")
            mime_type = self.MIME_TYPES.get(ext, "image/png")
            
            # Read and encode
            with open(path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            
            return {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": img_data
                }
            }
        except (IOError, OSError):
            return None
    
    def analyze_completion(self, text: str) -> bool:
        """
        Use Gemini to analyze if text indicates task completion.
        
        Args:
            text: Text to analyze.
            
        Returns:
            True if text indicates completion.
        """
        if not text or len(text.strip()) < 10:
            return False
        
        try:
            response = self.call(
                system_prompt="""Analyze text to determine if it indicates task completion.
Respond with ONLY one word:
- YES - if the text clearly indicates task completion, success, or work is done
- NO - if the text indicates ongoing work, errors, questions, or incomplete state""",
                user_prompt=f"Does this indicate task completion?\n\n{text[:1500]}"
            )
            return response.strip().upper().startswith("YES")
        except Exception:
            return False
    
    def analyze_review_issues(self, review_result: str) -> bool:
        """
        Analyze if a code review found significant issues.
        
        Args:
            review_result: Code review output text.
            
        Returns:
            True if review found issues requiring fixes.
        """
        if not review_result or len(review_result.strip()) < 10:
            return False
        
        try:
            response = self.call(
                system_prompt="""Analyze code review output to determine if it found significant issues.
Respond with ONLY one word:
- YES - if the review found errors, bugs, or security issues that MUST be fixed
- NO - if the review passed or only has minor suggestions""",
                user_prompt=f"Does this review indicate issues needing fixes?\n\n{review_result[:2000]}"
            )
            return response.strip().upper().startswith("YES")
        except Exception:
            return False
