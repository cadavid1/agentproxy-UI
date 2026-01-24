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
    ) -> str:
        """
        Make API call to Gemini with optional images.

        Args:
            system_prompt: System instruction for the model.
            user_prompt: User prompt/question.
            image_paths: Optional list of image file paths to include.
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum output tokens.

        Returns:
            Model's text response.
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
                }
            )
            start_time = time.time()
        else:
            gemini_span = None

        try:
            url = f"{self.API_URL}?key={self.api_key}"

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
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                response_text = result["candidates"][0]["content"]["parts"][0]["text"]

                # Record OTEL metrics
                if gemini_span:
                    duration = time.time() - start_time
                    telemetry.gemini_api_duration.record(duration)
                    gemini_span.set_attribute("gemini.response_length", len(response_text))
                    gemini_span.end()

                return response_text

        except urllib.error.HTTPError as e:
            if gemini_span:
                try:
                    from opentelemetry import trace as otel_trace
                    gemini_span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, f"HTTP {e.code}"))
                except ImportError:
                    pass
                gemini_span.end()
            return f"[Gemini API Error: {e.code} {e.reason}]"
        except urllib.error.URLError as e:
            if gemini_span:
                try:
                    from opentelemetry import trace as otel_trace
                    gemini_span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, "Network error"))
                except ImportError:
                    pass
                gemini_span.end()
            return f"[Gemini Network Error: {e.reason}]"
        except Exception as e:
            if gemini_span:
                try:
                    from opentelemetry import trace as otel_trace
                    gemini_span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, str(e)))
                    gemini_span.record_exception(e)
                except ImportError:
                    pass
                gemini_span.end()
            return f"[Gemini Error: {str(e)[:100]}]"
    
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
