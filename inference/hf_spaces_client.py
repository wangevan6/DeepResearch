#!/usr/bin/env python3
"""
HuggingFace Spaces API Client for Tongyi DeepResearch

This module provides a client wrapper for calling the Tongyi-DeepResearch model
via HuggingFace Spaces API with proper error handling, retries, and logging.

The client supports:
- HuggingFace Spaces API (primary)
- OpenRouter API (fallback)
- Exponential backoff retry logic
- Rate limiting handling
- Detailed logging and error reporting
"""

import os
import time
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HFSpacesClient:
    """
    Client for Tongyi-DeepResearch via HuggingFace Spaces API.

    This client handles API calls with proper retry logic, error handling,
    and supports both HF Spaces and OpenRouter endpoints.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "alibaba/tongyi-deepresearch-30b-a3b",
        timeout: float = 600.0,
        use_openrouter: bool = False
    ):
        """
        Initialize the HF Spaces client.

        Args:
            api_key: API key (from env if not provided)
            api_base: API base URL (from env if not provided)
            model: Model identifier
            timeout: Request timeout in seconds
            use_openrouter: If True, use OpenRouter instead of HF Spaces
        """

        # Determine API endpoint
        if use_openrouter:
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            self.api_base = "https://openrouter.ai/api/v1"
            self.model = model
            logger.info("🔄 Using OpenRouter API")
        else:
            self.api_key = api_key or os.getenv("HF_SPACES_API_KEY") or os.getenv("HF_TOKEN")
            self.api_base = api_base or os.getenv("HF_SPACES_ENDPOINT")

            # Default HF Spaces endpoint if not provided
            if not self.api_base:
                self.api_base = "https://alibaba-nlp-tongyi-deepresearch.hf.space/api/v1"
                logger.warning(f"⚠️  No HF_SPACES_ENDPOINT set, using default: {self.api_base}")

            self.model = model
            logger.info(f"🚀 Using HuggingFace Spaces API: {self.api_base}")

        if not self.api_key:
            raise ValueError(
                "No API key found! Please set HF_SPACES_API_KEY or HF_TOKEN environment variable."
            )

        # Initialize OpenAI client (HF Spaces uses OpenAI-compatible format)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=timeout
        )

        self.timeout = timeout
        self.use_openrouter = use_openrouter

        logger.info(f"✅ Client initialized: model={self.model}, timeout={timeout}s")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.85,
        top_p: float = 0.95,
        max_tokens: int = 10000,
        presence_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        max_retries: int = 10
    ) -> Dict[str, Any]:
        """
        Make a chat completion API call with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            presence_penalty: Presence penalty
            stop: Stop sequences
            max_retries: Maximum number of retry attempts

        Returns:
            API response dict

        Raises:
            Exception: If all retry attempts fail
        """

        if stop is None:
            stop = ["\n<tool_response>", "<tool_response>"]

        last_exception = None

        for attempt in range(max_retries):
            try:
                logger.debug(f"🔄 Attempt {attempt + 1}/{max_retries}")

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    presence_penalty=presence_penalty,
                    stop=stop
                )

                # Log success
                logger.debug(f"✅ API call successful (attempt {attempt + 1})")

                return response

            except Exception as e:
                last_exception = e
                error_msg = str(e)

                logger.warning(f"⚠️  API call failed (attempt {attempt + 1}/{max_retries}): {error_msg}")

                # Check for rate limiting
                if "rate limit" in error_msg.lower() or "429" in error_msg:
                    wait_time = min(2 ** attempt, 60)  # Cap at 60 seconds
                    logger.info(f"⏳ Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                # Check for server errors (5xx)
                if "500" in error_msg or "502" in error_msg or "503" in error_msg:
                    wait_time = min(2 ** attempt, 30)
                    logger.info(f"⏳ Server error. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                # Check for timeout
                if "timeout" in error_msg.lower():
                    logger.warning(f"⏱️  Request timed out after {self.timeout}s")
                    # Increase wait time for timeouts
                    wait_time = min(2 ** (attempt + 1), 60)
                    logger.info(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                # For other errors, wait with exponential backoff
                wait_time = min(2 ** attempt, 30)
                logger.info(f"⏳ Retrying in {wait_time}s...")
                time.sleep(wait_time)

        # All retries exhausted
        error_msg = f"❌ API call failed after {max_retries} attempts. Last error: {last_exception}"
        logger.error(error_msg)
        raise Exception(error_msg)

    def extract_content(self, response: Any) -> str:
        """
        Extract content from API response.

        Args:
            response: API response object

        Returns:
            Extracted text content
        """
        try:
            # Handle OpenAI-style response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                choice = response.choices[0]

                # Extract message content
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content

                    # Handle reasoning content (for o1/o3 models)
                    if hasattr(choice.message, 'reasoning_content') and choice.message.reasoning_content:
                        reasoning = choice.message.reasoning_content
                        content = reasoning + "\n" + content if content else reasoning

                    return content or ""

            # Fallback: try to extract from dict
            if isinstance(response, dict):
                if 'choices' in response and len(response['choices']) > 0:
                    message = response['choices'][0].get('message', {})
                    return message.get('content', '')

            logger.warning("⚠️  Unexpected response format")
            return str(response)

        except Exception as e:
            logger.error(f"❌ Error extracting content: {e}")
            return ""

    def get_usage_stats(self, response: Any) -> Dict[str, int]:
        """
        Extract token usage statistics from response.

        Args:
            response: API response object

        Returns:
            Dict with 'prompt_tokens', 'completion_tokens', 'total_tokens'
        """
        try:
            if hasattr(response, 'usage'):
                usage = response.usage
                return {
                    'prompt_tokens': getattr(usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(usage, 'completion_tokens', 0),
                    'total_tokens': getattr(usage, 'total_tokens', 0)
                }

            # Fallback for dict response
            if isinstance(response, dict) and 'usage' in response:
                usage = response['usage']
                return {
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0)
                }

        except Exception as e:
            logger.warning(f"⚠️  Could not extract usage stats: {e}")

        return {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}


# Factory function for easy client creation
def create_client(use_openrouter: bool = False, **kwargs) -> HFSpacesClient:
    """
    Factory function to create an HF Spaces client.

    Args:
        use_openrouter: If True, use OpenRouter API instead of HF Spaces
        **kwargs: Additional arguments passed to HFSpacesClient

    Returns:
        Configured HFSpacesClient instance
    """
    return HFSpacesClient(use_openrouter=use_openrouter, **kwargs)


# Test function
def test_client():
    """Test the HF Spaces client with a simple query."""

    print("=" * 70)
    print("Testing HF Spaces API Client")
    print("=" * 70)
    print()

    try:
        # Create client
        client = create_client(use_openrouter=False)

        # Test message
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer briefly."}
        ]

        print("📤 Sending test request...")
        print(f"   Messages: {len(messages)}")
        print()

        # Make API call
        response = client.chat_completion(messages, max_tokens=100)

        # Extract content
        content = client.extract_content(response)
        usage = client.get_usage_stats(response)

        print("✅ Test successful!")
        print()
        print(f"📥 Response: {content}")
        print()
        print(f"📊 Token usage:")
        print(f"   - Prompt: {usage['prompt_tokens']}")
        print(f"   - Completion: {usage['completion_tokens']}")
        print(f"   - Total: {usage['total_tokens']}")
        print()

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        print()
        print("Troubleshooting:")
        print("  - Check that HF_SPACES_API_KEY or HF_TOKEN is set in .env")
        print("  - Verify the API endpoint is correct")
        print("  - Try using OpenRouter instead: use_openrouter=True")
        return False


if __name__ == "__main__":
    test_client()
