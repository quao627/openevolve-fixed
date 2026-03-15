"""
Claude Code CLI interface for LLMs.

Uses the `claude` CLI (with Max plan) instead of direct API calls,
allowing OpenEvolve to run on a Claude subscription instead of paying per-token.
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)

# Appended to every user prompt to ensure code output format
OUTPUT_FORMAT_INSTRUCTION = """

IMPORTANT: You must respond with the complete evolved Python program wrapped in a ```python code block. Output ONLY the code block — no explanations, no commentary before or after. The code block must contain the COMPLETE program (all imports, all functions, everything).
"""


class ClaudeCodeLLM(LLMInterface):
    """LLM interface using Claude Code CLI subprocess."""

    def __init__(self, model_cfg=None):
        self.model = getattr(model_cfg, "name", "claude-opus-4-6") or "claude-opus-4-6"
        self.max_tokens = getattr(model_cfg, "max_tokens", 16384) or 16384
        self.timeout = getattr(model_cfg, "timeout", 600) or 600

        # Map model names to Claude Code model flags
        self._model_flag = self._resolve_model_flag(self.model)

        logger.info(f"Initialized ClaudeCodeLLM with model={self.model}")

    def _resolve_model_flag(self, model_name: str) -> str:
        """Map model name to claude CLI --model flag."""
        model_map = {
            "claude-opus-4-6": "opus",
            "claude-sonnet-4-6": "sonnet",
            "claude-haiku-4-5": "haiku",
            "opus": "opus",
            "sonnet": "sonnet",
            "haiku": "haiku",
        }
        return model_map.get(model_name, model_name)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        return await self.generate_with_context(
            system_message="",
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using Claude Code CLI."""
        # Build user prompt from messages
        user_parts = []
        for msg in messages:
            content = msg.get("content", "")
            if content:
                user_parts.append(content)
        user_prompt = "\n\n".join(user_parts) + OUTPUT_FORMAT_INSTRUCTION

        # Run claude CLI with system prompt and user prompt
        response = await self._call_claude(system_message, user_prompt)
        return response

    async def _call_claude(self, system_message: str, user_prompt: str) -> str:
        """Call the claude CLI and return the response text."""
        cmd = [
            "claude",
            "-p",  # Print mode: non-interactive, read prompt from stdin
            "--model", self._model_flag,
            "--max-turns", "1",
        ]

        # Add system prompt if provided
        if system_message:
            cmd.extend(["--system-prompt", system_message])

        logger.info(f"Calling Claude Code CLI: model={self._model_flag}, prompt_len={len(user_prompt)}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=user_prompt.encode("utf-8")),
                timeout=self.timeout,
            )

            if proc.returncode != 0:
                err_msg = stderr.decode("utf-8", errors="replace")[:1000]
                logger.error(f"Claude Code CLI error (rc={proc.returncode}): {err_msg}")
                raise RuntimeError(f"Claude Code CLI failed: {err_msg}")

            response = stdout.decode("utf-8", errors="replace")

            if not response.strip():
                logger.warning("Claude Code CLI returned empty response")
                raise RuntimeError("Empty response from Claude Code CLI")

            logger.info(f"Claude Code CLI response: {len(response)} chars")
            return response

        except asyncio.TimeoutError:
            logger.error(f"Claude Code CLI timed out after {self.timeout}s")
            raise RuntimeError(f"Claude Code CLI timed out after {self.timeout}s")


def init_claude_code_client(model_cfg) -> ClaudeCodeLLM:
    """Factory function for use with LLMModelConfig.init_client."""
    return ClaudeCodeLLM(model_cfg)
