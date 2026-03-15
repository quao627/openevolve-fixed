"""
Claude Code CLI interface for LLMs.

Uses the `claude` CLI (with Max plan) instead of direct API calls,
allowing OpenEvolve to run on a Claude subscription instead of paying per-token.
"""

import asyncio
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


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
        # Build the full prompt: system message + user messages
        parts = []
        if system_message:
            parts.append(system_message)
        for msg in messages:
            content = msg.get("content", "")
            if content:
                parts.append(content)
        full_prompt = "\n\n".join(parts)

        # Run claude CLI in non-interactive mode
        response = await self._call_claude(full_prompt)
        return response

    async def _call_claude(self, prompt: str) -> str:
        """Call the claude CLI and return the response text."""
        # Write prompt to a temp file to avoid shell escaping issues
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="oe_prompt_"
        ) as f:
            f.write(prompt)
            prompt_file = f.name

        try:
            cmd = [
                "claude",
                "-p",  # Print mode: non-interactive, read from stdin
                "--model", self._model_flag,
                "--max-turns", "1",
            ]

            logger.info(f"Calling Claude Code CLI: model={self._model_flag}")

            # Read prompt from file and pipe to claude
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
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
        finally:
            Path(prompt_file).unlink(missing_ok=True)


def init_claude_code_client(model_cfg) -> ClaudeCodeLLM:
    """Factory function for use with LLMModelConfig.init_client."""
    return ClaudeCodeLLM(model_cfg)
