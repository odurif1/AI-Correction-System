"""Debug capture for prompts and API calls.

Captures all prompts and API calls for debugging purposes.
Useful for:
- Debugging grading issues
- Analyzing prompts sent to LLMs
- Understanding responses received
- Optimizing prompts
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import json


@dataclass
class DebugCall:
    """Single API call debug info."""
    timestamp: str
    provider: str
    model: str
    call_type: str  # "vision" or "text"

    # Full prompt
    prompt: str

    # Images (paths for vision calls)
    images: List[str] = field(default_factory=list)

    # Response
    response: Optional[str] = None
    response_json: Optional[Dict] = None

    # Metadata
    duration_ms: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None
    llm_id: int = 0  # 0=unknown, 1=LLM1, 2=LLM2


class DebugCapture:
    """Captures all prompts and API calls for debugging."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calls: List[DebugCall] = []
        self.enabled = True

    def capture_call(
        self,
        provider: str,
        model: str,
        call_type: str,
        prompt: str,
        images: List[str] = None,
        response: str = None,
        response_json: Dict = None,
        duration_ms: float = 0,
        tokens: Dict = None,
        error: str = None,
        llm_id: int = 0
    ):
        """Capture an API call."""
        if not self.enabled:
            return

        call = DebugCall(
            timestamp=datetime.now().isoformat(),
            provider=provider,
            model=model,
            call_type=call_type,
            prompt=prompt,
            images=images or [],
            response=response,
            response_json=response_json,
            duration_ms=duration_ms,
            prompt_tokens=tokens.get("prompt", 0) if tokens else 0,
            completion_tokens=tokens.get("completion", 0) if tokens else 0,
            error=error,
            llm_id=llm_id
        )
        self.calls.append(call)

    def save(self, filename: str = "debug_log.json") -> Path:
        """Save all captured calls to file."""
        output_path = self.output_dir / filename
        data = {
            "capture_time": datetime.now().isoformat(),
            "total_calls": len(self.calls),
            "calls": [
                {
                    "timestamp": c.timestamp,
                    "provider": c.provider,
                    "model": c.model,
                    "call_type": c.call_type,
                    "prompt": c.prompt,
                    "images": c.images,
                    "response": c.response,
                    "response_json": c.response_json,
                    "duration_ms": c.duration_ms,
                    "tokens": {
                        "prompt": c.prompt_tokens,
                        "completion": c.completion_tokens
                    },
                    "error": c.error,
                    "llm_id": c.llm_id
                }
                for c in self.calls
            ]
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return output_path

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of captured calls."""
        return {
            "total_calls": len(self.calls),
            "vision_calls": sum(1 for c in self.calls if c.call_type == "vision"),
            "text_calls": sum(1 for c in self.calls if c.call_type == "text"),
            "total_tokens": sum(c.prompt_tokens + c.completion_tokens for c in self.calls),
            "errors": sum(1 for c in self.calls if c.error)
        }


# Global debug capture instance
_debug_capture: Optional[DebugCapture] = None


def init_debug(output_dir: Path) -> DebugCapture:
    """Initialize global debug capture."""
    global _debug_capture
    _debug_capture = DebugCapture(output_dir)
    return _debug_capture


def get_debug_capture() -> Optional[DebugCapture]:
    """Get the current debug capture instance."""
    return _debug_capture


def is_debug_enabled() -> bool:
    """Check if debug mode is enabled."""
    return _debug_capture is not None and _debug_capture.enabled


def save_debug_log(filename: str = "debug_log.json") -> Optional[Path]:
    """Save the debug log if debug mode is enabled."""
    global _debug_capture
    if _debug_capture:
        return _debug_capture.save(filename)
    return None
