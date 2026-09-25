from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any

@dataclass
class GenerateRequest:
    prompt: str
    width: int = 2048
    height: int = 2048
    steps: int = 40
    seed: int = -1
    negative_prompt: str | None = None
    true_cfg_scale: float = 1.0
    output_path: str = "outputs/generated.png"

@dataclass
class EngineResponse:
    ok: bool
    request_id: str
    event: str
    data: dict[str, Any]

    def json(self) -> dict[str, Any]:
        return asdict(self)
