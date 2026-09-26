from __future__ import annotations
import json, sys, traceback, uuid
from dataclasses import fields
from protocol import EngineResponse, GenerateRequest
from qwen_engine import QwenEngine

engine = QwenEngine()
allowed = {f.name for f in fields(GenerateRequest)}

def emit(request_id: str, event: str, data: dict, ok: bool = True) -> None:
    print(json.dumps(EngineResponse(ok, request_id, event, data).json(), ensure_ascii=False), flush=True)

def main() -> None:
    emit("system", "ready", {"protocol": 1})
    for line in sys.stdin:
        request_id = str(uuid.uuid4())
        try:
            message = json.loads(line)
            request_id = str(message.get("id") or request_id)
            command = message.get("command")
            if command == "ping":
                emit(request_id, "pong", {})
            elif command == "component-self-test":
                result = engine.component_self_test(lambda state: emit(request_id, "progress", {"state": state}))
                emit(request_id, "completed", result)
            elif command == "self-test":
                prompt = message.get("payload", {}).get("prompt", "a simple red apple on a white background")
                if not isinstance(prompt, str):
                    raise TypeError(f"self-test prompt must be str, got {type(prompt).__name__}")
                result = engine.self_test(prompt, lambda state: emit(request_id, "progress", {"state": state}))
                emit(request_id, "completed", result)
            elif command == "generate":
                payload = {k: v for k, v in message.get("payload", {}).items() if k in allowed}
                req = GenerateRequest(**payload)
                emit(request_id, "progress", {"state": "request-validated", "request": {
                    "promptType": type(req.prompt).__name__,
                    "promptLength": len(req.prompt) if isinstance(req.prompt, str) else None,
                    "negativePromptType": type(req.negative_prompt).__name__ if req.negative_prompt is not None else "NoneType",
                    "negativePromptLength": len(req.negative_prompt) if isinstance(req.negative_prompt, str) else None,
                    "inputPath": bool(req.input_path), "width": req.width, "height": req.height,
                    "steps": req.steps, "trueCfgScale": req.true_cfg_scale,
                }})
                if not isinstance(req.prompt, str):
                    raise TypeError(f"prompt must be str, got {type(req.prompt).__name__}")
                if req.negative_prompt is not None and not isinstance(req.negative_prompt, str):
                    raise TypeError(f"negative_prompt must be str or None, got {type(req.negative_prompt).__name__}")
                result = engine.generate(req, lambda state: emit(request_id, "progress", {"state": state}))
                emit(request_id, "completed", result)
            elif command == "shutdown":
                emit(request_id, "shutdown", {})
                return
            else:
                emit(request_id, "error", {"message": f"Unknown command: {command}"}, False)
        except Exception as exc:
            emit(request_id, "error", {"message": str(exc), "trace": traceback.format_exc(limit=4)}, False)

if __name__ == "__main__":
    main()
