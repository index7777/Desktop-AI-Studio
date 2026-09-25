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
            elif command == "generate":
                payload = {k: v for k, v in message.get("payload", {}).items() if k in allowed}
                req = GenerateRequest(**payload)
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
