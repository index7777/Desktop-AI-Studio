# AI Engine

Local JSON-lines sidecar for Desktop AI Studio. The UI/Tauri layer stays independent from model internals.

## Protocol

Send one JSON object per stdin line. Responses are one JSON object per stdout line.

Example request:

```json
{"id":"demo","command":"generate","payload":{"prompt":"台北雨夜中的霓虹招牌","width":2048,"height":2048,"steps":40,"seed":42}}
```

Commands: `ping`, `generate`, `shutdown`.

The first generation lazily loads `Qwen/Qwen-Image-2.1`. Model location can be overridden with `AI_STUDIO_MODEL`.
