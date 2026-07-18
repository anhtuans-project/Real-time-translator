import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .connection_manager import ConnectionManager
from .engine_factory import build_engines, Engines
from .session_state import SessionState

# Load .env (e.g. ASR_REMOTE_URL) before build_engines() reads env vars.
# Explicit repo-root path + override=True so .env always wins (regardless of
# CWD or any stale OS env var from a previous shell `set`/`export`).
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global engines - loaded ONCE at startup
engines: Engines | None = None
manager = ConnectionManager()
# Pipeline sống theo session_id: 1 session_id -> 1 SessionState. WS mới đến sẽ
# supersede pipeline cũ (xem ws_endpoint) -> không bao giờ có 2 pipeline song song.
sessions: dict[str, SessionState] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engines
    import os
    remote = os.getenv("ASR_REMOTE_URL", "")
    logger.info("Loading AI engines at startup... (ASR_REMOTE_URL=%s)",
                remote if remote else "<empty -> local CPU ASR>")
    engines = build_engines()
    logger.info("AI engines loaded successfully.")
    app.state.startup_complete = True
    yield
    logger.info("Shutting down...")
    app.state.startup_complete = False

app = FastAPI(lifespan=lifespan)
app.state.startup_complete = False

@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)

    logger.info("[ws_endpoint] startup_complete=%s, engines=%s",
                 app.state.startup_complete, engines is not None)

    if engines is None:
        logger.error("Engines not loaded!")
        await websocket.close(code=1011, reason="Server not ready")
        return

    # Supersede: nếu session_id đã có pipeline sống (reload/StrictMode/reconnect để
    # lại WS cũ), huỷ pipeline cũ TRƯỚC khi tạo mới. finalize=False -> không chạm
    # shared RemoteASR/MT, không đẩy utterance rác lên frontend mới. manager.connect
    # đã đóng WS cũ -> ws_endpoint cũ sẽ thoát và gọi shutdown() (idempotent, no-op).
    old = sessions.get(session_id)
    if old is not None:
        logger.info("[%s] New WS supersedes existing session -> killing old pipeline.", session_id)
        try:
            await old.shutdown(finalize=False)
        except Exception as e:
            logger.warning("[%s] Old session shutdown error: %s", session_id, e)

    session = SessionState(session_id, manager, engines, languages=("vi", "en"))
    sessions[session_id] = session

    chunk_count = 0
    try:
        while True:
            msg = await websocket.receive()

            # Handle disconnects
            if msg.get("type") == "websocket.disconnect":
                logger.info("[%s] WebSocket disconnected (received %d audio chunks)", session_id, chunk_count)
                break

            if "bytes" in msg:
                chunk_count += 1
                if chunk_count <= 3 or chunk_count % 100 == 0:
                    logger.info("[%s] Audio chunk #%d (%d bytes)", session_id, chunk_count, len(msg["bytes"]))
                # Binary audio data -> push to processing queue
                await session.enqueue_audio(msg["bytes"])
            elif "text" in msg:
                # JSON control messages
                try:
                    data = json.loads(msg["text"])
                    logger.info("[%s] Control message: %s", session_id, data.get("type"))
                    await session.on_control(data)
                except json.JSONDecodeError:
                    await manager.push(session_id, {"type": "error", "message": "Invalid JSON"})

    except WebSocketDisconnect:
        pass
    finally:
        # Chỉ pop nếu session này vẫn là của mình (tránh xóa session mới hơn).
        if sessions.get(session_id) is session:
            sessions.pop(session_id, None)
        await session.shutdown()
        manager.disconnect(session_id, websocket)

# Serve frontend static files if dist directory exists
frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    @app.get("/{catchall:path}")
    async def serve_frontend(catchall: str):
        file_path = frontend_dist / catchall
        if catchall and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
