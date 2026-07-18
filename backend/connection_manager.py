from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, ws: WebSocket):
        await ws.accept()
        self.active[session_id] = ws

    def disconnect(self, session_id: str, ws: WebSocket | None = None):
        # Only disconnect if the ws matches the current active one
        # (prevents a stale connection from removing a newer one)
        if ws is None or self.active.get(session_id) is ws:
            self.active.pop(session_id, None)

    async def push(self, session_id: str, payload: dict):
        ws = self.active.get(session_id)
        if ws:
            await ws.send_json(payload)

    async def push_bytes(self, session_id: str, payload: bytes):
        ws = self.active.get(session_id)
        if ws:
            await ws.send_bytes(payload)
