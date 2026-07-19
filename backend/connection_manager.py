from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, ws: WebSocket):
        await ws.accept()
        # Một session_id chỉ giữ 1 WS sống. WS mới đến -> đóng WS cũ (nếu còn) để
        # ws_endpoint cũ thoát khỏi receive() và shutdown pipeline cũ, tránh 2
        # pipeline song song trên cùng mic + 2 RemoteASR connect GPU server.
        prev = self.active.get(session_id)
        self.active[session_id] = ws
        if prev is not None and prev is not ws:
            try:
                await prev.close(code=1000, reason="Superseded by new connection")
            except Exception:
                pass

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
