"""
WebSocket manager for real-time progress updates.

Provides connection management and message broadcasting for grading sessions.
"""

import asyncio
import json
from typing import Dict, Set, Optional, Any
from fastapi import WebSocket
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for grading sessions.

    Each session can have multiple connected clients that receive
    real-time progress updates.
    """

    def __init__(self):
        # Map of session_id -> set of websocket connections
        self._connections: Dict[str, Set[WebSocket]] = {}
        # Map of session_id -> progress callback
        self._callbacks: Dict[str, Any] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        """
        Accept a new WebSocket connection for a session.

        Args:
            websocket: The WebSocket connection
            session_id: The session to subscribe to
        """
        await websocket.accept()

        if session_id not in self._connections:
            self._connections[session_id] = set()

        self._connections[session_id].add(websocket)
        logger.info(f"WebSocket connected for session {session_id}")

    def disconnect(self, websocket: WebSocket, session_id: str):
        """
        Remove a WebSocket connection.

        Args:
            websocket: The WebSocket to disconnect
            session_id: The session it was subscribed to
        """
        if session_id in self._connections:
            self._connections[session_id].discard(websocket)
            if not self._connections[session_id]:
                del self._connections[session_id]

        logger.info(f"WebSocket disconnected for session {session_id}")

    async def broadcast(self, session_id: str, message: dict):
        """
        Broadcast a message to all connections for a session.

        Args:
            session_id: The session to broadcast to
            message: The message to send (will be JSON encoded)
        """
        if session_id not in self._connections:
            return

        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()

        message_json = json.dumps(message, ensure_ascii=False)
        disconnected = set()

        for websocket in self._connections[session_id]:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.warning(f"Failed to send to websocket: {e}")
                disconnected.add(websocket)

        # Clean up disconnected websockets
        for ws in disconnected:
            self.disconnect(ws, session_id)

    async def broadcast_event(self, session_id: str, event_type: str, data: dict):
        """
        Broadcast a typed event to all connections.

        Args:
            session_id: The session to broadcast to
            event_type: The type of event (e.g., 'copy_start', 'question_done')
            data: The event data
        """
        await self.broadcast(session_id, {"type": event_type, **data})

    def get_connection_count(self, session_id: str) -> int:
        """Get the number of active connections for a session."""
        return len(self._connections.get(session_id, set()))

    def create_progress_callback(self, session_id: str):
        """
        Create a progress callback function for the grading orchestrator.

        The returned function can be passed to GradingSessionOrchestrator.grade_all()
        to send real-time progress updates via WebSocket.

        Args:
            session_id: The session to broadcast to

        Returns:
            Async callback function
        """
        manager = self

        async def callback(event_type: str, data: dict):
            await manager.broadcast_event(session_id, event_type, data)

        return callback


# Global connection manager instance
manager = ConnectionManager()
