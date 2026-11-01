"""Apron adapter: Hermes as a bot client of an Apron chat server (https://apron.chat/protocol.md).

config.yaml ``platforms.apron.extra``: ``url`` (``wss://host/ws``), ``ca_file`` (PEM bundle; default the
system trust store), ``users`` (Apron user_id -> toolsets; anyone else is ignored). Env: ``APRON_TOKEN``.
"""

import asyncio
import contextlib
import json
import logging
import random
import ssl
import uuid
from typing import Any, Optional

import websockets
from websockets.exceptions import WebSocketException

from gateway.config import Platform, PlatformConfig
from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret
from gateway.platforms.access_policy_mixin import OwnAccessPolicyMixin
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.helpers import MessageDeduplicator, cancel_task

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 32000  # well under the 256 KiB request cap
REQUEST_TIMEOUT = 30.0
RETRY_INITIAL, RETRY_MAX = 2.0, 60.0
DENIED, RETRY_AFTER = -32001, -32002


class ApronError(Exception):
    def __init__(self, error: dict):
        super().__init__(error.get("message") or "Apron error")
        self.code = error.get("code")
        self.retry_after = (error.get("data") or {}).get("retry_after") if self.code == RETRY_AFTER else None


# What opening and signing in can raise: network/TLS, the WebSocket, a refused or malformed reply.
_OPEN_ERRORS = (ApronError, OSError, TimeoutError, WebSocketException, ValueError, KeyError)


def _parse_users(raw: Any) -> dict[str, list[str]]:
    users = {}
    for user_id, toolsets in (raw.items() if isinstance(raw, dict) else ()):
        names = [str(t).strip() for t in toolsets if str(t).strip()] if isinstance(toolsets, list) else []
        if names:
            users[str(user_id)] = names
        else:
            # An empty override would fall back to platform_toolsets.apron, so refuse rather than widen.
            logger.warning("Apron: user %s has no toolsets list; ignoring them", user_id)
    return users


def _is_dm(room: dict) -> bool:
    return bool(room.get("private")) and len(room.get("members") or ()) <= 2


class ApronAdapter(OwnAccessPolicyMixin, BasePlatformAdapter):
    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    ALLOW_ALL_ENV_PREFIX = "APRON"
    # Intake drops anyone not in ``users``, so gateway authz may trust what reaches it.
    _dm_policy = _group_policy = "allowlist"

    def __init__(self, config: PlatformConfig, **kwargs):
        super().__init__(config=config, platform=Platform("apron"))
        extra = config.extra or {}
        self.url = str(extra.get("url") or "")
        self.ca_file = extra.get("ca_file") or None
        self._users = _parse_users(extra.get("users"))
        self._allow_from = self._users
        self._token = _get_scoped_secret("APRON_TOKEN", "") or ""
        self._ws = None
        self._task: Optional[asyncio.Task] = None
        self._pending: dict[str, asyncio.Future] = {}
        self._bot_id = ""
        self._ignored: set[str] = set()
        self._dedup = MessageDeduplicator()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self.url or not self._token:
            self._set_fatal_error("config_missing", "Apron needs platforms.apron.extra.url and APRON_TOKEN",
                                  retryable=False)
            return False
        if not self._acquire_platform_lock("apron-bot-token", self._token, "Apron bot token"):
            return False
        try:
            ws = await self._open()
        except _OPEN_ERRORS as e:
            self._release_platform_lock()
            denied = isinstance(e, ApronError) and e.code == DENIED
            self._set_fatal_error("auth_failed" if denied else "connect_failed", f"Apron: {e}", retryable=not denied)
            return False
        self._mark_connected()
        self._task = asyncio.create_task(self._run(ws))
        logger.info("Apron: connected to %s as %s", self.url, self._bot_id)
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        await cancel_task(self._task)
        self._release_platform_lock()

    async def _open(self):
        ctx = ssl.create_default_context(cafile=self.ca_file) if self.ca_file else None
        ws = await websockets.connect(self.url, ssl=ctx, open_timeout=REQUEST_TIMEOUT)
        try:
            # ``auth`` may precede the server frame; skip everything until its reply.
            await ws.send(json.dumps({"id": "auth", "method": "auth",
                                      "params": {"scheme": "token", "token": self._token, "agent": "hermes"}}))
            while (reply := json.loads(await asyncio.wait_for(ws.recv(), REQUEST_TIMEOUT))).get("id") != "auth":
                pass
            if "error" in reply:
                raise ApronError(reply["error"])
            self._bot_id = reply["result"]["you"]["user_id"]
        except BaseException:
            await ws.close()
            raise
        self._ws = ws
        return ws

    async def _run(self, ws) -> None:
        delay = RETRY_INITIAL
        while True:
            if ws is not None:
                delay = RETRY_INITIAL
                await self._serve(ws)
            if not self._running:
                return
            await asyncio.sleep(delay * (1 + 0.2 * random.random()))
            delay = min(delay * 2, RETRY_MAX)
            try:
                ws = await self._open()
                logger.info("Apron: reconnected")
            except ApronError as e:
                if e.code == DENIED:  # the spec says never to retry a refused sign-in by itself
                    self._set_fatal_error("auth_failed", f"Apron: {e}", retryable=False)
                    await self._notify_fatal_error()
                    return
                ws = None
                logger.warning("Apron: reconnect failed: %s", e)
            except _OPEN_ERRORS as e:
                ws = None
                logger.warning("Apron: reconnect failed: %s", e)

    async def _serve(self, ws) -> None:
        try:
            async for raw in ws:
                with contextlib.suppress(ValueError):
                    self._on_frame(json.loads(raw))
        except WebSocketException as e:
            logger.warning("Apron: connection lost: %s", e)
        finally:
            self._ws = None
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("Apron connection closed"))

    def _on_frame(self, frame: dict) -> None:
        fut = self._pending.get(frame.get("id"))
        if fut is not None and not fut.done():
            if "error" in frame:
                fut.set_exception(ApronError(frame["error"]))
            else:
                fut.set_result(frame.get("result") or {})
        elif frame.get("method") == "message":
            # Classifying the room needs a request whose reply this same reader must deliver.
            task = asyncio.create_task(self._on_message(frame.get("params") or {}))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _request(self, method: str, params: dict) -> dict:
        if self._ws is None:
            raise ConnectionError("Apron not connected")
        # Servers deduplicate by (user, id) across reconnects, so ids must never repeat.
        req_id = uuid.uuid4().hex
        self._pending[req_id] = fut = asyncio.get_running_loop().create_future()
        try:
            await self._ws.send(json.dumps({"id": req_id, "method": method, "params": params}))
            return await asyncio.wait_for(fut, REQUEST_TIMEOUT)
        finally:
            self._pending.pop(req_id, None)

    async def _room(self, room_id: str) -> dict:
        try:
            result = await self._request("room_list", {"room_id": room_id, "members": True})
        except ApronError:
            return {}
        return (result.get("joined") or result.get("not_joined") or [{}])[0]

    async def _on_message(self, msg: dict) -> None:
        sender = msg.get("from") or {}
        user_id = str(sender.get("user_id") or "")
        message_id = msg.get("message_id")
        # Edits and moves are later snapshots of a message (log_id > message_id): only creations are input.
        if not message_id or msg.get("log_id") != message_id or msg.get("deleted") or user_id == self._bot_id:
            return
        if user_id not in self._users:
            if user_id not in self._ignored:
                self._ignored.add(user_id)
                logger.warning("Apron: ignoring messages from %s (not in platforms.apron.extra.users)", user_id)
            return
        if self._dedup.is_duplicate(message_id):
            return
        body = msg.get("body") or {}
        room_id = str(msg.get("room_id") or "")
        room = await self._room(room_id)
        dm = _is_dm(room)
        if not dm and self._bot_id not in (body.get("mentions") or ()):
            return
        source = self.build_source(
            chat_id=room_id, chat_name=room.get("title") or room_id, chat_type="dm" if dm else "group",
            user_id=user_id, user_name=sender.get("name") or user_id)
        await self.handle_message(MessageEvent(
            source=source, text=body.get("text") or "", message_type=MessageType.TEXT,
            message_id=message_id, raw_message=msg))

    def toolsets_for_source(self, source) -> Optional[list[str]]:
        return self._users.get(str(source.user_id or ""))

    async def _post(self, params: dict) -> SendResult:
        try:
            result = await self._request("message", params)
        except ApronError as e:
            return SendResult(success=False, error=str(e), retryable=e.retry_after is not None,
                              retry_after=e.retry_after)
        except (ConnectionError, TimeoutError, WebSocketException) as e:
            return SendResult(success=False, error=str(e) or "Apron request timed out", retryable=True)
        return SendResult(success=True, message_id=result.get("message_id"))

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[dict[str, Any]] = None) -> SendResult:
        params = {"room_id": chat_id, "body": {"text": content, "format": "markdown"}}
        if reply_to:
            params["reply_to"] = {"message_id": reply_to}
        return await self._post(params)

    async def edit_message(self, chat_id: str, message_id: str, content: str, *, finalize: bool = False) -> SendResult:
        return await self._post({"message_id": message_id, "room_id": chat_id,
                                 "body": {"text": content, "format": "markdown"}})

    async def _activity(self, chat_id: str, typing: int) -> None:
        if self._ws is not None:
            with contextlib.suppress(WebSocketException):
                await self._ws.send(json.dumps({"method": "activity", "params": {"room_id": chat_id, "typing": typing}}))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        await self._activity(chat_id, 8)

    async def stop_typing(self, chat_id: str) -> None:
        await self._activity(chat_id, 0)

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        room = await self._room(chat_id)
        return {"chat_id": chat_id, "name": room.get("title") or chat_id, "type": "dm" if _is_dm(room) else "group"}


def _configured(config) -> bool:
    return bool((getattr(config, "extra", None) or {}).get("url") and _get_scoped_secret("APRON_TOKEN"))


def register(ctx) -> None:
    ctx.register_platform(
        name="apron", label="Apron", adapter_factory=ApronAdapter, check_fn=lambda: True,
        validate_config=_configured, is_connected=_configured, required_env=["APRON_TOKEN"],
        max_message_length=MAX_MESSAGE_LENGTH, emoji="💬",
        platform_hint=("You are chatting over Apron, a small group chat. Messages render CommonMark "
                       "markdown. In shared rooms you only see messages that mention you."))
