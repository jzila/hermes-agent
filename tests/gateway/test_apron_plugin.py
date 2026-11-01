"""Apron platform plugin against an in-process fake Apron server: routing, default-deny of unlisted
users, per-user toolsets through the gateway's ``toolsets_for_source`` seam, and reconnect."""

from __future__ import annotations

import contextlib
import json

import pytest
import websockets

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource
from hermes_cli.tools_config import _get_platform_tools
from tests.gateway._plugin_adapter_loader import load_plugin_adapter
from tests.waiting import async_wait_until as _until

_plugin = load_plugin_adapter("apron")

USERS = {"john": ["terminal", "file", "todo"], "amanda": ["todo"]}
ROOMS = {
    "dm-john": {"room_id": "dm-john", "private": True, "members": [{"user_id": "hermes"}, {"user_id": "john"}]},
    "family": {"room_id": "family", "title": "Family",
               "members": [{"user_id": "hermes"}, {"user_id": "john"}, {"user_id": "amanda"}]},
}


class FakeApron:
    """Just enough of aprond: token auth, ``room_list`` by ``room_id``, and posting messages."""

    def __init__(self):
        self.conns, self.posted, self.methods = [], [], []
        self.auths = 0
        self.token = "good"

    async def handler(self, ws):
        self.conns.append(ws)
        await ws.send(json.dumps({"method": "server", "params": {"apron": 10, "auth": ["token"]}}))
        async for raw in ws:
            req = json.loads(raw)
            method, params = req.get("method"), req.get("params") or {}
            self.methods.append(method)
            if method == "auth":
                self.auths += 1
                reply = ({"result": {"you": {"user_id": "hermes"}}} if params.get("token") == self.token
                         else {"error": {"code": -32001, "message": "Invalid token"}})
            elif method == "room_list":
                reply = {"result": {"joined": [ROOMS[params["room_id"]]]}}
            elif method == "message":
                self.posted.append(params)
                reply = {"result": {"message_id": params.get("message_id") or f"9{len(self.posted)}"}}
            else:
                continue
            await ws.send(json.dumps({"id": req["id"], **reply}))

    async def push(self, room_id, user_id, text, *, message_id="100", log_id=None, mentions=()):
        await self.conns[-1].send(json.dumps({"method": "message", "params": {
            "message_id": message_id, "log_id": log_id or message_id, "room_id": room_id,
            "from": {"user_id": user_id, "name": user_id.title()},
            "body": {"text": text, "format": "plain", "mentions": list(mentions)}}}))


@contextlib.asynccontextmanager
async def _apron(monkeypatch):
    monkeypatch.setenv("APRON_TOKEN", "good")
    monkeypatch.setattr(_plugin, "RETRY_INITIAL", 0.01)
    fake = FakeApron()
    async with websockets.serve(fake.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        adapter = _plugin.ApronAdapter(PlatformConfig(enabled=True, extra={"url": f"ws://127.0.0.1:{port}", "users": USERS}))
        adapter.inbound = []

        async def _capture(event):
            adapter.inbound.append(event)

        adapter.handle_message = _capture
        try:
            yield fake, adapter
        finally:
            await adapter.disconnect()


@pytest.mark.asyncio
async def test_dm_message_is_routed_with_sender_and_room(monkeypatch):
    async with _apron(monkeypatch) as (fake, adapter):
        assert await adapter.connect() is True
        await fake.push("dm-john", "john", "hello")
        await _until(lambda: adapter.inbound)
        src = adapter.inbound[0].source
        assert (src.platform.value, src.chat_id, src.chat_type, src.user_id, src.user_name) == (
            "apron", "dm-john", "dm", "john", "John")
        assert adapter.inbound[0].text == "hello"


@pytest.mark.asyncio
async def test_shared_room_needs_a_mention_and_edits_and_own_messages_are_not_input(monkeypatch):
    async with _apron(monkeypatch) as (fake, adapter):
        await adapter.connect()
        await fake.push("family", "amanda", "chatting with John", message_id="101")
        await fake.push("dm-john", "john", "an edit", message_id="102", log_id="103")
        await fake.push("dm-john", "hermes", "my own reply", message_id="104")
        await fake.push("family", "amanda", "@hermes add milk", message_id="105", mentions=["hermes"])
        await _until(lambda: adapter.inbound)
        assert [(e.text, e.source.chat_type) for e in adapter.inbound] == [("@hermes add milk", "group")]


@pytest.mark.asyncio
async def test_unlisted_user_is_dropped_before_anything_is_sent(monkeypatch, caplog):
    async with _apron(monkeypatch) as (fake, adapter):
        await adapter.connect()
        for n in range(3):
            await fake.push("dm-john", "mallory", "hi", message_id=f"20{n}")
        await fake.push("dm-john", "john", "after", message_id="210")
        await _until(lambda: adapter.inbound)
        assert [e.source.user_id for e in adapter.inbound] == ["john"]
        # No room lookup, no reply, and one log line however many messages arrive.
        assert fake.methods.count("room_list") == 1 and fake.posted == []
        assert sum("mallory" in r.getMessage() for r in caplog.records) == 1


def test_gateway_authorizes_only_listed_users(monkeypatch):
    from gateway.run import GatewayRunner
    for key in ("GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS", "APRON_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    platform = Platform("apron")
    cfg = PlatformConfig(enabled=True, extra={"url": "ws://x", "users": {**USERS, "nobody": []}})
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={platform: cfg})
    runner.adapters = {platform: _plugin.ApronAdapter(cfg)}
    runner.pairing_store, runner.pairing_stores = None, {}

    def src(user, chat_type="dm"):
        return SessionSource(platform=platform, user_id=user, chat_id="r", user_name=user, chat_type=chat_type)

    assert runner._is_user_authorized(src("john")) is True
    assert runner._is_user_authorized(src("amanda", "group")) is True
    assert runner._is_user_authorized(src("mallory")) is False
    # A user listed without toolsets would inherit platform_toolsets.apron, so they are not let in.
    assert runner._is_user_authorized(src("nobody")) is False


def test_each_user_gets_their_own_toolsets():
    from gateway.run import GatewayRunner
    adapter = _plugin.ApronAdapter(PlatformConfig(enabled=True, extra={"url": "ws://x", "users": USERS}))
    runner = object.__new__(GatewayRunner)
    runner._delivery_adapter_for = lambda source: adapter
    platform_cfg = {"platform_toolsets": {"apron": ["web"]}}

    def resolve(user):
        source = SessionSource(platform=Platform("apron"), user_id=user, chat_id="r", chat_type="dm")
        return GatewayRunner._resolve_enabled_toolsets_for_source(runner, platform_cfg, source, "apron")

    john, amanda = resolve("john"), resolve("amanda")
    assert {"terminal", "file", "todo"} <= set(john)
    assert "terminal" not in amanda and "file" not in amanda and "todo" in amanda
    # Validated exactly as a platform_toolsets.apron list of hers would be.
    assert amanda == sorted(_get_platform_tools({"platform_toolsets": {"apron": ["todo"]}}, "apron"))


@pytest.mark.asyncio
async def test_reply_and_streaming_edit(monkeypatch):
    async with _apron(monkeypatch) as (fake, adapter):
        await adapter.connect()
        sent = await adapter.send("dm-john", "**hi**", reply_to="100")
        assert sent.success and sent.message_id == "91"
        assert (await adapter.edit_message("dm-john", sent.message_id, "**hi there**")).success
        assert fake.posted == [
            {"room_id": "dm-john", "body": {"text": "**hi**", "format": "markdown"}, "reply_to": {"message_id": "100"}},
            {"message_id": "91", "room_id": "dm-john", "body": {"text": "**hi there**", "format": "markdown"}},
        ]


@pytest.mark.asyncio
async def test_reconnects_after_the_connection_drops(monkeypatch):
    async with _apron(monkeypatch) as (fake, adapter):
        await adapter.connect()
        await fake.conns[-1].close()
        await _until(lambda: fake.auths == 2 and adapter._ws is not None)
        await fake.push("dm-john", "john", "still there?")
        await _until(lambda: adapter.inbound)
        assert adapter.inbound[0].text == "still there?"
        assert (await adapter.send("dm-john", "yes")).success


@pytest.mark.asyncio
async def test_refused_token_is_fatal_and_not_retried(monkeypatch):
    async with _apron(monkeypatch) as (fake, adapter):
        await adapter.connect()
        fake.token = "rotated"
        await fake.conns[-1].close()
        await _until(lambda: adapter.has_fatal_error)
        assert fake.auths == 2 and adapter._fatal_error_retryable is False


@pytest.mark.asyncio
async def test_connect_with_a_bad_token_fails_without_retry(monkeypatch):
    async with _apron(monkeypatch) as (fake, adapter):
        fake.token = "other"
        assert await adapter.connect() is False
        assert adapter._fatal_error_code == "auth_failed" and adapter._fatal_error_retryable is False
