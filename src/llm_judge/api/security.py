"""API protection: authentication, rate limiting, body-size limits and request IDs."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class ApiError(Exception):
    """An error that maps directly onto an HTTP response with the standard envelope."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        headers: dict[str, str] | None = None,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers or {}
        self.details = details


def request_id_from(header: str | None) -> str:
    """Use a well-formed client-supplied X-Request-ID, else generate one."""
    if header and _REQUEST_ID_RE.match(header):
        return header
    return uuid.uuid4().hex


def presented_key(headers: Any, header_name: str) -> str | None:
    """A key from ``<header_name>`` or an ``Authorization: Bearer`` header."""
    key = headers.get(header_name)
    if key:
        return str(key)
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return str(auth[7:].strip()) or None
    return None


def key_matches(presented: str | None, accepted: list[str]) -> bool:
    """Constant-time comparison against every accepted key."""
    if not presented:
        return False
    found = False
    for key in accepted:
        found |= hmac.compare_digest(presented.encode(), key.encode())
    return found


def parse_keys(raw: str | None) -> list[str]:
    return [k.strip() for k in (raw or "").split(",") if k.strip()]


def client_identity(key: str | None, client_host: str | None) -> str:
    """Rate-limit identity: a hash of the API key if present, else the client address."""
    if key:
        return "key:" + hashlib.sha256(key.encode()).hexdigest()[:16]
    return f"ip:{client_host or 'unknown'}"


class RateLimiter:
    """Thread-safe token bucket per identity: ``per_minute`` requests, bursting to the same."""

    def __init__(self, per_minute: int, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.per_minute = per_minute
        self._rate = per_minute / 60.0
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, float]] = {}

    def check(self, identity: str) -> float:
        """Consume one token. Returns 0 if allowed, else seconds until a token frees up."""
        if self.per_minute <= 0:
            return 0.0
        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.get(identity, (float(self.per_minute), now))
            tokens = min(float(self.per_minute), tokens + (now - last) * self._rate)
            if tokens >= 1:
                self._buckets[identity] = (tokens - 1, now)
                if len(self._buckets) > 10_000:  # bound memory under many distinct clients
                    self._buckets.pop(next(iter(self._buckets)))
                return 0.0
            self._buckets[identity] = (tokens, now)
            return (1 - tokens) / self._rate


class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Reject request bodies over ``max_bytes`` with 413, with or without Content-Length."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        declared = headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(scope, send)
            return

        received = 0
        exceeded = False
        started = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    exceeded = True
                    raise _BodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            # Frameworks may catch the receive error and answer with their own
            # (e.g. 400 "error parsing the body"); replace that with a 413.
            nonlocal started
            if exceeded:
                if message["type"] == "http.response.start" and not started:
                    started = True
                    await self._reject(scope, send)
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _BodyTooLarge:
            if not started:
                await self._reject(scope, send)

    async def _reject(self, scope: Scope, send: Send) -> None:
        request_id = scope.get("state", {}).get("request_id")
        body = json.dumps(
            {
                "error": {
                    "code": "payload_too_large",
                    "message": f"Request body exceeds {self.max_bytes} bytes.",
                    "request_id": request_id,
                    "details": None,
                }
            }
        ).encode()
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ]
        if request_id:
            headers.append((b"x-request-id", request_id.encode()))
        await send({"type": "http.response.start", "status": 413, "headers": headers})
        await send({"type": "http.response.body", "body": body})
