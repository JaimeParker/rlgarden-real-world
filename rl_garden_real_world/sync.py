"""Actor<->learner HTTP sync layer for real-robot training.

Self-written, stdlib-only (``http.server`` / ``urllib.request``) -- SERL uses
an external ``agentlace`` package for this, but it is a pip dependency not
vendored anywhere in this repo, so it isn't reused here. Transitions flow
actor -> learner; policy parameters flow learner -> actor. Neither side may
block the other's real-time loop: ``LearnerSyncServer`` accepts a transition
POST and hands it off to a callback immediately, and ``ActorSyncClient``
queues transitions and polls for params on a background thread so the
actor's fixed-frequency control loop never waits on network I/O.
"""
from __future__ import annotations

import io
import json
import logging
import queue
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)

# Transitions are deserialised with ``torch.load(weights_only=True)``, so a
# payload may only contain tensors and plain containers/primitives (dict,
# list, tuple, str, int, float, bool, None) -- never arbitrary pickled objects.
# Stats travel as JSON on /stats.
_NETWORK_ERRORS = (urllib.error.URLError, ConnectionError, TimeoutError)


class LearnerSyncServer:
    """Runs on the learner process.

    Receives transitions via POST /transition (each call synchronously
    invokes ``on_transition`` on the server's request-handling thread --
    callers that mutate shared state from it are responsible for their own
    locking). Serves the latest published policy params via
    GET /policy_params, versioned so a client only downloads a payload when
    the version differs from the one it already has cached.
    """

    def __init__(
        self,
        host: str,
        port: int,
        on_transition: Callable[[dict[str, Any]], None],
    ) -> None:
        self._on_transition = on_transition
        self._params_lock = threading.Lock()
        self._params_bytes: Optional[bytes] = None
        self._params_version = 0
        self._stats_lock = threading.Lock()
        self._latest_stats: dict[str, dict[str, Any]] = {}

        handler = _make_handler(self)
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def server_address(self) -> tuple[str, int]:
        return self._httpd.server_address

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5.0)

    def publish_params(self, state_dict: dict[str, Any]) -> None:
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        with self._params_lock:
            self._params_bytes = buffer.getvalue()
            self._params_version += 1

    def latest_stats(self) -> dict[str, dict[str, Any]]:
        """Most recent stats dict posted by each actor, keyed by ``actor_id``
        (``"unknown"`` when the actor did not stamp one)."""
        with self._stats_lock:
            return {actor_id: dict(stats) for actor_id, stats in self._latest_stats.items()}

    def _record_stats(self, stats: dict[str, Any]) -> None:
        actor_id = str(stats.get("actor_id", "unknown"))
        with self._stats_lock:
            self._latest_stats[actor_id] = stats
        logger.info("actor stats [%s]: %s", actor_id, stats)

    def _latest_params(self) -> tuple[int, Optional[bytes]]:
        with self._params_lock:
            return self._params_version, self._params_bytes


def _make_handler(server: LearnerSyncServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            del format, args  # silence stdlib's default stderr access log

        def do_POST(self) -> None:
            if self.path not in ("/transition", "/stats"):
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            if self.path == "/stats":
                server._record_stats(json.loads(body))
            else:
                transition = torch.load(io.BytesIO(body), weights_only=True)
                server._on_transition(transition)
            self.send_response(204)
            self.end_headers()

        def do_GET(self) -> None:
            if not self.path.startswith("/policy_params"):
                self.send_response(404)
                self.end_headers()
                return
            client_version = -1
            if "?" in self.path:
                query = self.path.split("?", 1)[1]
                params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
                if "version" in params:
                    client_version = int(params["version"])
            version, payload = server._latest_params()
            if payload is None or version == client_version:
                self.send_response(204)
                self.send_header("X-Params-Version", str(version))
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-Params-Version", str(version))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


class ActorSyncClient:
    """Runs on the actor process.

    ``push_transition`` and ``latest_policy_params`` are both non-blocking:
    transitions are queued and drained by a background thread that owns the
    actual HTTP round trip, and policy params are polled on a fixed interval
    by that same background thread and cached locally -- the actor's control
    loop only ever reads the cache, it never makes a network call itself.
    This is required for the actor to hold its fixed control frequency: a
    slow learner (e.g. mid gradient-step) must never stall robot control.

    The push queue is bounded (``max_queue_size``, default 600 -- roughly
    one minute of backlog at a 10Hz control loop) and drops the oldest
    queued (not-yet-sent) transition when full, never blocking the caller:
    a robot control loop cares about recency of what reaches the learner
    more than completeness, and must never stall waiting for network I/O.
    ``dropped_transitions`` exposes the running count so a silently-lossy
    run is detectable rather than just quietly training on less data.

    ``LearnerSyncServer`` has no per-client state -- multiple
    ``ActorSyncClient`` instances (separate processes, e.g. one per robot)
    can already push to and pull from one learner with no protocol changes.
    ``actor_id``, when set, is stamped onto every pushed transition purely
    for observability on the learner side (see ``LearnerLoop.received_by_actor``);
    it carries no routing behavior.

    Failure handling: network errors on push/poll are tolerated (the control
    loop must not stall), but counted across both workers. The first failure
    and every 10th afterwards log a warning with the target host:port; after
    ``max_consecutive_failures`` in a row the client gives up, and the next
    ``push_transition`` / ``latest_policy_params`` call on the actor's thread
    raises ``RuntimeError`` (any success resets the count). ``start()`` also
    runs a one-time TCP preflight so a wrong host/port fails immediately.
    """

    def __init__(
        self,
        learner_url: str,
        poll_interval: float = 1.0,
        timeout: float = 5.0,
        max_queue_size: int = 600,
        actor_id: Optional[str] = None,
        max_consecutive_failures: int = 30,
        preflight: bool = True,
    ) -> None:
        self._base_url = learner_url.rstrip("/")
        parsed = urllib.parse.urlparse(self._base_url)
        self._host = parsed.hostname or ""
        self._port = parsed.port or 80
        self._max_consecutive_failures = max_consecutive_failures
        self._preflight = preflight
        self._failure_lock = threading.Lock()
        self._consecutive_failures = 0
        self._fatal_error: Optional[RuntimeError] = None
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._actor_id = actor_id

        self._queue: "queue.Queue[Any]" = queue.Queue(
            maxsize=max_queue_size
        )
        self._dropped_lock = threading.Lock()
        self._dropped = 0
        self._params_lock = threading.Lock()
        self._cached_version = -1
        self._cached_params: Optional[dict[str, Any]] = None

        self._stop_event = threading.Event()
        self._push_thread = threading.Thread(target=self._push_worker, daemon=True)
        self._poll_thread = threading.Thread(target=self._poll_worker, daemon=True)

    def _unreachable_error(self, attempts: int) -> RuntimeError:
        return RuntimeError(
            f"learner unreachable at {self._host}:{self._port} after {attempts} attempts; "
            "check --sync_host/--sync_port and that the learner process is running"
        )

    def _preflight_check(self) -> None:
        try:
            socket.create_connection((self._host, self._port), timeout=2.0).close()
        except OSError as exc:
            raise self._unreachable_error(1) from exc

    def _record_success(self) -> None:
        with self._failure_lock:
            self._consecutive_failures = 0

    def _record_failure(self, exc: BaseException) -> None:
        """Counts a network failure; stores and raises the fatal
        ``RuntimeError`` once ``max_consecutive_failures`` is reached."""
        with self._failure_lock:
            self._consecutive_failures += 1
            count = self._consecutive_failures
            if count == 1 or count % 10 == 0:
                logger.warning(
                    "learner at %s:%s unreachable (%d consecutive failures): %s",
                    self._host, self._port, count, exc,
                )
            if count >= self._max_consecutive_failures:
                self._fatal_error = self._unreachable_error(count)
                raise self._fatal_error

    def _raise_if_fatal(self) -> None:
        if self._fatal_error is not None:
            raise self._fatal_error

    def post_stats(self, stats: dict[str, Any]) -> None:
        """Non-blocking: queues a JSON-serialisable stats dict for the
        learner's ``latest_stats()`` (subject to the same drop-oldest queue)."""
        self._raise_if_fatal()
        if self._actor_id is not None:
            stats = {**stats, "actor_id": self._actor_id}
        self._put_dropping_oldest(_StatsMessage(stats))

    def start(self) -> None:
        if self._preflight:
            self._preflight_check()
        self._push_thread.start()
        self._poll_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._put_dropping_oldest(None)  # unblock the push worker's queue.get()
        self._push_thread.join(timeout=5.0)
        self._poll_thread.join(timeout=5.0)

    @property
    def dropped_transitions(self) -> int:
        with self._dropped_lock:
            return self._dropped

    def _put_dropping_oldest(self, item: Optional[dict[str, Any]]) -> None:
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()  # discard oldest queued (unsent) item
                except queue.Empty:
                    pass  # push worker drained it concurrently; retry put
                else:
                    with self._dropped_lock:
                        self._dropped += 1

    def push_transition(self, transition: dict[str, Any]) -> None:
        self._raise_if_fatal()
        if self._actor_id is not None:
            transition = {**transition, "actor_id": self._actor_id}
        self._put_dropping_oldest(transition)

    def latest_policy_params(self) -> Optional[dict[str, Any]]:
        """Returns the most recently cached params, or ``None`` before the
        first successful poll. Never blocks or makes a network call."""
        self._raise_if_fatal()
        with self._params_lock:
            return self._cached_params

    def _push_worker(self) -> None:
        while not self._stop_event.is_set():
            item = self._queue.get()
            if item is None:
                continue
            try:
                if isinstance(item, _StatsMessage):
                    path, data = "/stats", json.dumps(item.stats).encode()
                else:
                    buffer = io.BytesIO()
                    torch.save(item, buffer)
                    path, data = "/transition", buffer.getvalue()
                req = urllib.request.Request(f"{self._base_url}{path}", data=data, method="POST")
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    resp.read()
                self._record_success()
            except _NETWORK_ERRORS as exc:
                # dropped item; robot control must not stall on this
                try:
                    self._record_failure(exc)
                except RuntimeError:
                    return  # fatal error is re-raised on the actor's thread

    def _poll_worker(self) -> None:
        while not self._stop_event.wait(self._poll_interval):
            try:
                self._poll_once()
            except RuntimeError:
                return  # fatal error is re-raised on the actor's thread

    def _poll_once(self) -> None:
        try:
            req = urllib.request.Request(
                f"{self._base_url}/policy_params?version={self._cached_version}"
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                version = int(resp.headers.get("X-Params-Version", self._cached_version))
                if resp.status == 204:
                    self._cached_version = version
                    self._record_success()
                    return
                body = resp.read()
                params = torch.load(io.BytesIO(body), weights_only=False)
            with self._params_lock:
                self._cached_version = version
                self._cached_params = params
            self._record_success()
        except _NETWORK_ERRORS as exc:
            self._record_failure(exc)  # keeps the previously cached params until fatal


class _StatsMessage:
    __slots__ = ("stats",)

    def __init__(self, stats: dict[str, Any]) -> None:
        self.stats = stats
