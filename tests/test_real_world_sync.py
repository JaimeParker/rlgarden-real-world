"""Tests for the actor<->learner HTTP sync layer."""
from __future__ import annotations

import io
import threading
import time
import urllib.error
import urllib.request

import pytest
import torch

from rl_garden_real_world.sync import ActorSyncClient, LearnerSyncServer


def _start_server(on_transition):
    server = LearnerSyncServer("127.0.0.1", 0, on_transition=on_transition)
    server.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}"


def test_push_transition_reaches_learner_callback():
    received = []
    event = threading.Event()

    def on_transition(transition):
        received.append(transition)
        event.set()

    server, url = _start_server(on_transition)
    client = ActorSyncClient(url, poll_interval=10.0)
    try:
        client.push_transition(
            {
                "obs": torch.zeros(1, 4),
                "next_obs": torch.ones(1, 4),
                "action": torch.zeros(1, 2),
                "reward": torch.tensor([1.0]),
                "done": torch.tensor([False]),
            }
        )
        client.start()
        assert event.wait(timeout=5.0), "transition never reached the learner"
    finally:
        client.stop()
        server.stop()

    assert len(received) == 1
    torch.testing.assert_close(received[0]["next_obs"], torch.ones(1, 4))


def test_policy_params_round_trip_and_version_gating():
    server, url = _start_server(on_transition=lambda t: None)
    client = ActorSyncClient(url, poll_interval=10.0)
    try:
        # No params published yet.
        client._poll_once()
        assert client.latest_policy_params() is None

        server.publish_params({"w": torch.tensor([1.0, 2.0])})
        client._poll_once()
        params = client.latest_policy_params()
        assert params is not None
        torch.testing.assert_close(params["w"], torch.tensor([1.0, 2.0]))
        version_after_first_poll = client._cached_version

        # Polling again with no new publish must not change the version.
        client._poll_once()
        assert client._cached_version == version_after_first_poll

        server.publish_params({"w": torch.tensor([3.0, 4.0])})
        client._poll_once()
        torch.testing.assert_close(
            client.latest_policy_params()["w"], torch.tensor([3.0, 4.0])
        )
        assert client._cached_version == version_after_first_poll + 1
    finally:
        server.stop()


def test_push_transition_survives_learner_being_down_temporarily():
    # Client created against a port nobody is listening on; push must not
    # raise (the robot control loop must never crash on a network error).
    client = ActorSyncClient(
        "http://127.0.0.1:1", poll_interval=10.0, timeout=0.5, preflight=False
    )
    client.start()
    try:
        client.push_transition({"obs": torch.zeros(1, 2)})
        client._poll_once()  # must not raise either
        assert client.latest_policy_params() is None
    finally:
        client.stop()


def test_push_transition_drops_oldest_when_queue_full():
    # No server/threads needed -- push_transition() enqueues synchronously,
    # so filling the bounded queue and inspecting it is deterministic.
    client = ActorSyncClient("http://127.0.0.1:1", poll_interval=10.0, max_queue_size=4)
    for seq in range(10):
        client.push_transition({"seq": seq})

    assert client._queue.qsize() == 4
    assert client.dropped_transitions == 6
    remaining_seqs = [t["seq"] for t in client._queue.queue if t is not None]
    assert remaining_seqs == [6, 7, 8, 9]


def test_stop_does_not_block_when_queue_full():
    # Regression guard for the Item-1 fix: stop()'s sentinel used to be a
    # plain blocking queue.put(None), which could hang forever against a
    # full queue. The learner responds (with a small per-call delay so the
    # queue genuinely backs up under max_queue_size=2), so nothing here
    # depends on network-timeout behavior -- just that stop() returns fast.
    def on_transition(transition):
        time.sleep(0.05)

    server, url = _start_server(on_transition)
    client = ActorSyncClient(url, poll_interval=10.0, max_queue_size=2)
    try:
        client.start()
        for seq in range(8):
            client.push_transition({"seq": seq})

        start = time.perf_counter()
        client.stop()
        elapsed = time.perf_counter() - start
        assert elapsed < 3.0, f"stop() blocked for {elapsed:.2f}s with a full queue"
    finally:
        server.stop()


def _free_closed_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_preflight_failure_raises_with_hint():
    port = _free_closed_port()
    client = ActorSyncClient(f"http://127.0.0.1:{port}", poll_interval=10.0)
    with pytest.raises(RuntimeError, match=rf"learner unreachable at 127.0.0.1:{port}") as info:
        client.start()
    assert "--sync_host/--sync_port" in str(info.value)


def test_preflight_passes_against_live_server():
    server, url = _start_server(on_transition=lambda t: None)
    client = ActorSyncClient(url, poll_interval=10.0)
    try:
        client.start()
    finally:
        client.stop()
        server.stop()


def test_consecutive_failures_raise_after_limit_and_success_resets(monkeypatch):
    client = ActorSyncClient(
        "http://127.0.0.1:1", poll_interval=10.0, max_consecutive_failures=3
    )

    def refuse(*args, **kwargs):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    client._poll_once()
    client._poll_once()
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        client._poll_once()
    # the fatal error is surfaced on the actor's thread too.
    with pytest.raises(RuntimeError, match="learner unreachable at 127.0.0.1:1"):
        client.push_transition({"obs": torch.zeros(1)})
    with pytest.raises(RuntimeError):
        client.latest_policy_params()


def test_success_resets_failure_counter(monkeypatch):
    server, url = _start_server(on_transition=lambda t: None)
    client = ActorSyncClient(url, poll_interval=10.0, max_consecutive_failures=3)
    real_urlopen = urllib.request.urlopen
    try:
        monkeypatch.setattr(
            urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(ConnectionResetError())
        )
        client._poll_once()
        client._poll_once()
        assert client._consecutive_failures == 2
        monkeypatch.setattr(urllib.request, "urlopen", real_urlopen)
        client._poll_once()
        assert client._consecutive_failures == 0
        client.latest_policy_params()  # no fatal error
    finally:
        server.stop()


def test_stats_round_trip_keeps_latest_per_actor():
    server, url = _start_server(on_transition=lambda t: None)
    client_a = ActorSyncClient(url, poll_interval=10.0, actor_id="a")
    client_b = ActorSyncClient(url, poll_interval=10.0, actor_id="b")
    try:
        client_a.start()
        client_b.start()
        client_a.post_stats({"episode_return": 1.0, "step": 10})
        client_a.post_stats({"episode_return": 2.0, "step": 20})
        client_b.post_stats({"episode_return": 5.0})
        deadline = time.time() + 5.0
        while time.time() < deadline:
            stats = server.latest_stats()
            if stats.get("a", {}).get("step") == 20 and "b" in stats:
                break
            time.sleep(0.02)
        assert stats["a"] == {"episode_return": 2.0, "step": 20, "actor_id": "a"}
        assert stats["b"]["episode_return"] == 5.0
    finally:
        client_a.stop()
        client_b.stop()
        server.stop()


def test_transition_with_plain_containers_round_trips_under_weights_only():
    payload = {
        "obs": {"state": torch.zeros(1, 3), "rgb": torch.zeros(1, 2, 2, 3, dtype=torch.uint8)},
        "action": torch.zeros(1, 2),
        "done": torch.tensor([False]),
        "intervened": True,
        "grasp_penalty": 0.5,
        "actor_id": "a",
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)
    assert loaded["intervened"] is True and loaded["actor_id"] == "a"
    torch.testing.assert_close(loaded["obs"]["state"], payload["obs"]["state"])


def test_server_loads_transitions_with_weights_only(monkeypatch):
    seen = {}
    real_load = torch.load

    def spy(*args, **kwargs):
        seen["weights_only"] = kwargs.get("weights_only")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    received = threading.Event()
    server, url = _start_server(lambda t: received.set())
    client = ActorSyncClient(url, poll_interval=10.0)
    try:
        client.push_transition({"obs": torch.zeros(1), "intervened": False})
        client.start()
        assert received.wait(timeout=5.0)
    finally:
        client.stop()
        server.stop()
    assert seen["weights_only"] is True
