"""Tests for the actor<->learner HTTP sync layer."""
from __future__ import annotations

import threading
import time

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
    client = ActorSyncClient("http://127.0.0.1:1", poll_interval=10.0, timeout=0.5)
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
