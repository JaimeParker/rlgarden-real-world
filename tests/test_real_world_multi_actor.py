"""Multiple actors against one learner: LearnerSyncServer has no per-client
state, so N separate ActorSyncClient instances (one per robot process) can
already push to and pull from one learner with zero protocol changes -- this
verifies that end to end, real HTTP, no mocks (same style as
test_real_world_sync.py)."""
from __future__ import annotations

import threading

import torch

from rl_garden_real_world.learner_loop import LearnerLoop
from rl_garden_real_world.sync import ActorSyncClient, LearnerSyncServer


def _start_server(on_transition):
    server = LearnerSyncServer("127.0.0.1", 0, on_transition=on_transition)
    server.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}"


def _transition(seq: int) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.full((1, 4), float(seq)),
        "next_obs": torch.full((1, 4), float(seq) + 1),
        "action": torch.zeros(1, 2),
        "reward": torch.tensor([1.0]),
        "done": torch.tensor([False]),
    }


def test_two_actors_reach_one_learner_and_share_broadcast_params():
    received = []
    received_event = threading.Event()

    def on_transition(transition):
        received.append(transition)
        if len(received) >= 2:
            received_event.set()

    server, url = _start_server(on_transition)
    client_a = ActorSyncClient(url, poll_interval=10.0, actor_id="robot_a")
    client_b = ActorSyncClient(url, poll_interval=10.0, actor_id="robot_b")
    try:
        client_a.push_transition(_transition(0))
        client_b.push_transition(_transition(1))
        client_a.start()
        client_b.start()
        assert received_event.wait(timeout=5.0), "not all transitions reached the learner"

        actor_ids = {t["actor_id"] for t in received}
        assert actor_ids == {"robot_a", "robot_b"}

        server.publish_params({"w": torch.tensor([1.0, 2.0])})
        client_a._poll_once()
        client_b._poll_once()
        torch.testing.assert_close(
            client_a.latest_policy_params()["w"], torch.tensor([1.0, 2.0])
        )
        torch.testing.assert_close(
            client_b.latest_policy_params()["w"], torch.tensor([1.0, 2.0])
        )
    finally:
        client_a.stop()
        client_b.stop()
        server.stop()


class _FakeReplayBuffer:
    def __init__(self):
        self.add_calls = []

    def add(self, obs, next_obs, action, reward, done, **kwargs):
        self.add_calls.append(dict(obs=obs, action=action, **kwargs))


class _FakePolicy:
    def state_dict(self):
        return {}


class _FakeAgent:
    def __init__(self):
        self.buffer_device = "cpu"
        self.replay_buffer = _FakeReplayBuffer()
        self.learning_starts = 10_000  # never crosses -> run() just receives
        self.utd = 1.0
        self.policy = _FakePolicy()
        self.checkpoint_dir = None
        self.checkpoint_freq = 0
        self.save_replay_buffer = False
        self.save_final_checkpoint = False
        self.global_update = 0


def test_learner_loop_tallies_concurrent_multi_actor_pushes():
    agent = _FakeAgent()
    loop = LearnerLoop(agent, "127.0.0.1", 0, idle_poll_interval=0.02)
    host, port = None, None

    thread = threading.Thread(target=loop.run, daemon=True)
    thread.start()
    try:
        # Wait for the server to actually be listening before connecting.
        import time

        deadline = time.time() + 5.0
        while loop._server.server_address[1] == 0 and time.time() < deadline:
            time.sleep(0.01)
        host, port = loop._server.server_address
        url = f"http://{host}:{port}"

        clients = [ActorSyncClient(url, poll_interval=10.0, actor_id=f"robot_{i}") for i in range(3)]
        for c in clients:
            c.start()
        for i, c in enumerate(clients):
            for seq in range(4):
                c.push_transition(_transition(seq))

        deadline = time.time() + 5.0
        while loop.received_transitions < 12 and time.time() < deadline:
            time.sleep(0.02)

        assert loop.received_transitions == 12
        assert loop.received_by_actor == {"robot_0": 4, "robot_1": 4, "robot_2": 4}
        assert all("actor_id" not in call for call in agent.replay_buffer.add_calls)
        for c in clients:
            c.stop()
    finally:
        loop.stop()
        thread.join(timeout=5.0)
