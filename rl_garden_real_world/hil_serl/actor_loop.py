"""HIL-SERL's ActorLoop. HITL intervention and the hybrid continuous+discrete
action are handled by the env wrapper stack and the policy respectively, so
the only overrides needed are tagging pushed transitions with whether a human
was intervening -- ``HilSerlLearnerLoop`` uses this to route the transition
into the online replay buffer or the growing demo buffer -- and forwarding
``grasp_penalty`` when ``GraspPenaltyWrapper`` is in the env stack (it flows
straight through to ``DictReplayBuffer.add(grasp_penalty=...)`` once
received, no further routing needed).
"""
from __future__ import annotations

from rl_garden_real_world.actor_loop import ActorLoop


class HilSerlActorLoop(ActorLoop):
    def _extra_transition_fields(self, info: dict) -> dict:
        fields = {"intervened": "intervene_action" in info}
        if "grasp_penalty" in info:
            fields["grasp_penalty"] = info["grasp_penalty"]
        return fields
