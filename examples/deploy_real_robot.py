"""Run a trained policy on a real robot through the ``real_robot`` backend (eval only).

    python examples/deploy_real_robot.py --checkpoint runs/<run>/checkpoints/checkpoint_<step>.pt \\
        --algo rlpd --real_robot.remote 192.168.1.10:7000 --real_robot.num-joints 7 \\
        --real_robot.home-joints 0 0 0 -1.5 0 1.5 0.8 --episodes 5

Logic lives in ``rl_garden_real_world/deploy.py``. Exits 2 on a robot ``DeviceError``.
"""
from __future__ import annotations

import sys

import tyro

from rl_garden_real_world.deploy import DeployArgs, main

if __name__ == "__main__":
    sys.exit(main(tyro.cli(DeployArgs)))
