import gymnasium as gym
import supersuit as ss
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import FrameStackObservation


def get_env(
    env_id: str, stack_num: int, action_mask: bool = False, env_kwargs: dict = {}
) -> gym.Env:
    env = gym.make(
        env_id,
        **env_kwargs,
    )
    env = ss.normalize_obs_v0(env)
    env = FrameStackObservation(env, stack_num)  # type: ignore
    env = FlattenObservation(env)
    if action_mask:
        env = ActionMasker(env, lambda env: env.action_masks())  # type: ignore
    return env