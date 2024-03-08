"""register environments for use with gymnasium.make."""

from gymnasium.envs.registration import register

register(
    id="s5/simulation-v0",
    entry_point="envs.schedulizer:Simulation",
)
