from gym.envs.registration import register

register(
    id='ABModel-v0',
    entry_point='ABModel.env:ABModelEnv',
)