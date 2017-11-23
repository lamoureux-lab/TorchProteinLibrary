from gym.envs.registration import register

register(
    id='ABModel-v0',
    entry_point='ABModel.ABModel_env_partial:ABModelEnv',
)

register(
    id='ABModel-v1',
    entry_point='ABModel.ABModel_env_full:ABModelEnv',
)