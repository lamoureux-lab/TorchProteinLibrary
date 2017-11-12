from gym.envs.registration import register

register(
    id='ABModel-L13',
    entry_point='ABModel.env:ABEnv',
)