# vim: tabstop=2 softtabstop=2 shiftwidth=2 expandtab
from gym.envs.registration import registry, register, make, spec

register(
    id='ModelBasedCartPoleFromRandomPolicy-v0',
    entry_point='gym_model_cartpole.model:ModelBasedCartPoleEnv',
    kwargs={
      'name':'cartpolefromrandom',
      'sstates_filename':'random.sstates.csv',
      'model_filename':'random.tmodel.h5'
    }
)

register(
    id='ModelBasedCartPoleFromDQNPolicy-v0',
    entry_point='gym_model_cartpole.model:ModelBasedCartPoleEnv',
    kwargs={
      'name':'cartpolefromdqn',
      'sstates_filename':'dqn.sstates.csv',
      'model_filename':'dqn.tmodel.h5'
    }
)
