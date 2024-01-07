
from rldev.agents.pref.sac import DiagGaussianActor, QFunction
from rldev.configs.xconf import Conf
from rldev.configs.registry.registry import get, register


conf = Conf()

conf.num_seed_steps = 1000
conf.reset_update = 100
conf.topK = 5

conf.steps = 1000000
conf.replay_buffer_capacity = conf.ref("steps")
conf.test_every_n_steps = 10000
conf.test_episodes = 10
conf.log_every_n_steps = 3000
conf.log_save_tb = True
conf.save_video = False
conf.seed = 1
conf.gradient_update = 2
conf.tau = 0.005
conf.discount = 0.99
conf.batch_size = 512
conf.learnable_alpha = (
  True, dict(init=0.1, lr=0.0001, betas=[0.9, 0.999]))

conf.qf = Conf()
conf.qf.cls = QFunction
conf.qf.kwargs = Conf()
conf.qf.kwargs.dims = [256, 256, 256]
conf.qf.kwargs.lr = 0.0003
conf.qf.kwargs.betas = [0.9, 0.999]
conf.qf.kwargs.decay = 0.0
conf.qf.kwargs.n_funcs = 2

conf.update_qf_target_every_n_steps = 2

conf.pi = Conf()
conf.pi.cls = DiagGaussianActor
conf.pi.kwargs = Conf()
conf.pi.kwargs.dims = [256, 256, 256]
conf.pi.kwargs.log_std_bounds = [-5, 2]
conf.pi.kwargs.lr = 0.0003
conf.pi.kwargs.betas = [0.9, 0.999]
conf.pi.kwargs.decay = 0.0

conf.update_pi_every_n_steps = 1

register("sac", conf)