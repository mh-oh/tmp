
from rldev.agents.pref.sac import DiagGaussianActor, DoubleQCritic
from rldev.configs.xconf import Conf
from rldev.configs.registry.registry import register


conf = Conf()

conf.activation = 'tanh'
conf.num_seed_steps = 1000
conf.reset_update = 100

conf.steps = 1000000
conf.replay_buffer_capacity = conf.ref("steps")
conf.test_every_n_steps = 10000
conf.test_episodes = 10
conf.log_every_n_steps = 3000
conf.log_save_tb = True
conf.save_video = False
conf.seed = 1
conf.gradient_update = 2

conf.policy = Conf()
conf.policy.kwargs = Conf()
conf.policy.kwargs.discount = 0.99
conf.policy.kwargs.init_temperature = 0.1
conf.policy.kwargs.alpha_lr = 0.0001
conf.policy.kwargs.alpha_betas = [0.9, 0.999]
conf.policy.kwargs.batch_size = 512
conf.policy.kwargs.learnable_temperature = True

conf.query = Conf()
conf.query.starter_mode = "uniform"
conf.query.starter_kwargs = dict()
conf.query.mode = "uniform"
conf.query.kwargs = dict()

conf.qf = Conf()
conf.qf.cls = DoubleQCritic
conf.qf.kwargs = Conf()
conf.qf.kwargs.hidden_dim = 256
conf.qf.kwargs.hidden_depth = 3
conf.qf.kwargs.lr = 0.0003
conf.qf.kwargs.betas = [0.9, 0.999]
conf.qf.kwargs.tau = 0.005
conf.qf.kwargs.target_update_frequency = 2

conf.pi = Conf()
conf.pi.cls = DiagGaussianActor
conf.pi.kwargs = Conf()
conf.pi.kwargs.hidden_depth = 3
conf.pi.kwargs.hidden_dim = 256
conf.pi.kwargs.log_std_bounds = [-5, 2]
conf.pi.kwargs.lr = 0.0003
conf.pi.kwargs.betas = [0.9, 0.999]
conf.pi.kwargs.update_frequency = 1

register("sac", conf)