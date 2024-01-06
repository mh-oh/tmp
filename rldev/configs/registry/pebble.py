
from sklearn.cluster import DBSCAN, KMeans

from rldev.agents.pref.sac import DiagGaussianActor, DoubleQCritic
from rldev.configs.xconf import Conf
from rldev.configs.registry.registry import get, register


conf = Conf()

conf.fusion = 3
conf.activation = 'tanh'
conf.reward_lr = 0.0003
conf.budget = 50
conf.segment_length = 50

conf.num_seed_steps = 1000
conf.num_unsup_steps = 9000
conf.num_interact = 5000
conf.reward_update = 10
conf.reset_update = 100
conf.topK = 5
conf.max_feedback = 10000
conf.label_margin = 0.0
conf.teacher_beta = -1
conf.teacher_gamma = 1
conf.teacher_eps_mistake = 0
conf.teacher_eps_skip = 0
conf.teacher_eps_equal = 0
conf.reward_schedule = 0
conf.steps = 1000000
conf.replay_buffer_capacity = conf.steps
conf.test_every_n_steps = 10000
conf.num_eval_episodes = 10
conf.log_every_n_steps = 3000
conf.log_save_tb = True
conf.save_video = False
conf.seed = 1
conf.gradient_update = 2

conf.query = Conf()
conf.query.mode = "uniform"
conf.query.kwargs = dict()

conf.policy = Conf()
conf.policy.kwargs = Conf()
conf.policy.kwargs.discount = 0.99
conf.policy.kwargs.init_temperature = 0.1
conf.policy.kwargs.alpha_lr = 0.0001
conf.policy.kwargs.alpha_betas = [0.9, 0.999]
conf.policy.kwargs.batch_size = 512
conf.policy.kwargs.learnable_temperature = True

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

register("uniform", conf)

conf = get("uniform")
conf.query.mode = "entropy"
conf.query.kwargs = dict(scale=10)
register("entropy", conf)

conf = get("uniform")
conf.query.mode = "disagree"
conf.query.kwargs = dict(scale=10)
register("disagree", conf)

conf = get("uniform")
conf.query.mode = "uniform_aligned"
conf.query.kwargs = dict(cluster=KMeans(n_clusters=3))
register("uniform-kmeans-3", conf)

conf = get("uniform")
conf.query.mode = "entropy_aligned"
conf.query.kwargs = dict(cluster=KMeans(n_clusters=3))
register("entropy-kmeans-3", conf)

