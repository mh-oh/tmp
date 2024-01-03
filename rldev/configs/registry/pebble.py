
from rldev.agents.pref.sac import DiagGaussianActor, DoubleQCritic
from rldev.configs.xconf import Conf
from rldev.configs.registry.registry import get, register


conf = Conf()

conf.segment = 50
conf.activation = 'tanh'
conf.num_seed_steps = 1000
conf.num_unsup_steps = 9000
conf.num_interact = 5000
conf.reward_lr = 0.0003
conf.reward_batch = 50
conf.reward_update = 10
conf.feed_type = "uniform"
conf.reset_update = 100
conf.topK = 5
conf.ensemble_size = 3
conf.max_feedback = 10000
conf.large_batch = 10
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

conf.aligned_goals = False
conf.discard_outlier_goals = False # unused if aligned_goals=False
conf.cluster = Conf() # unused if aligned_goals=False
conf.cluster.cls = "dbscan"
conf.cluster.kwargs = dict(eps=0.3)

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
conf.segment = 10000
register("uniform-long", conf)

conf = get("uniform")
conf.feed_type = "entropy"
register("entropy", conf)

conf = get("uniform")
conf.aligned_goals = True
register("uniform-aligned-include-outliers", conf)

conf = get("uniform-aligned-include-outliers")
conf.discard_outlier_goals = True
register("uniform-aligned", conf)

conf = get("uniform-aligned")
conf.feed_type = "greedy_aligned_entropy"
register("entropy-aligned", conf)

conf = get("entropy-aligned")
conf.cluster.cls = "kmeans"
conf.cluster.kwargs = dict(n_clusters=2)
register("entropy-aligned-kmeans", conf)

conf = get("uniform-aligned")
conf.cluster.cls = "kmeans"
conf.cluster.kwargs = dict(n_clusters=2)
register("uniform-aligned-kmeans", conf)

conf = get("entropy-aligned")
conf.cluster.cls = "kmeans"
conf.cluster.kwargs = dict(n_clusters=2)
conf.segment = 10000
register("entropy-aligned-kmeans-2-long", conf)