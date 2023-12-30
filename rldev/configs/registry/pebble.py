
from rldev.agents.core.bpref.sac import SACPolicy, DiagGaussianActor, DoubleQCritic
from rldev.configs.xconf import Conf, required
from rldev.configs.registry.registry import get, register


conf = Conf()

conf.logging = {}
conf.logging.wandb = "rldev"

conf.experiment = 'PEBBLE'
conf.segment = 50
conf.activation = 'tanh'
conf.num_seed_steps = 1000
conf.num_unsup_steps = 9000
conf.num_interact = 5000
conf.reward_lr = 0.0003
conf.reward_batch = 50
conf.reward_update = 10
conf.feed_type = 0
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
conf.device = "cuda"
conf.log_every_n_steps = 3000
conf.log_save_tb = True
conf.save_video = False
conf.seed = 1
conf.gradient_update = 2
conf.aligned_goals = False

conf.policy = {}
conf.policy.name = "sac"
conf.policy.cls = SACPolicy

conf.policy.kwargs = {}
conf.policy.kwargs.obs_dim = required(int)
conf.policy.kwargs.action_dim = required(int)
conf.policy.kwargs.action_range = required(list)
conf.policy.kwargs.device = conf.device
conf.policy.kwargs.discount = 0.99
conf.policy.kwargs.init_temperature = 0.1
conf.policy.kwargs.alpha_lr = 0.0001
conf.policy.kwargs.alpha_betas = [0.9, 0.999]
conf.policy.kwargs.actor_lr = 0.0003
conf.policy.kwargs.actor_betas = [0.9, 0.999]
conf.policy.kwargs.actor_update_frequency = 1
conf.policy.kwargs.critic_lr = 0.0003
conf.policy.kwargs.critic_betas = [0.9, 0.999]
conf.policy.kwargs.critic_tau = 0.005
conf.policy.kwargs.critic_target_update_frequency = 2
conf.policy.kwargs.batch_size = 512
conf.policy.kwargs.learnable_temperature = True

conf.critic = {}
conf.critic.cls = DoubleQCritic
conf.critic.kwargs = {}
conf.critic.kwargs.hidden_dim = 256
conf.critic.kwargs.hidden_depth = 3

conf.actor = {}
conf.actor.cls = DiagGaussianActor
conf.actor.kwargs = {}
conf.actor.kwargs.hidden_depth = 3
conf.actor.kwargs.hidden_dim = 256
conf.actor.kwargs.log_std_bounds = [-5, 2]

register("pebble", conf)


conf = get("pebble")
conf.aligned_goals = True
register("pebble-aligned", conf)