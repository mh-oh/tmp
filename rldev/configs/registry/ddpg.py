
import torch as th
import numpy as np

from rldev.configs.xconf import Conf
from rldev.configs.registry.registry import get, register


conf = Conf()

conf.steps = 1000000
conf.gamma = 0.99 # discount factor
conf.actor_lr = 1e-3 # actor learning rate
conf.critic_lr = 1e-3 # critic learning rate
conf.actor_weight_decay = 0. # weight decay to apply to actor
conf.action_l2_regularization = 1e-2 # (, 'l2 penalty for action norm'),
conf.critic_weight_decay = 0. # (, 'weight decay to apply to critic'),
conf.optimize_every = 2 # (, 'how often optimize is called, in terms of environment steps'),
conf.batch_size = 2000 # (, 'batch size for training the actors/critics'),
conf.warm_up = 10000 # (, 'minimum steps in replay buffer needed to optimize'),  
conf.initial_explore = 10000 # (, 'steps that actor acts randomly for at beginning of training'), 
conf.grad_norm_clipping = -1. # (, 'gradient norm clipping'),
conf.grad_value_clipping = -1. # (, 'gradient value clipping'),
conf.target_network_update_frac = 0.005 # (, 'polyak averaging coefficient for target networks'),
conf.target_network_update_freq = 1 # (, 'how often to update target networks; NOTE: TD3 uses this too!'),
conf.clip_target_range = (-np.inf, np.inf) # (, 'q/value targets are clipped to this range'),
conf.td3_noise = 0.1 # (, 'noise added to next step actions in td3'),
conf.td3_noise_clip = 0.3 # (, 'amount to which next step noise in td3 is clipped'),
conf.td3_delay = 2 # (, 'how often the actor is trained, in terms of critic training steps, in td3'),
conf.entropy_coef = 0.2 # (, 'Entropy regularization coefficient for SAC'),
conf.policy_opt_noise = 0. # (, 'how much policy noise to add to actor optimization'),
conf.action_noise = 0.1 # (, 'maximum std of action noise'),
conf.eexplore = 0. # (, 'how often to do completely random exploration (overrides action noise)'),
conf.go_eexplore = 0.1 # (, 'epsilon exploration bonus from each point of go explore, when using intrinsic curiosity'),
conf.go_reset_percent = 0. # (, 'probability to reset episode early for each point of go explore, when using intrinsic curiosity'),
conf.overshoot_goal_percent = 0. # (, 'if using instrinsic FIRST VISIT goals, should goal be overshot on success?'),
conf.direct_overshoots = False # (, 'if using overshooting, should it be directed in a straight line?'),
conf.dg_score_multiplier = 1. # (, 'if using instrinsic goals, score multiplier for goal candidates that are in DG distribution'),
conf.cutoff_success_threshold = (0.3, 0.7), # thresholds for decreasing/increasing the cutoff
conf.initial_cutoff = -3 # (, 'initial (and minimum) cutoff for intrinsic goal curiosity'),
conf.activ = "gelu" # (, 'activation to use for hidden layers in networks'),
conf.curiosity_beta = -3. # (, 'beta to use for curiosity_alpha module'),
conf.sigma_l2_regularization = 0. # (, 'l2 regularization on sigma critics log variance'),
conf.seed = 0 # (, 'random seed'),
conf.replay_size = int(1e6) # (, 'maximum size of replay buffer'),
conf.save_replay_buf = False # (, 'save replay buffer checkpoint during training?'),
conf.num_envs = 12 # (, 'number of parallel envs to run'),
conf.num_eval_envs = 10 # (, 'number of parallel eval envs to run'),
conf.log_every_n_steps = 5000 # (, 'how often to log things'),
conf.varied_action_noise = False # (, 'if true, action noise for each env in vecenv is interpolated between 0 and action noise'),
conf.use_actor_target = False # (, 'if true, use actor target network to act in the environment'),
conf.her = "futureactual_2_2" # (, 'strategy to use for hindsight experience replay'),
conf.prioritized_mode = "none" # (, 'buffer prioritization strategy'),
conf.future_warm_up = 25000 # (, 'minimum steps in replay buffer needed to stop doing ONLY future sampling'),  
conf.sparse_reward_shaping = 0. # (, 'coefficient of euclidean distance reward shaping in sparse goal envs'),
conf.n_step_returns = 1 # (, 'if using n-step returns, how many steps?'),
conf.slot_based_state = False # (, 'if state is organized by slot; i.e., [batch_size, num_slots, slot_feats]'),
conf.modalities = ["observation"] # (, 'keys the agent accesses in dictionary env for observations'),
conf.goal_modalities = ["desired_goal"] # (, 'keys the agent accesses in dictionary env for goals')
conf.policy_layers = (512, 512, 512)
register("ddpg-default", conf)


conf = get("ddpg-default")
conf.gamma = 0.98
conf.actor_lr = 1e-3
conf.critic_lr = 1e-3
conf.actor_weight_decay = 0.
conf.action_l2_regularization = 1e-1
conf.target_network_update_freq = 40
conf.target_network_update_frac = 0.05
conf.optimize_every = 1
conf.batch_size = 2000
conf.warm_up = 2500
conf.initial_explore = 5000
conf.replay_size = int(1e6)
conf.clip_target_range = (-50.,0.)
conf.action_noise = 0.1
conf.eexplore = 0.1
conf.go_eexplore = 0.1
conf.go_reset_percent = 0.
conf.her = "rfaab_1_4_3_1_1"
conf.grad_value_clipping = 5.
register("ddpg-protoge", conf)


conf = get("ddpg-protoge")
conf.eexplore = None
conf.grad_value_clipping = -1
conf.her = "futureactual_2_2"
conf.replay_size = int(2.5e6)
conf.initial_explore = 5000
conf.warm_up = 5000
conf.action_l2_regularization = 1e-2
conf.optimize_every = 2
conf.target_network_update_freq = 10
conf.activ = "relu"
register("ddpg+her", conf)
register("ddpg", conf)
