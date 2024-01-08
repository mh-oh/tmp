
from sklearn.cluster import DBSCAN, KMeans

from rldev.agents.pref.sac import DiagGaussianActor, QFunction
from rldev.configs.xconf import Conf
from rldev.configs.registry.registry import get, register


conf = get("sac")

conf.topK = 5

conf.fusion = 3
conf.activation = 'tanh'
conf.reward_lr = 0.0003
conf.budget = 50
conf.segment_length = 50

conf.num_unsup_steps = 9000
conf.num_interact = 5000
conf.reward_update = 10
conf.max_feedback = 10000
conf.label_margin = 0.0
conf.teacher_beta = -1
conf.teacher_gamma = 1
conf.teacher_eps_mistake = 0
conf.teacher_eps_skip = 0
conf.teacher_eps_equal = 0
conf.reward_schedule = 0

conf.query = Conf()
conf.query.mode = "uniform"
conf.query.kwargs = dict()

register("uniform", conf)

conf = get("uniform")
conf.segment_length = 100000
register("uniform-traj", conf)

conf = get("uniform-traj")
conf.query.mode = "entropy"
conf.query.kwargs = dict(scale=10)
register("entropy-traj", conf)

conf = get("uniform-traj")
conf.query.mode = "disagree"
conf.query.kwargs = dict(scale=10)
register("disagree-traj", conf)

conf = get("uniform-traj")
conf.query.mode = "uniform_aligned"
conf.query.kwargs = dict(cluster=KMeans(n_clusters=3))
register("uniform-kmeans-3-traj", conf)

conf = get("uniform-traj")
conf.query.mode = "entropy_aligned"
conf.query.kwargs = dict(cluster=KMeans(n_clusters=3))
register("entropy-kmeans-3-traj", conf)

conf = get("uniform-traj")
conf.query.mode = "entropy_aligned"
conf.query.kwargs = dict(cluster=KMeans(n_clusters=2))
register("entropy-kmeans-2-traj", conf)

