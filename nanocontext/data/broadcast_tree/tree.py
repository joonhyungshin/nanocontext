from nanocontext.tree import PerfectTreeConfig, BroadcastPolicy, BroadcastForest
from nanocontext.utils import d_order


def block_autoregressive_forest(config: PerfectTreeConfig, policy: BroadcastPolicy, batch_height=None, num_trees=1):
    d, height = config.d, config.height
    batch_height = min(height, batch_height) if batch_height is not None else height
    batch_depth = height - batch_height
    forest = None
    for tree_idx in range(d ** batch_depth):
        if tree_idx == 0:
            root_values = None
        else:
            lca_height = d_order(tree_idx, d) + 1
            root_values = forest.get_root_values()
            for _ in range(2 * lca_height):
                root_values = policy.broadcast(root_values)
        batch_conf = PerfectTreeConfig(d, batch_height)
        forest = BroadcastForest(batch_conf, policy, root_values=root_values, num_trees=num_trees)
        forest.sample()
        yield forest


def block_autoregressive_tree(config: PerfectTreeConfig, policy: BroadcastPolicy, batch_height=None):
    for forest in block_autoregressive_forest(config, policy, batch_height=batch_height):
        yield forest[0]
