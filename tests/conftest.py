import pytest

from nanocontext.data.broadcast_tree.loader import BroadcastTreeStreamer
from nanocontext.tree import (
    BroadcastTree, PerfectTreeConfig, IsingBroadcastPolicy, LazyBroadcastTree, ColoringBroadcastPolicy
)
from nanocontext.data.broadcast_tree import (
    PerfectTreeTokenizer, SegmentSummaryTokenizer, PathSummaryTokenizer,
    broadcast_tree_data_loader
)


@pytest.fixture
def config():
    return PerfectTreeConfig(d=3, height=6)


@pytest.fixture
def ising_policy():
    return IsingBroadcastPolicy(rho=0.5)


@pytest.fixture
def coloring_policy():
    return ColoringBroadcastPolicy(k=3)


@pytest.fixture
def ising_tree(config, ising_policy):
    return BroadcastTree(config, ising_policy)


@pytest.fixture
def ising_lazy_tree(config, ising_policy):
    return LazyBroadcastTree(config, ising_policy)


@pytest.fixture
def coloring_tree(config, coloring_policy):
    return BroadcastTree(config, coloring_policy)


@pytest.fixture
def coloring_lazy_tree(config, coloring_policy):
    return LazyBroadcastTree(config, coloring_policy)


@pytest.fixture
def ising_tokenizer(ising_policy):
    return PerfectTreeTokenizer(64, ising_policy.get_domain())


@pytest.fixture
def coloring_tokenizer(coloring_policy):
    return PerfectTreeTokenizer(64, coloring_policy.get_domain())


@pytest.fixture
def ising_seg_tokenizer(ising_policy):
    return SegmentSummaryTokenizer(64, ising_policy.get_domain())


@pytest.fixture
def coloring_seg_tokenizer(coloring_policy):
    return SegmentSummaryTokenizer(64, coloring_policy.get_domain())


@pytest.fixture
def ising_cpt_tokenizer(ising_policy):
    return PathSummaryTokenizer(64, ising_policy.get_domain())


@pytest.fixture
def coloring_cpt_tokenizer(coloring_policy):
    return PathSummaryTokenizer(64, coloring_policy.get_domain())


@pytest.fixture
def ising_streamer(config, ising_policy, ising_tokenizer):
    return BroadcastTreeStreamer(ising_tokenizer, config, ising_policy)


@pytest.fixture
def coloring_streamer(config, coloring_policy, coloring_tokenizer):
    return BroadcastTreeStreamer(coloring_tokenizer, config, coloring_policy)


@pytest.fixture
def ising_seg_streamer(config, ising_policy, ising_seg_tokenizer):
    return BroadcastTreeStreamer(ising_seg_tokenizer, config, ising_policy)


@pytest.fixture
def coloring_seg_streamer(config, coloring_policy, coloring_seg_tokenizer):
    return BroadcastTreeStreamer(coloring_seg_tokenizer, config, coloring_policy)


@pytest.fixture
def ising_cpt_streamer(config, ising_policy, ising_cpt_tokenizer):
    return BroadcastTreeStreamer(ising_cpt_tokenizer, config, ising_policy)


@pytest.fixture
def coloring_cpt_streamer(config, coloring_policy, coloring_cpt_tokenizer):
    return BroadcastTreeStreamer(coloring_cpt_tokenizer, config, coloring_policy)


@pytest.fixture
def ising_data_loader(config, ising_policy, ising_tokenizer):
    data_loader = broadcast_tree_data_loader(ising_tokenizer, config, ising_policy,
                                             16, 64, summary_every=None)
    return data_loader


@pytest.fixture
def ising_data_sample_loader(config, ising_policy, ising_tokenizer):
    data_loader = broadcast_tree_data_loader(ising_tokenizer, config, ising_policy,
                                             16, 64, summary_every=None, mode="sample")
    return data_loader


@pytest.fixture
def coloring_data_loader(config, coloring_policy, coloring_tokenizer):
    data_loader = broadcast_tree_data_loader(coloring_tokenizer, config, coloring_policy,
                                             16, 64, summary_every=None)
    return data_loader


@pytest.fixture
def ising_cpt_data_loader(config, ising_policy, ising_cpt_tokenizer):
    summary_len = len(ising_cpt_tokenizer.init_summary_tokens(config))
    seq_len = 64
    summary_every = seq_len + 1 - 2 * summary_len
    data_loader = broadcast_tree_data_loader(ising_cpt_tokenizer, config, ising_policy,
                                             16, 64, summary_every=summary_every)
    return data_loader


@pytest.fixture
def ising_cpt_sample_data_loader(config, ising_policy, ising_cpt_tokenizer):
    summary_len = len(ising_cpt_tokenizer.init_summary_tokens(config))
    seq_len = 64
    summary_every = seq_len + 1 - 2 * summary_len
    data_loader = broadcast_tree_data_loader(ising_cpt_tokenizer, config, ising_policy,
                                             16, 64, mode="sample", summary_every=summary_every)
    return data_loader


@pytest.fixture
def coloring_cpt_sample_data_loader(config, coloring_policy, coloring_cpt_tokenizer):
    summary_len = len(coloring_cpt_tokenizer.init_summary_tokens(config))
    seq_len = 64
    summary_every = seq_len + 1 - 2 * summary_len
    data_loader = broadcast_tree_data_loader(coloring_cpt_tokenizer, config, coloring_policy,
                                             16, 64, mode="sample", summary_every=summary_every)
    return data_loader


@pytest.fixture
def coloring_cpt_stream_data_loader(config, coloring_policy, coloring_cpt_tokenizer):
    summary_len = len(coloring_cpt_tokenizer.init_summary_tokens(config))
    seq_len = 64
    summary_every = seq_len + 1 - 2 * summary_len
    data_loader = broadcast_tree_data_loader(coloring_cpt_tokenizer, config, coloring_policy,
                                             16, 64, mode="stream", summary_every=summary_every)
    return data_loader
