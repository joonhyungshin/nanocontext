import pytest

from nanocontext.data.broadcast_tree.loader import BroadcastTreeStreamer
from nanocontext.tree import (
    BroadcastTree, PerfectTreeConfig, IsingBroadcastChannel, LazyBroadcastTree, ColoringBroadcastChannel
)
from nanocontext.data.broadcast_tree import (
    PerfectTreeTokenizer, SegmentSummaryTokenizer, PathSummaryTokenizer,
    broadcast_tree_data_loader
)


@pytest.fixture
def config():
    return PerfectTreeConfig(d=3, height=6)


@pytest.fixture
def ising_channel():
    return IsingBroadcastChannel(rho=0.5)


@pytest.fixture
def coloring_channel():
    return ColoringBroadcastChannel(k=3)


@pytest.fixture
def ising_tree(config, ising_channel):
    return BroadcastTree(config, ising_channel)


@pytest.fixture
def ising_lazy_tree(config, ising_channel):
    return LazyBroadcastTree(config, ising_channel)


@pytest.fixture
def coloring_tree(config, coloring_channel):
    return BroadcastTree(config, coloring_channel)


@pytest.fixture
def coloring_lazy_tree(config, coloring_channel):
    return LazyBroadcastTree(config, coloring_channel)


@pytest.fixture
def ising_tokenizer(ising_channel):
    return PerfectTreeTokenizer(64, ising_channel.get_state_space())


@pytest.fixture
def coloring_tokenizer(coloring_channel):
    return PerfectTreeTokenizer(64, coloring_channel.get_state_space())


@pytest.fixture
def ising_seg_tokenizer(ising_channel):
    return SegmentSummaryTokenizer(64, ising_channel.get_state_space())


@pytest.fixture
def coloring_seg_tokenizer(coloring_channel):
    return SegmentSummaryTokenizer(64, coloring_channel.get_state_space())


@pytest.fixture
def ising_cpt_tokenizer(ising_channel):
    return PathSummaryTokenizer(64, ising_channel.get_state_space())


@pytest.fixture
def coloring_cpt_tokenizer(coloring_channel):
    return PathSummaryTokenizer(64, coloring_channel.get_state_space())


@pytest.fixture
def ising_streamer(config, ising_channel, ising_tokenizer):
    return BroadcastTreeStreamer(ising_tokenizer, config, ising_channel)


@pytest.fixture
def coloring_streamer(config, coloring_channel, coloring_tokenizer):
    return BroadcastTreeStreamer(coloring_tokenizer, config, coloring_channel)


@pytest.fixture
def ising_seg_streamer(config, ising_channel, ising_seg_tokenizer):
    return BroadcastTreeStreamer(ising_seg_tokenizer, config, ising_channel)


@pytest.fixture
def coloring_seg_streamer(config, coloring_channel, coloring_seg_tokenizer):
    return BroadcastTreeStreamer(coloring_seg_tokenizer, config, coloring_channel)


@pytest.fixture
def ising_cpt_streamer(config, ising_channel, ising_cpt_tokenizer):
    return BroadcastTreeStreamer(ising_cpt_tokenizer, config, ising_channel)


@pytest.fixture
def coloring_cpt_streamer(config, coloring_channel, coloring_cpt_tokenizer):
    return BroadcastTreeStreamer(coloring_cpt_tokenizer, config, coloring_channel)


@pytest.fixture
def ising_data_loader(config, ising_channel, ising_tokenizer):
    data_loader = broadcast_tree_data_loader(ising_tokenizer, config, ising_channel,
                                             16, 64, summary_every=None)
    return data_loader


@pytest.fixture
def ising_data_sample_loader(config, ising_channel, ising_tokenizer):
    data_loader = broadcast_tree_data_loader(ising_tokenizer, config, ising_channel,
                                             16, 64, summary_every=None, mode="sample")
    return data_loader


@pytest.fixture
def coloring_data_loader(config, coloring_channel, coloring_tokenizer):
    data_loader = broadcast_tree_data_loader(coloring_tokenizer, config, coloring_channel,
                                             16, 64, summary_every=None)
    return data_loader


@pytest.fixture
def ising_cpt_data_loader(config, ising_channel, ising_cpt_tokenizer):
    summary_len = len(ising_cpt_tokenizer.init_summary_tokens(config))
    seq_len = 64
    summary_every = seq_len + 1 - 2 * summary_len
    data_loader = broadcast_tree_data_loader(ising_cpt_tokenizer, config, ising_channel,
                                             16, 64, summary_every=summary_every)
    return data_loader


@pytest.fixture
def ising_cpt_sample_data_loader(config, ising_channel, ising_cpt_tokenizer):
    summary_len = len(ising_cpt_tokenizer.init_summary_tokens(config))
    seq_len = 64
    summary_every = seq_len + 1 - 2 * summary_len
    data_loader = broadcast_tree_data_loader(ising_cpt_tokenizer, config, ising_channel,
                                             16, 64, mode="sample", summary_every=summary_every)
    return data_loader


@pytest.fixture
def coloring_cpt_sample_data_loader(config, coloring_channel, coloring_cpt_tokenizer):
    summary_len = len(coloring_cpt_tokenizer.init_summary_tokens(config))
    seq_len = 64
    summary_every = seq_len + 1 - 2 * summary_len
    data_loader = broadcast_tree_data_loader(coloring_cpt_tokenizer, config, coloring_channel,
                                             16, 64, mode="sample", summary_every=summary_every)
    return data_loader


@pytest.fixture
def coloring_cpt_stream_data_loader(config, coloring_channel, coloring_cpt_tokenizer):
    summary_len = len(coloring_cpt_tokenizer.init_summary_tokens(config))
    seq_len = 64
    summary_every = seq_len + 1 - 2 * summary_len
    data_loader = broadcast_tree_data_loader(coloring_cpt_tokenizer, config, coloring_channel,
                                             16, 64, mode="stream", summary_every=summary_every)
    return data_loader
