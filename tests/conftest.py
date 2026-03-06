import pytest

from nanocontext.tree import (
    BroadcastTree, PerfectTreeConfig, IsingBroadcastPolicy, LazyBroadcastTree, ColoringBroadcastPolicy
)
from nanocontext.data.broadcast_tree import (
    BroadcastTreeTokenizer, SegmentSummaryTokenizer, HierarchySummaryTokenizer,
    broadcast_tree_data_loader
)


@pytest.fixture
def config():
    return PerfectTreeConfig(d=3, height=10)


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
    return BroadcastTreeTokenizer(64, ising_policy)


@pytest.fixture
def coloring_tokenizer(coloring_policy):
    return BroadcastTreeTokenizer(64, coloring_policy)


@pytest.fixture
def ising_seg_tokenizer(ising_policy):
    return SegmentSummaryTokenizer(64, ising_policy)


@pytest.fixture
def coloring_seg_tokenizer(coloring_policy):
    return SegmentSummaryTokenizer(64, coloring_policy)


@pytest.fixture
def ising_cpt_tokenizer(ising_policy):
    return HierarchySummaryTokenizer(64, ising_policy)


@pytest.fixture
def coloring_cpt_tokenizer(coloring_policy):
    return HierarchySummaryTokenizer(64, coloring_policy)


@pytest.fixture
def ising_data_loader(config, ising_tokenizer):
    data_loader = broadcast_tree_data_loader(ising_tokenizer, config, 16, 64, summary=False)
    return data_loader


@pytest.fixture
def coloring_data_loader(config, coloring_tokenizer):
    data_loader = broadcast_tree_data_loader(coloring_tokenizer, config, 16, 64, summary=False)
    return data_loader


@pytest.fixture
def ising_cpt_data_loader(config, ising_cpt_tokenizer):
    data_loader = broadcast_tree_data_loader(ising_cpt_tokenizer, config, 16, 64, summary=True)
    return data_loader
