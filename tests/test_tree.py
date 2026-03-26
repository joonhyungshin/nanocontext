def test_ancestors(ising_tree):
    assert ising_tree.get_ancestors_of_leaf(3) == ising_tree.get_ancestors_of_leaf(4)


def test_coloring(coloring_tree):
    coloring_tree.sample()
    for depth in range(1, coloring_tree.height + 1):
        for idx in range(coloring_tree.d ** depth):
            node = (depth, idx)
            parent = (depth - 1, idx // coloring_tree.d)
            assert coloring_tree.get_value(node) != coloring_tree.get_value(parent)


def test_coloring_lazy(coloring_lazy_tree):
    for leaf_idx in range(10, 500, 7):
        leaf = coloring_lazy_tree.get_value((coloring_lazy_tree.height, leaf_idx))
        values = coloring_lazy_tree.get_ancestors_of_leaf(leaf_idx)
        values.append(leaf)
        for i in range(coloring_lazy_tree.height):
            assert values[i] != values[i + 1]
