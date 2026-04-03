from dataclasses import dataclass


class AbstractOrderedTree:
    def children_stream(self, node):
        raise NotImplementedError

    def get_parent(self, node):
        raise NotImplementedError

    def get_children_list(self, node):
        return list(self.children_stream(node))

    def num_children(self, node):
        return len(self.get_children_list(node))

    def get_root(self):
        raise NotImplementedError

    def get_value(self, node):
        raise NotImplementedError

    def value_to_char(self, value):
        raise NotImplementedError

    def value_domain_size(self):
        raise NotImplementedError

    def ancestors_stream(self, node):
        node = self.get_parent(node)
        while node is not None:
            yield node
            node = self.get_parent(node)

    def get_ancestors_values(self, node):
        ancestors = [self.get_value(ancestor) for ancestor in self.ancestors_stream(node)]
        ancestors.reverse()
        return ancestors

    def get_root_value(self):
        return self.get_value(self.get_root())

    def subtree_leaves_stream(self, node):
        children = self.get_children_list(node)
        if children:
            for child in children:
                yield from self.subtree_leaves_stream(child)
        else:
            yield node

    def leaves_stream(self):
        yield from self.subtree_leaves_stream(self.get_root())

    def leaves_values_stream(self):
        for leaf in self.leaves_stream():
            yield self.get_value(leaf)

    def _draw_node(self, node, canvas, canvas_idx, depth):
        canvas[depth] += " " * (canvas_idx - len(canvas[depth]))
        canvas[depth] += self.value_to_char(node.value)
        children = self.get_children_list(node)
        if children:
            if len(canvas) == depth + 1:
                canvas.append("")
            for child in children:
                canvas_idx = self._draw_node(child, canvas, canvas_idx, depth + 1)
        else:
            canvas_idx += 1
        return canvas_idx

    def __str__(self):
        canvas = [""]
        self._draw_node(self.get_root(), canvas, 0, 0)
        return "\n".join(canvas)

class LinkedOrderedTree(AbstractOrderedTree):
    class Node:
        def __init__(self, value=None):
            self.parent = None
            self.children = []
            self.value = value

        def add_child(self, child):
            self.children.append(child)
            child.parent = self

        def create_child(self, value=None):
            child = LinkedOrderedTree.Node(value)
            self.add_child(child)
            return child

        def get_parent_or_create(self):
            created = False
            if self.parent is None:
                self.parent = LinkedOrderedTree.Node()
                self.parent.add_child(self)
                created = True
            return self.parent, created

        def traverse(self):
            yield self
            for child in self.children:
                yield from child.traverse()

    def is_singleton(self):
        return len(self.root.children) == 0

    def __init__(self, domain=None):
        self.root = self.Node()
        self.domain = domain

    def get_root(self):
        return self.root

    def get_value(self, node):
        return node.value

    def get_parent(self, node):
        return node.parent

    def children_stream(self, node):
        return iter(node.children)

    def get_children_list(self, node):
        return node.children

    def leaves_stream(self):
        for node in self.root.traverse():
            if not node.children:
                yield node

    def value_to_char(self, value):
        if value is None or self.domain is None:
            return "#"
        return self.domain.value_to_char(value)


@dataclass
class PerfectTreeConfig:
    d: int
    height: int


class AbstractPerfectTree(AbstractOrderedTree):
    def __init__(self, config: PerfectTreeConfig):
        super().__init__()
        self.config = config
        self.d = config.d
        self.height = config.height

    @property
    def num_leaves(self):
        return self.d ** self.height

    def value_at(self, depth, idx):
        raise NotImplementedError

    def get_value(self, node):
        return self.value_at(*node)

    def get_root(self):
        return 0, 0

    def get_parent(self, node):
        depth, idx = node
        if depth == 0:
            return None
        return depth - 1, idx // self.d

    def children_stream(self, node):
        depth, idx = node
        if depth < self.height:
            for i in range(self.d):
                yield depth + 1, idx * self.d + i

    def num_children(self, node):
        depth, idx = node
        return self.d if depth < self.height else 0

    def segment_stream(self, start, end):
        assert 0 <= start <= end <= self.num_leaves
        idx = start
        while idx < end:
            segment_depth = self.height
            segment_len = 1
            segment_idx = idx
            while segment_idx % self.d == 0 and idx + segment_len * self.d <= end:
                segment_depth -= 1
                segment_len *= self.d
                segment_idx //= self.d
            yield segment_depth, segment_idx
            idx += segment_len

    def subtree_stream(self, depth, idx, batch_height=None):
        subtree = self.get_subtree(depth, idx)
        target_height = subtree.height
        batch_height = min(target_height, batch_height) if batch_height is not None else target_height
        batch_depth = target_height - batch_height
        for batch_idx in range(self.d ** batch_depth):
            batch_node = (batch_depth, batch_idx)
            yield subtree.get_subtree(batch_depth, batch_idx), subtree.get_ancestors_values(batch_node)

    def leaves_stream(self, start_idx=0):
        for idx in range(start_idx, self.d ** self.height):
            yield self.height, idx

    def leaves_values_stream(self, start_idx=0):
        for leaf in self.leaves_stream(start_idx=start_idx):
            yield self.get_value(leaf)

    def get_ancestors_of_leaf(self, leaf_idx):
        leaf = (self.height, leaf_idx)
        return self.get_ancestors_values(leaf)

    def get_subtree(self, depth, idx):
        return PerfectSubtree(self, depth, idx) if depth > 0 else self


class PerfectSubtree(AbstractPerfectTree):
    def __init__(self, tree: AbstractPerfectTree, depth, idx):
        super().__init__(PerfectTreeConfig(d=tree.d, height=tree.height - depth))
        self.tree = tree
        self.depth = depth
        self.idx = idx

    def value_at(self, depth, idx):
        return self.tree.value_at(self.depth + depth, self.idx * (self.d ** depth) + idx)


class ValueDomain:
    def get_size(self):
        raise NotImplementedError

    def value_to_index(self, value):
        raise NotImplementedError

    def index_to_value(self, index):
        raise NotImplementedError

    def value_to_char(self, value):
        raise NotImplementedError
