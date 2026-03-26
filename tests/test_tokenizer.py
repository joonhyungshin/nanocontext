def check_tokens(tokenizer, tokens, d, height):
    assert tokens[0] == tokenizer.bos_token
    assert len(tokens) == (d + 1) * (d ** (height - 1))
    for i in range(1, len(tokens)):
        token = tokens[i]
        if i % (d + 1) == 0:
            assert tokenizer.is_variable_token(token, tokenizer.PUNC_TOKEN_NAME)
        else:
            assert tokenizer.is_variable_token(token, tokenizer.VAL_TOKEN_NAME)


def test_simple_tokenizer(ising_tree, ising_tokenizer):
    d, height = ising_tree.d, ising_tree.height
    tokens = ising_tokenizer.tokenize(ising_tree, prepend_bos=True)
    check_tokens(ising_tokenizer, tokens, d, height)


def test_lazy_tokenizer(ising_lazy_tree, ising_tokenizer):
    d, height = ising_lazy_tree.d, ising_lazy_tree.height
    tokens = ising_tokenizer.tokenize_lazy(ising_lazy_tree, batch_height=6, prepend_bos=True)
    check_tokens(ising_tokenizer, tokens, d, height)


def test_start_from_middle_tokenizer(ising_tree, ising_tokenizer):
    tokens = ising_tokenizer.tokenize(ising_tree, prepend_bos=True)
    sfm_tokens = ising_tokenizer.tokenize(ising_tree, token_start_idx=10, prepend_bos=True)
    assert tokens[10:] == sfm_tokens
    tokens = ising_tokenizer.tokenize(ising_tree, prepend_bos=False)
    sfm_tokens = ising_tokenizer.tokenize(ising_tree, token_start_idx=7, prepend_bos=False)
    assert tokens[7:] == sfm_tokens


def test_batch_tokenizer(ising_lazy_tree, ising_tokenizer):
    ising_lazy_tree.get_subtree_or_sample(0, 0, keep_memory=True)
    tokens_one = ising_tokenizer.tokenize_lazy(ising_lazy_tree, batch_height=3, prepend_bos=True)
    tokens_two = ising_tokenizer.tokenize_lazy(ising_lazy_tree, batch_height=4, prepend_bos=True)
    tokens_three = ising_tokenizer.tokenize_lazy(ising_lazy_tree, batch_height=5, prepend_bos=True)
    assert tokens_one == tokens_two
    assert tokens_one == tokens_three


def test_start_from_middle_lazy_tokenizer(ising_lazy_tree, ising_tokenizer):
    ising_lazy_tree.get_subtree_or_sample(0, 0, keep_memory=True)
    tokens = ising_tokenizer.tokenize_lazy(ising_lazy_tree, batch_height=3, prepend_bos=True)
    sfm_tokens = ising_tokenizer.tokenize_lazy(ising_lazy_tree, batch_height=5, token_start_idx=12, prepend_bos=True)
    assert tokens[12:] == sfm_tokens


def test_decoder(ising_tree, ising_tokenizer):
    tokens = ising_tokenizer.tokenize(ising_tree, prepend_bos=True)
    trees = ising_tokenizer.decode_trees(tokens)
    assert len(trees) == 1
    assert list(trees[0].leaves_values_stream()) == list(ising_tree.leaves_values_stream())


def test_ising_seg_tokenizer(ising_lazy_tree, ising_seg_tokenizer):
    for token, summary in ising_seg_tokenizer.tokenize_with_summary_stream(ising_lazy_tree, range(10),
                                                                           batch_height=6, prepend_bos=False):
        pass


def test_ising_seg_start_from_middle_tokenizer(ising_lazy_tree, ising_seg_tokenizer):
    ising_lazy_tree.get_subtree_or_sample(0, 0, keep_memory=True)
    stream_one = ising_seg_tokenizer.tokenize_with_summary_stream(ising_lazy_tree, [13, 17], batch_height=6)
    stream_two = ising_seg_tokenizer.tokenize_with_summary_stream(ising_lazy_tree, [13, 17], batch_height=4,
                                                                  token_start_idx=10)
    for tok_sum_one, tok_sum_two in zip(stream_one, stream_two):
        assert tok_sum_one[1] == tok_sum_two[1]


def test_ising_cpt_start_from_middle_tokenizer(ising_lazy_tree, ising_cpt_tokenizer):
    ising_lazy_tree.get_subtree_or_sample(0, 0, keep_memory=True)
    stream_one = ising_cpt_tokenizer.tokenize_with_summary_stream(ising_lazy_tree, [13, 17, 22], batch_height=6)
    stream_two = ising_cpt_tokenizer.tokenize_with_summary_stream(ising_lazy_tree, [13, 17, 22], batch_height=4,
                                                                  token_start_idx=7)
    for tok_sum_one, tok_sum_two in zip(stream_one, stream_two):
        assert tok_sum_one[1] == tok_sum_two[1]


def test_coloring_cpt_start_from_middle_tokenizer(coloring_lazy_tree, coloring_cpt_tokenizer):
    coloring_lazy_tree.get_subtree_or_sample(0, 0, keep_memory=True)
    for i in range(729):
        stream_one = coloring_cpt_tokenizer.tokenize_with_summary_stream(coloring_lazy_tree, [i + 7, i + 20],
                                                                         batch_height=6)
        stream_two = coloring_cpt_tokenizer.tokenize_with_summary_stream(coloring_lazy_tree, [i + 7, i + 20],
                                                                         batch_height=6,
                                                                         token_start_idx=i)
        for tok_sum_one, tok_sum_two in zip(stream_one, stream_two):
            assert tok_sum_one[1] == tok_sum_two[1]
            assert len(tok_sum_two[0]) == 7 or tok_sum_two[0] == tok_sum_one[0]


def test_ising_seg_start_from_last(ising_lazy_tree, ising_seg_tokenizer):
    d, height = ising_lazy_tree.d, ising_lazy_tree.height
    num_tokens = (d ** (height - 1)) * (d + 1) - 1
    stream = ising_seg_tokenizer.tokenize_with_summary_stream(ising_lazy_tree, [], token_start_idx=num_tokens)
    tok_sums = list(stream)
    assert len(tok_sums) == 1
    tok_sum = tok_sums[0]
    assert len(tok_sum[0]) == 0


def test_ising_streamer(ising_seg_streamer, ising_cpt_streamer):
    seg_stream = ising_seg_streamer.tokenized_trees_with_summaries_stream(10, batch_height=5, token_start_idx=10)
    cpt_stream = ising_cpt_streamer.tokenized_trees_with_summaries_stream(10, batch_height=4, token_start_idx=20)
    _, tokens, _ = next(seg_stream)
    assert len(tokens) == 0
    _, tokens, _ = next(cpt_stream)
    assert len(tokens) == 0
    for _ in range(10):
        _, tokens, _ = next(seg_stream)
        assert len(tokens) == 10
        _, tokens, _ = next(cpt_stream)
        assert len(tokens) == 10
