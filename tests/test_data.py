def test_ising_loader(ising_data_loader):
    for _ in range(10):
        x, y = next(ising_data_loader)
        for i in range(x.shape[0]):
            assert (x[i, 1:] == y[i, :-1]).all()


def test_ising_sum_loader(config, ising_cpt_tokenizer, ising_cpt_data_loader):
    summary_len = len(ising_cpt_tokenizer.init_summary_tokens(config))
    for _ in range(10):
        x, y = next(ising_cpt_data_loader)
        for i in range(x.shape[0]):
            assert x[i, 0] == ising_cpt_tokenizer.summary_start_token
            assert x[i, summary_len - 1] == ising_cpt_tokenizer.summary_end_token
            assert y[i, -1] == ising_cpt_tokenizer.summary_end_token
            assert y[i, -summary_len] == ising_cpt_tokenizer.summary_start_token
            for j in range(1, x.shape[1] - 1):
                if j < summary_len - 1:
                    assert y[i, j] == -1
                else:
                    assert x[i, j + 1] == y[i, j]


def test_ising_sum_sample_loader(config, ising_cpt_tokenizer, ising_cpt_sample_data_loader):
    summary_len = len(ising_cpt_tokenizer.init_summary_tokens(config))
    for _ in range(10):
        x, y = next(ising_cpt_sample_data_loader)
        for i in range(x.shape[0]):
            assert x[i, 0] == ising_cpt_tokenizer.summary_start_token
            assert x[i, summary_len - 1] == ising_cpt_tokenizer.summary_end_token
            assert y[i, -1] == ising_cpt_tokenizer.summary_end_token
            assert y[i, -summary_len] == ising_cpt_tokenizer.summary_start_token
            for j in range(1, x.shape[1] - 1):
                if j < summary_len - 1:
                    assert y[i, j] == -1
                else:
                    assert x[i, j + 1] == y[i, j]
