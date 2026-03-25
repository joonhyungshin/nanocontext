from collections import deque


def uniform_slices_from_concatenation(generator, size):
    token_buffer = deque()
    while True:
        while len(token_buffer) < size:
            contents = next(generator)
            try:
                token_buffer.extend(contents)
            except TypeError:
                token_buffer.append(contents)
        tokens = [token_buffer.popleft() for _ in range(size)]
        yield tokens
