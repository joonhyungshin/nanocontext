# nanocontext
[nanochat](https://github.com/karpathy/nanochat) learns to broadcast on trees

## Overview

This is a project for training a language model on a synthetic language encoded by broadcasting on ordered trees. This is recursively constructed as the following.

- The root has value + or -, each w.p. 1/2.
- Given the value of a node, each of its children inherits the value w.p. $\rho$ and resamples w.p. $1-\rho$, independently of its siblings.

Each non-leaf node represents a _context_ of the leaves in its subtree. For instance, a node at a small depth contains many leaves in its subtrees, representing a long context. This is analogous to how a document is organized: chapters, sections, paragraphs, sentences, and so on.

For simplicity, we will only train on perfect $d$-ary trees, which means that all leaves have a common depth $h$ and all non-leaf nodes have exactly $d$ children. A (poorly-trained) model might output a non-perfect tree, though.  

### Tokenizing a tree

In a real world document, there are clear indicators that mark a beginning of a new context: punctuation marks between sentences, indentations between paragraphs, and so on. Analogously, in addition to the values of the leaves, we insert "punctuations" whenever we start a new subtree, and the type of each punctuation depends on the height of the new subtree. For example, the sequence
```
[BOS] + + 1 - + 2 - - 1 - +
```
encodes a perfect binary tree with leaves `+ + - + - - - +`. Note that `[BOS]` is a special token that indicates the beginning of a sequence.


## Running the code

Currently, the project is written purely in Python and the dependencies are managed by [Pixi](https://pixi.prefix.dev/latest/). After installing Pixi, you should prepend `pixi run -e dsi-cpu` or `pixi run -e dsi-cuda` to every run of a Python script. The latter will only work if CUDA is available. For instance, you can run the following to display a help text.
```shell
pixi run -e dsi-cpu python -m nanocontext --help
```

### Training nanochat

You can train nanochat on the broadcast model using the script `python -m nanocontext train` (prepend `pixi run` appropriately). The number of arguments is huge but the following are the most relevant.

- Tree parameters
  - `-d`: number of children of each non-leaf node $d$. Defaults to 3, i.e., ternary trees.
  - `--rho`: correlation strength $\rho$.
  - `--height`: height (or depth) of the tree $h$.
- Model (nanochat) hyperparameters
  - `--context-size`: maximum context size the model can learn. Defaults to 2048.
  - `--layers`: number of transformer layers. Defaults to 20.
  - `--heads`: number of attention heads. If not given inferred from the number of layers.
  - `--model-dim`: embedding dimension of the tokens. If not given inferred from the number of layers.
- Training parameters
  - `--num-iterations`: number of training iterations. If not given inferred from the model size.
- Logging
  - `--save-to`: path to save the trained model.
  - `--wandb`: [W&B](https://wandb.ai/site) logging mode (`online`, `offline`, or `disabled`). Defaults to `online`.

### Generating tree

Once nanochat is trained, you can ask it to generate a tree using the script `python -m nanocontext generate`.

- Model (nanochat) hyperparameters: same as above.
- Model loading
  - `--model-path`: path to the trained model.
- Sampling parameters
  - `--max-tokens`: maximum number of tokens to generate.
  - `--temperature`: sampling temperature. Defaults to 1.
  - `--samples`: number of samples to generate. Defaults to 1.

Currently, you must specify the model hyperparameters same as the above training step. This is planned to be reworked in the future.


## TODO

- Checkpointing
- Other evaluation metrics...
- Teach nanochat to use memory? Maybe chain-of-thoughts?
