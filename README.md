# nanocontext
[nanochat](https://github.com/karpathy/nanochat) learns to broadcast on trees

## Overview

This is a project for training a language model on a synthetic language encoded by broadcasting on ordered trees. This is recursively constructed as the following.

- A value of the root is sampled.
- Given the value of a node, each of its children is sampled through the "broadcast channel," (conditionally) independently of its siblings.

Each non-leaf node represents a _context_ of the leaves in its subtree. For instance, a node at a small depth contains many leaves in its subtrees, representing a long context. This is analogous to how a document is organized: chapters, sections, paragraphs, sentences, and so on.

For simplicity, we will only train on perfect $d$-ary trees, which means that all leaves have a common depth $h$ and all non-leaf nodes have exactly $d$ children. A (poorly-trained) model might output a non-perfect tree, though.  

### Broadcast channels

The current project runs on the following two broadcasting channels.

- The **Ising** model, parametrized by the correlation strength $\rho\in[0,1]$.
  - The root has value + or - (which we call _spins_), each w.p. 1/2.
  - Given the spin of a node, each of its children inherits the value w.p. $\rho$ and resamples w.p. $1-\rho$.
- The **coloring** model, parametrized by the number of colors $q\in\mathbb{Z}$.
  - The root chooses one of the $q$ colors $\{1,\cdots,q\}$ uniformly at random.
  - Given the color of a node, each of its children chooses one of the $q-1$ colors not chosen by its parent, uniformly at random.


### Tokenizing a tree

In a real world document, there are clear indicators that mark a beginning of a new context: punctuation marks between sentences, indentations between paragraphs, and so on. Analogously, in addition to the values of the leaves, we insert "punctuations" whenever we start a new subtree, and the type of each punctuation depends on the height of the new subtree. For example, in the Ising model, the sequence
```
[BOS] + + 1 - + 2 - - 1 - +
```
encodes a perfect binary tree with leaves `+ + - + - - - +`. Note that `[BOS]` is a special token that indicates the beginning of a sequence. Ignoring the `[BOS]` token, we need $d^h+d^{h-1}-1$ tokens to encode the leaves of a perfect $d$-ary tree with height $h$.

### Training nanochat as a reasoning model

When the model generates a document (tree), if the context size is smaller than the number of leaves in a single tree ($d^h$), then the old tokens are forgotten and only the recent tokens are used. This leads to a distribution of total spin that is different from the ground truth.

We want the model to somehow memorize and leverage long context. Our strategy is to train the model to periodically manage its own _memory states_. We train the model so that given the current memory state, it can predict a next few tokens and the next memory state.

The crucial part is to design state transitions and to encode state information, efficient enough to fit in a small context size. Two approaches are used.

- "Segment" encoding. This is a bottom-up approach; as the model sees (either from the training sequence or from its own generated tokens) a new value and if that concludes a subtree of height $h_1$, the last $d^{h_1}$ tokens is summarized using the root value of that subtree. This approach requires $O(dh)$ space for encoding the memory state.
- "Path" encoding. The current state is encoded using the values of the ancestors of the leaf we wish to generate at a given time. This is a top-down approach; when the model generates a document, it starts by generating the root and goes all the way down to the new leaf. This approach uses only $O(h\log d)$ space so is more efficient.


## Running the code

Currently, the project is written purely in Python and the dependencies are managed by [Pixi](https://pixi.prefix.dev/latest/). After installing Pixi, you should prepend `pixi run -e dsi-cpu` or `pixi run -e dsi-cuda` to every run of a Python script. The latter will only work if CUDA is available. For instance, you can run the following to display a help text.
```shell
pixi run -e dsi-cpu python -m nanocontext --help
```
To run the tests:
```shell
pixi run -e macos python -m pytest
```

### Training nanochat

You can train nanochat on the broadcast model using the script `python -m nanocontext train` (prepend `pixi run` appropriately). The number of arguments is huge but the following are the most relevant. For other arguments, run `python -m nanocontext train --help`.

- Tree parameters
  - `-d`: number of children of each non-leaf node $d$. Defaults to 3, i.e., ternary trees.
  - `--height`: height (or depth) of the tree $h$.
- Broadcasting parameters (exactly one of the following two must be set)
  - `--rho`: correlation strength $\rho$, for the Ising model.
  - `-k`: number of colors $k$, for the coloring model.
- Model (nanochat) hyperparameters
  - `--context-size`: maximum context size the model can learn. Defaults to 2048.
  - `--layers`: number of transformer layers. Defaults to 20.
  - `--heads`: number of attention heads. If not given inferred from the number of layers.
  - `--model-dim`: embedding dimension of the tokens. If not given inferred from the number of layers.
- Training parameters
  - `--num-iterations`: number of training iterations. If not given inferred from the model size.
  - `--summary-mode`: train the model as a reasoning model (`segment`, `path`, or `disabled`)
- Logging
  - `--save-to`: path to save the trained model.
  - `--wandb`: [W&B](https://wandb.ai/site) logging mode (`online`, `offline`, or `disabled`). Defaults to `disabled`.

For example, the following command trains a 3-layers GPT model on ternary trees with height 3 sampled using the Ising broadcast channel with $\rho=0.9$, and saves the model to `model.pt`.
```shell
pixi run -e dsi-cpu python -m nanocontext train -d 3 --height 3 --rho 0.9 --layers 3 --num-iterations 300 --device-batch-size 1 --total-batch-size 512 --context-size 128 --save-to model.pt
```

Remember that to use GPUs you should change the environment from `dsi-cpu` to `dsi-cuda`. Also, you would want to increase the model and batch sizes to leverage the better hardware. The default configuration is suitable with this environment.
```shell
pixi run -e dsi-cuda python -m nanocontext train -d 3 --height 3 --rho 0.9 --layers 10 --num-iterations 3000 --context-size 1024 --save-to model.pt
```

### Generating trees

Once nanochat is trained, you can ask it to generate a tree using the script `python -m nanocontext generate`.

- Tree parameters (same as the training)
  - `-d`: number of children of each non-leaf node $d$. Defaults to 3, i.e., ternary trees.
  - `--height`: height (or depth) of the tree $h$.
- Model loading
  - `--model-path`: path to the trained model.
- Sampling parameters
  - `--max-tokens`: maximum number of tokens to generate.
  - `--temperature`: sampling temperature. Defaults to 1.
  - `--samples`: number of samples to generate. Defaults to 1.
  - `--patch`: patch the punctuation tokens to try to make the tree perfect: see below.

The following commands generates 5 samples with the saved model. Use the same $d$ and $h$ the model was trained on.
```shell
pixi run -e dsi-cpu python -m nanocontext generate --model-path model.pt --max-tokens 100 -d 3 --height 3 --samples 5
```


### Evaluating nanochat

We use different evaluation metric for the two broadcasting models.

- For the **Ising** model, we plot the distribution of the sum of spins (_total spin_ in short) of the leafs of a generated tree. Since it takes exactly $d^h+d^{h-1}-1$ tokens to encode a tree, we let the model generate $d^h+d^{h-1}-1$ tokens and count the number of `+` tokens and `-` tokens (even if the generated language is ill-formed).
- For the **coloring** model, we construct a perfect tree whose leaves are those sampled from the model, and try to reconstruct the color of the root. We plot the proportion of the trees where the color of the root can be successfully reconstructed.

Note that when the context size of a model is much smaller than $d^h+d^{h-1}-1$, the model might not know what punctuation token to generate during sampling. For this scenario, especially in the coloring experiment, we ignore the correctness of the punctuation tokens and just map the generated colors to the leaves of an imaginary perfect tree. From the implementation perspective, this is done through "patching" the punctuation tokens by replacing them with the correct ones.

You can evaluate the trained model by running `python -m nanocontext evaluate`.

- Tree parameters (same as the training)
  - `-d`: number of children of each non-leaf node $d$. Defaults to 3, i.e., ternary trees.
  - `--height`: height (or depth) of the tree $h$.
- Model loading
  - `--model-path`: path to the trained model.
- Sampling parameters
  - `--samples`: number of samples to generate. Defaults to 1024.

Similar to the generation, make sure to use the same $d$ and $h$ the model was trained on.
```shell
pixi run -e dsi-cpu python -m nanocontext evaluate --model-path model.pt -d 3 --height 
```


### Simulating block autoregressive process

Trained models with context size smaller than the number of leaves are expected to sample from the distribution similar to the _block autoregressive process_, which is a Markovian version of the ground truth language. Namely, for some $0\leq w\leq h$, we sample $d^{h-w}$ times the leaves of a subtree of height $w$, but each time we sample we condition only on the most recently generated subtree. Specifically, this process can be understood as a theoretical approximation of the behavior of the autoregressive model with context size $d^w$.

This purpose of this command to compare the true model with the block autoregressive process. You can generate from the exact block autoregressive process and go through the same evaluation procedure by running `python -m nanocontext simulate`.

- Tree parameters
  - `-d`: number of children of each non-leaf node $d$. Defaults to 3, i.e., ternary trees.
  - `--height`: height (or depth) of the tree $h$.
  - `--markov-height`: height (or depth) of the subtree $w$ for each block.
- Broadcasting parameters (exactly one of the following two must be set)
  - `--rho`: correlation strength $\rho$, for the Ising model.
  - `-k`: number of colors $k$, for the coloring model.
- Sampling parameters
  - `--samples`: number of samples to generate. Defaults to 1024.

For instance, the following command simulates the block autoregressive process for a 3-coloring experiment on 4-ary tree with height 6, where each step we generate a subtree of height 5.
```shell
pixi run -e dsi-cpu python -m nanocontext simulate -d 4 --height 6 -k 3 --markov-height 5 --samples 1000
```

## TODO

- Checkpointing
