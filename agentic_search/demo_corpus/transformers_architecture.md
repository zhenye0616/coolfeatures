# The Transformer Architecture

The Transformer architecture was introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al. at Google Brain. It replaced recurrent neural networks (RNNs) and long short-term memory (LSTM) networks as the dominant architecture for sequence-to-sequence tasks.

## Core Components

The Transformer consists of an encoder and a decoder, each composed of stacked layers. The encoder processes the input sequence and produces a set of continuous representations. The decoder generates the output sequence one token at a time, attending to both the encoder output and previously generated tokens.

Each encoder layer contains two sub-layers: a multi-head self-attention mechanism and a position-wise feed-forward network. Each decoder layer adds a third sub-layer: cross-attention over the encoder's output. Residual connections and layer normalization are applied around each sub-layer.

## Self-Attention Mechanism

Self-attention computes a weighted sum of value vectors, where the weights are determined by the compatibility between query and key vectors. For each position in the sequence, the model learns to attend to relevant positions across the entire input.

The attention function operates on queries (Q), keys (K), and values (V):

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

The scaling factor sqrt(d_k) prevents the dot products from growing too large, which would push the softmax into regions with extremely small gradients.

## Multi-Head Attention

Rather than performing a single attention function, the Transformer uses multi-head attention. This projects the queries, keys, and values h times with different learned linear projections, performs attention in parallel, and concatenates the results.

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this capability.

## Positional Encoding

Since the Transformer contains no recurrence and no convolution, it needs some way to make use of the order of the sequence. Positional encodings are added to the input embeddings at the bottom of the encoder and decoder stacks. The original paper used sinusoidal functions of different frequencies.

## Impact

The Transformer architecture enabled a new generation of language models, including BERT (2018), GPT-2 (2019), T5 (2019), and GPT-3 (2020). Its parallelizable nature made it significantly faster to train than RNNs on modern GPU hardware. The architecture has since been adapted for computer vision (Vision Transformer), audio processing, and protein structure prediction (AlphaFold 2).
