import torch.nn as nn
from .Attention import MultiHeadAttention
from .CoreLayers import FeedForward, LayerNormalization


class EncoderLayer(nn.Module):
    """
    An encoder layer in the stack of the encoder. This module takes as input the word embedding and perform the core
    transformations of the Transformer. The encoder layer performs the multi-head attention mechanism on the word
    embeddings inputted and then pass the outputs through a feed forward layer.
    """
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        """
        Inputs:
            d_model: the dimensionality of the outputs <hyperparameter>
            d_ff: the number of hidden units for our feed forward layer <hyperparameter>
            num_heads: the number of heads to split the attention down <hyperparameter>
            dropout: the rate of dropout regularization <hyperparameter>
        """
        super(EncoderLayer, self).__init__()
        # we first apply the multi-head attention mechanism
        self.inputs_normalization = LayerNormalization(d_model)    # we can also use nn.LayerNorm here
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)

        # we then pass the attention weighted embeddings through a feed forward network
        self.feed_forward_normalization = LayerNormalization(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(self, positional_encoded_word_embeddings, mask=None):
        """
        Encode the word embeddings by transforming them into hidden representations using the attention mechanism.

        Inputs:
            positional_encoded_word_embeddings: the word embeddings that are already encoded with positional information
            mask: the mask which is used within the attention mechanism to block out tokens that should be ignored

            The inputs should have shape:
            positional_encoded_word_embeddings: (batch_size, seq_len, d_model)
        Output:
            The encoded hidden representations of the word embeddings.
        """
        # we first normalize our inputs (check out the normalization module for more details)
        normalized_word_embeddings = self.inputs_normalization(positional_encoded_word_embeddings)

        # we then apply the attention computations to weigh our projected word embeddings by compatibility
        # this is essentially the same as passing normalized_word_embeddings 3 times -- just wanted to use one line
        attention_weights = self.self_attention(*([normalized_word_embeddings] * 3), mask)
        attention_weights = self.attention_dropout(attention_weights)    # regularization

        # we add the weights to the original word embeddings to apply the attention computations to them
        attention_weighted_embeddings = attention_weights + positional_encoded_word_embeddings

        # we then pass our attention-weighted embeddings through a feed forward network, but first we need to normalize
        # the activations
        normalized_weighted_embeddings = self.feed_forward_normalization(attention_weighted_embeddings)
        feed_forward_outputs = self.feed_forward(normalized_weighted_embeddings)
        feed_forward_outputs = self.feed_forward_dropout(feed_forward_outputs)

        # the outputs of the layer is the sum of the attention weighted embeddings and the feed forward outputs --
        # this is a subtlety that differs from the paper but is present in the tensor2tensor implementation

        return attention_weighted_embeddings + feed_forward_outputs


class Encoder(nn.Module):
    """
    This module implements the Encoder in the Transformer, which is a stack of encoder layers that are sequentially
    applied starting from the original word embeddings. The Encoder uses the multi-head attention mechanism.
    """
    def __init__(self, d_model, d_ff, num_heads, num_layers, dropout=0.1):
        """
        Inputs:
            d_model: the dimensionality of the outputs <hyperparameter>
            d_ff: the number of hidden units for our feed forward layer <hyperparameter>
            num_heads: the number of heads to split the attention down <hyperparameter>
            num_layers: the number of encoder layers to apply <hyperparameter>
            dropout: the rate of dropout regularization <hyperparameter>
        """
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)]
        )

        self.outputs_normalization = LayerNormalization(d_model)

    def forward(self, positional_encoded_word_embeddings, mask=None):
        """
        Apply the encoder layers to the word embeddings.

        Inputs:
            positional_encoded_word_embeddings: the word embeddings that are already encoded with positional information
            mask: the mask which is used within the attention mechanism to block out tokens that should be ignored

            The inputs should have shape:
            positional_encoded_word_embeddings: (batch_size, seq_len, d_model)
        Output:
            The encoded hidden representations of the word embeddings.
        """
        encoded_word_embeddings = positional_encoded_word_embeddings
        # noinspection PyTypeChecker
        for encode in self.encoder_layers:    # apply each layer sequentially
            encoded_word_embeddings = encode(encoded_word_embeddings)

        # normalize the encodings
        return self.outputs_normalization(encoded_word_embeddings)
