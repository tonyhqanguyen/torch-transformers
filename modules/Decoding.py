import torch.nn as nn
from .Attention import MultiHeadAttention
from .CoreLayers import FeedForward, LayerNormalization


class DecoderLayer(nn.Module):
    """
    A decoder layer in the stack of the decoder. This module takes output from
    the encoder, applies the attention mechanism and decode the outputs of the
    sequence-to-sequence model. Here, there are 2 attention layers, since the
    decoder uses the decoded outputs to the "current" time step to decode the
    next time step output, we need to apply attention over the decoded output,
    and we also need to apply attention to weigh the importance of the input --
    the output given by the encoder. After the attention mechanism is performed,
    we pass the outputs to a feed forward layer.
    """
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        """
        Inputs:
            d_model: the dimensionality of the outputs <hyperparameter>
            d_ff: the number of hidden units for our feed forward layer
                <hyperparameter>
            num_heads: the number of heads to split the attention down
                <hyperparameter>
            dropout: the rate of dropout regularization <hyperparameter>
        """
        super(DecoderLayer, self).__init__()

        # the first attention module -- we first apply self attention to the
        # decoded sequence & itself
        self.dec_outs_normalization = LayerNormalization(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.self_attention_dropout = nn.Dropout(dropout)

        # the second attention module -- we then apply attention to the outputs
        # of the decoded sequence & the encoder outputs
        self.enc_dec_normalization = LayerNormalization(d_model)
        self.enc_dec_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_dec_dropout = nn.Dropout(dropout)

        # we then feed forward our outputs
        self.feed_forward_normalization = LayerNormalization(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(self, decoded_outputs, encoder_outputs, padding_mask=None,
                look_ahead_mask=None, cache=None):
        """
        Decode the encoded inputs using the current decoded outputs to produce
        the next token.

        Inputs:
            decoded_outputs: during training, this is simply the expected
                outputs, where as during inference, this is
            the outputs which the model has decoded to a certain time step which
                is used to infer subsequent tokens
            encoder_outputs: the hidden representation outputted by the encoder
                from the inputs
            padding_mask: the mask which indicates which tokens are padding
                tokens to ignore -- used for encoder-decoder attention
            look_ahead_mask: the mask which indicates which tokens are future
                tokens to ignore, this should have been combined with the
                padding mask for the decoded_outputs tensor to also mask out
                padded tokens -- used for decoder self attention
            cache: for storing computed attention weights for later usage
                without re-computation

            The inputs should have the following shape:
            decoded_outputs:
                during training: (batch_size, [target] seq_len, d_model),
                during inference: (batch_size, curr_seq_len, d_model)
            encoder_outputs: (batch_size, [input] seq_len, d_model)
        """
        # normalize the decoded sequence
        normalized_dec_outs = self.dec_outs_normalization(decoded_outputs)

        # compute the self attention weights --
        # *([normalized_dec_outs] * 3) == normalized_dec_outs passed in 3 times
        # this is the attention using the decoded sequence and itself
        self_attention_weights = self.self_attention(
            *([normalized_dec_outs] * 3), look_ahead_mask)
        self_attention_weights = self.self_attention_dropout(
            self_attention_weights)
        self_attention_weighted_dec_outs = decoded_outputs \
                                           + self_attention_weights

        # normalize the attention weighted decoder outputs
        normalized_self_attention_dec_outs = self.enc_dec_normalization(
            self_attention_weighted_dec_outs)

        # compute the encoder-decoder attention weights
        # the key and value of this attention module are the encoder outputs and
        # the query are the decoded outputs we want to compute how much weight
        # each token from the encoder outputs should have when we are decoding
        # using the current decoded outputs
        enc_dec_attention_weights = self.enc_dec_attention(
            normalized_self_attention_dec_outs, encoder_outputs,
            encoder_outputs, padding_mask, cache)
        enc_dec_attention_weights = self.enc_dec_dropout(
            enc_dec_attention_weights)
        full_attention_weighted_dec_outs = self_attention_weighted_dec_outs \
                                           + enc_dec_attention_weights

        # we will then pass our attention weighted (both self attention and with
        # the encoder outputs) through a feed forward layer
        normalized_full_attention_dec_outs = self.feed_forward_normalization(
            full_attention_weighted_dec_outs)
        feed_forward_outputs = self.feed_forward(
            normalized_full_attention_dec_outs)
        feed_forward_outputs = self.feed_forward_dropout(feed_forward_outputs)

        # we then sum the full attention weighted decoded outputs with the feed
        # forward outputs to get our final decoder outputs

        return full_attention_weighted_dec_outs + feed_forward_outputs


class Decoder(nn.Module):
    """
    This module implements the Decoder in the Transformer, which is a stack of
    decoder layers that are sequentially applied starting from the current
    decoded tokens and the encoder outputs. The Decoder uses two layers of the
    multi-head attention mechanism.
    """
    def __init__(self, d_model, d_ff, num_heads, num_layers, dropout=0.1):
        """
        Inputs:
            d_model: the dimensionality of the outputs <hyperparameter>
            d_ff: the number of hidden units for our feed forward layer
                <hyperparameter>
            num_heads: the number of heads to split the attention down
                <hyperparameter>
            num_layers: the number of encoder layers to apply <hyperparameter>
            dropout: the rate of dropout regularization <hyperparameter>
        """
        super(Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)]
        )

        self.outputs_normalization = LayerNormalization(d_model)

    def forward(self, decoded_inputs, encoder_outputs, padding_mask=None,
                look_ahead_mask=None, cache=None):
        """
        Apply the decoder layers to the current decoded inputs and encoder
        outputs.

        decoded_outputs: during training, this is simply the expected outputs,
            where as during inference, this is
        the outputs which the model has decoded to a certain time step which is
            used to infer subsequent tokens
        encoder_outputs: the hidden representation outputted by the encoder from
            the inputs
        padding_mask: the mask which indicates which tokens are padding tokens
            to ignore -- used for encoder-decoder attention
        look_ahead_mask: the mask which indicates which tokens are future tokens
            to ignore, this should have been combined with the padding mask for
            the decoded_outputs tensor to also mask out padded tokens -- used for
            decoder self attention
        cache: for storing computed attention weights for later usage without
            re-computation

        The inputs should have the following shape:
        decoded_outputs:
            during training: (batch_size, [target] seq_len, d_model),
            during inference: (batch_size, curr_seq_len, d_model)
        encoder_outputs: (batch_size, [input] seq_len, d_model)
        """
        if cache is not None:
            assert isinstance(cache, dict), "To cache decoder attention " \
                                            "results, pass in an empty " \
                                            "dictionary"
            assert not len(cache), "Cache should be passed as an empty " \
                                   "dictionary"

        decoded_outputs = decoded_inputs

        # noinspection PyTypeChecker
        for layer_num, decode in enumerate(self.decoder_layers):
            # by default, the cache for this layer is None
            computed_cache = None
            # if we want to cache our results
            # -- in this case cache should be a dictionary
            if cache is not None:
                # if a layer is not initiated, initiate our cache for this
                # current layer
                if layer_num not in cache:
                    cache[layer_num] = {}

                computed_cache = cache[layer_num]

            # if cache is None, then computed_cache here will be None and so
            # the attention will just be called with the empty cache and so our
            # results will not be store

            # if cache is not None, then we have given an empty dictionary into
            # the attention module for computed attention tensors to be kept
            decoded_outputs = decode(
                decoded_outputs, encoder_outputs, padding_mask,
                look_ahead_mask, computed_cache)

        return decoded_outputs
