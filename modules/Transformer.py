import torch
import torch.nn as nn
import numpy as np
from typing import Any
from .Encoding import Encoder
from .Decoding import Decoder
from .PositionalEncoding import PositionalEncoding
from .utils import initialize_xavier_weights
from .utils import create_padding_mask, create_look_ahead_mask


class Transformer(nn.Module):
    """
    The Transformer architecture as proposed by the paper "Attention is All You
    Need". This encoder-decoder architecture consists of an encoder and decoder
    stack which implements the attention mechanism.

    1. An input sentence is first tokenized and then converted into word
    embeddings. We then apply positional encoding to the word embeddings
    obtained so that we can incorporate positional information to the word
    embeddings.

    2. The positionally encoded word embeddings are then passed into the
    encoder, which converts these embeddings into hidden representations after
    applying self-attention. The self-attention mechanism uses dot product to
    compute compatibility between words within a sentence and uses this to scale
    the magnitude of the word embedding of each word within that sentence.

    3. The outputs from the encoder are given to the decoder, which decodes this
    into the output which we hopefully desire. The decoder also gets as input,
    the sequence of tokens that are decoded up to the current time step. This
    is because the Transformer is a language model, and so it decodes each word
    based on the given input, which the encoder transforms into a hidden
    representation, and the decoded-so-far sequence. During training, we use
    teacher-forcing, which is a technique that uses the true (expected) token as
    input to the next time step instead of the predicted token. Therefore, we
    simply use the entire expected output as input into our decoder in training.

    4. From the results outputted by our decoder, we perform a linear
    transformation where the output dimensions corresponds to the size of our
    vocabulary. We then perform softmax on the tensor to convert these values
    into a probability distribution. This is the output of the Transformer.

    We can then use a search algorithm (one of the best ones being Beam Search)
    to convert these probability distributions into actual sentences. This
    process is known as detokenization.
    """
    def __init__(self, vocab_size, d_model, d_ff, num_heads, num_layers,
                 dropout=0.1, use_embed_weights=True, set_seq_len=None,
                 device=torch.device("cpu")):
        """
        vocab_size: the number of unique words in our vocabulary
        d_model: the dimensionality of the outputs <hyperparameter>
        d_ff: the number of hidden units for our feed forward layer
            <hyperparameter>
        num_heads: the number of heads to split the attention down
            <hyperparameter>
        num_layers: the number of encoder layers to apply <hyperparameter>
        dropout: the rate of dropout regularization <hyperparameter>
        use_embed_weights: whether or not to use the embeddings' weights as the
            final projection weights
        set_seq_len: the longest sequence we ever have to process for
            pre-computation of the positional information
        device: device which operations are performed on
        """
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # configurations for our embedding module

        # the paper suggests that we divide our embeddings by sqrt(d_model)
        self.embedding_scale = 1 / np.sqrt(d_model)

        # we use a learn-able embedding module and initialize the weights to
        # have mean 0 and standard deviation 1/sqrt(d_model)
        self.embedding_module = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(
            self.embedding_module.weight, mean=0, std=d_model ** -0.5)

        # regularization for embeddings
        self.embedding_dropout = nn.Dropout(dropout)

        # if use_embed_weights is true, our output projection weights are tied
        # to the embedding module's weights we define this as a lambda function
        # that can be called -- we can also define a different method and point
        # the variable to that method instead
        if use_embed_weights:
            self.project_to_vocabs = \
                lambda decoder_out: \
                    self.embedding_module.weight.transpose(0, 1) @ decoder_out

        # otherwise, our output projection is a linear layer which we use Xavier
        # initialization on
        else:
            self.project_to_vocabs = nn.Linear(d_model, vocab_size)
            initialize_xavier_weights(self.project_to_vocabs)

        # for more details about the following modules, please read their
        # documentations we initialize the positional encoding module
        self.positional_encoder = PositionalEncoding(
            d_model, set_seq_len, device)

        # the encoder and decoder
        self.encoder = Encoder(d_model, d_ff, num_heads, num_layers, dropout)
        self.decoder = Decoder(d_model, d_ff, num_heads, num_layers, dropout)

    def encode(self, input_sentences):
        """
        Encode the sentences by transforming them into hidden representations
        using the attention mechanism.

        Input:
            input_sentences: the batch of sentences which we will process
                through the encoder

            The input should have shape (batch_size, seq_len)

        Output:
            The encoded representations of the input sentences.
        """
        device = input_sentences.device

        # we need to create a mask which indicates the positions which a token
        # is padding token and should be ignored
        padding_mask = create_padding_mask(input_sentences, device=device)
        # we convert our input into word embeddings
        input_sentences = self.embedding_module(input_sentences)

        # here, we use the padding mask to put a 0 wherever there is a padding
        # token -- the word embedding value for these tokens should be 0 since
        # they do not hold any meaning

        # this does an inplace fill
        input_sentences.masked_fill_(padding_mask.unsqueeze(2), 0)

        # as suggested by the paper, we scale the magnitude of the word
        # embeddings by sqrt(d_model)
        input_sentences.mul_(self.embedding_scale)

        # next step is to add positional information using positional encoding
        input_sentences = self.positional_encoder(input_sentences)

        # regularize the embeddings (as it is learned) using dropout
        input_sentences = self.embedding_dropout(input_sentences)

        return self.encoder(input_sentences, padding_mask[:, None, None, :])

    def decode(self, decoded_outputs, encoder_outputs, cache=None):
        """
        Using the inputs that are encoded by the encoder and the outputs that
        are decoded to the current time-step, decode the next token.
        During training, this is done simultaneously for all time-steps as we
        utilize teacher-forcing.

        Inputs:
            decoded_outputs: during training, this is simply the expected
                outputs, where as during inference, this is the outputs which
                the model has decoded to a certain time step which is used to
                infer subsequent tokens
            encoder_outputs: the hidden representation outputted by the encoder
                from the inputs
            cache: for storing computed attention weights for later usage
                without re-computation

            The inputs should have the following shape:
            decoded_outputs:
                during training: (batch_size, [target] seq_len, d_model),
                during inference: (batch_size, curr_seq_len, d_model)
            encoder_outputs: (batch_size, [input] seq_len, d_model)

        Outputs:
            The next time-step decoded output(s).
        """
        device = decoded_outputs.device

        # similar to the encoder we create a padding mask to indicate the tokens
        # that should be ignored as padding; we also use a look ahead mask to
        # indicate the tokens that should be ignored when we are decoding the
        # next token using the prefixes of the sentences -- this is what happens
        # during inference so we need to set up this environment for training as
        # well
        padding_mask = create_padding_mask(decoded_outputs, device=device)
        look_ahead_mask = create_look_ahead_mask(decoded_outputs, padding_mask)

        # convert to word embeddings
        decoded_outputs = self.embedding_module(decoded_outputs)

        # we zero-out the positions that are padding tokens in our word
        # embeddings
        decoded_outputs.masked_fill_(padding_mask.unsqueeze(2), 0)

        # note that we do not use the look ahead mask to zero-out our word
        # embeddings because the look ahead mask tells the attention mechanism
        # to ignore future words in the decoded outputs while decoding, but
        # those words (tokens) still have meaning, in contrast to padding tokens
        # which have no meaning at all

        # we scale the embeddings' magnitude by sqrt(d_model)
        decoded_outputs.mul_(self.embedding_scale)

        # we encode positional information to the word embeddings next
        decoded_outputs = self.positional_encoder(decoded_outputs)

        # dropout regularization
        decoded_outputs = self.embedding_dropout(decoded_outputs)

        return self.decoder(decoded_outputs, encoder_outputs, padding_mask,
                            look_ahead_mask, cache)

    def forward(self, input_sentences, decoded_outputs):
        """
        Forwards the input sentences through the Transformer architecture by
        feeding it into the encoder to produce a hidden representation that is
        used to decode the corresponding output(s).

        Inputs:
            Input:
            input_sentences: the batch of sentences which we will process
                through the encoder
            decoded_outputs: during training, this is simply the expected
                outputs, where as during inference, this is the outputs which
                the model has decoded to a certain time step which is used to
                infer subsequent tokens

            The inputs should have shape:
            input_sentences: (batch_size, [input] seq_len)
            decoded_outputs:
                during training: (batch_size, [target] seq_len, d_model),
                during inference: (batch_size, curr_seq_len, d_model)

        Outputs:
            The outputs decoded from the given input sentences.
        """
        encoder_outputs = self.encode(input_sentences)
        decoder_outputs = self.decode(decoded_outputs, encoder_outputs)

        return decoder_outputs

    def __call__(self, *args, **kwargs) -> Any:
        """
        This method is to simply suppress PyCharm warnings.
        """
        return super().__call__(*args, **kwargs)
