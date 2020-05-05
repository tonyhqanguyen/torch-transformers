import torch
import torch.nn as nn
import numpy as np
from .utils import initialize_xavier_weights


def scaled_dot_attention(query, key, value, mask=None, dropout=None):
    """
    Inputs:
        <query>, <key> and <value> are multi-dimensional arrays where each entry
        are the word embeddings that have positional information encoded and are
        also linearly projected into <num_heads>. <num_heads> is a
        hyperparameter of the architecture.

        <mask> is a multi-dimensional array indicating which entries of the
        input should be ignored, this is usually used to ignore padding tokens
        (tokens that are used to equalize the lengths of inputs within a batch)
        and tokens that are further ahead in the sentence in the decoding step.

        The inputs should have the following shapes:
            Query: (batch_size, num_heads, seq_len, d_head)
            Key:   (batch_size, num_heads, d_head, seq_len)
                or (batch_size, num_heads, seq_len, d_head)
            Value: (batch_size, num_heads, seq_len, d_head)
            Mask: (batch_size, None, None, seq_len)

    Outputs:
        The scaled word embeddings of all sentences in the value matrix based on
        the attention score calculated using dot product attention.


    The formula we implement is Softmax(QK^T/d_k)V
    """
    # first, we take the dot product of the query and the key to find the
    # "compatibility" between each word in the query with each word in the key

    # obviously, we will need to transpose the key matrix (K in the formula)
    # -- pretty standard as we need to match dimensions, however, these matrices
    # have 4 dimensions, for them to multiply correctly, we need their last 2
    # dimensions to match -- note that we only need to transpose the key matrix
    # if it has shape (..., d_head, seq_len)
    try:
        query_key = query @ key

    except RuntimeError:
        try:
            # switch the last 2 dimensions (seq_len, d_head) ->
            # (d_head, seq_len) before taking the dot product
            query_key = query @ key.transpose(2, 3)
        except RuntimeError as error:
            raise RuntimeError(f"""Inputs to the function scaled_dot_attention 
                                   is incorrect: dimensions of either the query 
                                   or key matrix are invalid. 
                                   Error log: {error}""")

    # the paper suggests that the above matrix is large in magnitude as d_head
    # grows, and so passing it through the softmax function as required by our
    # formula will result in values with extremely small gradient, we can see
    # this -- if we assume that for a query embedding and a key embedding, we
    # will have d_head number of entries for each embedding as independent
    # random variables with mean 0 and variance 1 -- that is the dot product of
    # q and k can be written as q · k = q1k1 + q2k2 + ... + q_{d_head}k_{d_head}

    # we know that E[XY] = E[X]E[Y], Var[X + Y] = Var[X] + Var[Y],
    # Var[XY] = Var[X]Var[Y] when X and Y are independent, we also know that
    # E[X + Y] = E[X] + E[Y] by linearity of the expectation function,
    # therefore, q · k has mean 0 and variance d_head

    # therefore, we scale the values of the matrix by sqrt(d_head)
    d_head = query.size(-1)    # last dimension of the query matrix is d_head

    # we use numpy here since torch is weird with integers
    reduced_query_key = query_key / np.sqrt(d_head)

    # we now mask the matrix above, because in our preprocessing step, we pad
    # our inputs so that every training example in this batch will have the same
    # sequence length (seq_len) so that we can compute our matrix operations --
    # these padded entries in the original embedding should not affect the
    # meaning of the sentence, and so we want to "zero" them out -- when we
    # input the matrix into our softmax function, we want these padded entries
    # to have 0 attention
    if mask is not None:
        # fill the padded tokens with the value -1 x 10^9 -- very negative
        try:
            logits = reduced_query_key.masked_fill_(mask, -1e9)

        except RuntimeError as error:
            raise RuntimeError(f"""Inputs to the function scaled_dot_attention 
                                   is incorrect: dimensions of the mask argument 
                                   is inconsistent with the query and key 
                                   computation. Error log: {error}""")

    # compute the attention score by performing softmax on the values of the
    # entries
    attention_scores = torch.softmax(logits, dim=3)

    # regularize by dropping out
    if dropout is not None:
        attention_scores = dropout(attention_scores)

    # value is the original matrix which we wish to scale using the dot product
    # attention mechanism, we then multiply our attention weight to each of the
    # corresponding entries in the value matrix
    return attention_scores, attention_scores @ value


class MultiHeadAttention(nn.Module):
    """
    The implementation of MultiHeadAttention as a neural network layer.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Inputs:
            d_model: the dimensionality of our outputs <hyperparameter>
            num_heads: the number of heads to split our attention mechanism into
                <hyperparameter>
            dropout: the dropout rate for regularization <hyperparameter>
        """
        assert not (d_model % num_heads), f"Parameter num_heads is not a " \
            f"divisor of parameter d_model, which is invalid since we need " \
            f"to split d_model num_heads time evenly. d_model: {d_model}, " \
            f"num_heads: {num_heads}"
        super(MultiHeadAttention, self).__init__()

        self.d_head = num_heads // d_model

        # we need to learn the projections for the query, key and value
        # representations into num_heads subspaces this is part of the attention
        # mechanism as described by the paper, where we linearly project
        self.query_projection = nn.Linear(
            in_features=d_model,
            out_features=num_heads * self.d_head,
            bias=False
        )

        self.key_projection = nn.Linear(
            in_features=d_model,
            out_features=num_heads * self.d_head,
            bias=False)

        self.value_projection = nn.Linear(
            in_features=d_model,
            out_features=num_heads * self.d_head,
            bias=False
        )

        # initialize the weights using the Xavier initialization technique
        initialize_xavier_weights(self.query_projection)
        initialize_xavier_weights(self.key_projection)
        initialize_xavier_weights(self.value_projection)

        self.output_projection = nn.Linear(
            in_features=num_heads * self.d_head,
            out_features=d_model,
            bias=False
        )
        # initialize weights for the output projection as well
        initialize_xavier_weights(self.output_projection)

        # dropout to regularize
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, matrix, batch_size):
        """
        Split the input matrix (should be query/key/value) into num_heads heads

        Inputs:
            matrix: the query, key or value matrix that we want to split
            batch_size: number of examples in the current batch

        Outputs:
            The same matrix, however, the last dimension is split into 2 more
            dimensions accordingly
        """
        split_matrix = matrix.view(batch_size, -1, self.num_heads, self.d_head)

        return split_matrix.transpose(1, 2)

    def forward(self, query, key, value, mask=None, cache=None,
                return_weights=False):
        """
        This method performs the attention mechanism:
            1. linearly project them using our learned projection matrices
            2. split the query, key and value into the correct number of heads
            3. compute the weighted sum of the value matrix using the scaled dot
               attention function
            4. concatenate and linearly project the weighted values

        Inputs:
            query: the matrix which is used as "reference" whose compatibility
                with the key matrix is measured
            key: the matrix whose compatibility with the query matrix is
                computed to obtain attention weights for value
            value: the matrix which we want to compute a weighted sum for using
                the attention computed from query & key

            These inputs should have the following shape:
                query: (batch_size, seq_len_q)
                key: (batch_size, seq_len_k)
                value: (batch_size, seq_len_v)

            Note that the length might be different for query and key-value
            matrices, since when we compute the encoder-decoder attention, we
            will use the embeddings for the input sentence, which might have a
            different length than the expected output sentence

        Outputs:
            The projection of the concatenation of the attention-weighted values
            projected into num_heads subspaces
        """
        # as required, the batch_size should be any of the matrix's first
        # dimension
        batch_size = query.size(0)

        # linearly project the query matrix
        query = self.query_projection(query)

        # split its head:
        query = self.split_heads(query, batch_size)

        # when we are computing the encoder-decoder attention, our key and value
        # will be from the encoder, which should already be computed and cached,
        # so we do not need to compute it again
        if cache is not None and "key" in cache:
            key, value = cache["key"], cache["value"]

        else:
            key, value = self.key_projection(key), self.value_projection(value)
            key= self.split_heads(key, batch_size).transpose(2, 3)
            value = self.split_heads(value, batch_size)
            if cache is not None:
                cache["key"], cache["value"] = key, value

        # we need to cast the mask as a Byte tensor so we can do a mask fill
        # operation in the scaled dot attention function as operations with mask
        # require that the mask can be interpreted as True/False values
        if mask is not None:
            mask = mask.type(torch.ByteTensor).to(query.device)

        # the scaled attention computed from our attention function
        weighted_values, attention_weights = \
            scaled_dot_attention(query, key, value, mask, self.dropout)

        # concatenation:
        # when we split the heads, we transposed the first and second dimensions,
        # so now we switch them back we need to call contiguous() because after
        # transposing a matrix, the values are not stored contiguously in memory
        # anymore; we finally combine the 2 last dimensions to obtain the
        # concatenated value matrix
        weighted_values = weighted_values.\
            tranpose(1, 2).\
            contiguous().\
            view(batch_size, -1, self.num_heads * self.d_head)

        # output projection:
        outputs = self.output_projection(weighted_values)

        if return_weights:
            return outputs, attention_weights

        return outputs
