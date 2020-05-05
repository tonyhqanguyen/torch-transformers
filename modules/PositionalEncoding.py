import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    The implementation of positional encoding which injects information of the
    relative and absolute positioning of words within sentences.
    """
    def __init__(self, d_model, set_seq_len=None, device=torch.device("cpu")):
        """
        Input:
            d_model: the dimensionality of the outputs <hyperparameter>
            set_seq_len: the longest sequence we ever have to process for
                pre-computation of the positional information
            device: device which operations are performed on
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # here we provide the option of pre-computing our positional encoding
        # tensor the positional encoding information is independent from the
        # semantic meaning of any sentence, so in other words, it is independent
        # from the word embeddings tensor, that is, we can precompute the
        # necessary positions and simply use this stored information as needed,
        # however, we need to compute positional encoding for every word
        # position in every training example, we need to ensure that
        # set_seq_len >= length of the longest training example within our
        # training dataset -- to optimize for space, we need set_seq_len to be
        # exactly that length and when we use our precomputed positional
        # encoding, we can just index the stored tensor up to the padded
        # seq_len of the batch
        self.precomputed_pos_encoding = None
        if set_seq_len is not None:
            # we do not want to register this as learn-able parameter of the
            # architecture, thus we use register_buffer to register this as an
            # attribute of the class but not something the optimizer needs to
            # change
            self.register_buffer(
                "precomputed_pos_encoding",
                self.get_positional_encoding(set_seq_len, device=device)
            )

    def compute_angles(self, positions, dimensions):
        """
        Compute the inner formula which the sin and cos functions will be
        called on.
        Input:
            positions: the word embeddings matrix that does not have positional
                encoding information

        Output:
            The computed formula for each position: pos/(10000^{2i/d_model})
        """
        return positions / (10000 ** (2 * dimensions / self.d_model))

    def get_positional_encoding(self, seq_len, device=torch.device("cpu")):
        """
        Compute the full formula which implements the positional information
        which will be added into our original word embeddings.

        Input:
            seq_len: the length of the longest example within our batch

        Output:
            The computed positional encoding for each position and dimensions
        """
        # we need to generate the positions tensor, which goes from 0 to
        # seq_len - 1
        positions = torch.arange(seq_len, dtype=torch.float)[:, None]

        # we need to generate the dimensions tensor, which goes from 0 to
        # d_model - 1
        dimensions = torch.arange(self.d_model, dtype=torch.float)[None, :]

        # note: we make positions of shape (seq_len, 1) and dimensions of
        # shape (1, d_model), so when we multiply the 2, we get a matrix that
        # computes the formula for every pair of (pos, i) where pos is a
        # position in the sentence and i is a dimension

        # get the inner part of the computation (the part which we call sin
        # and cos on)
        angles = self.compute_angles(positions, dimensions)

        # we apply sin on the even indices and cos on the odd indices
        # using torch's indexing syntax, to get the even indices, we start at
        # index 0 and take every second element, to get the odd indices, we
        # start at index 1 and take every second element

        # note that we want to index into the "i" part, meaning that the
        # decision of whether to apply a sin or a cos to the formula computed
        # from angles depends on the dimension which a value lies in, not the
        # position so we index into the second dimension of the tensor
        sines = torch.sin(angles[:, 0::2])
        cosines = torch.cos(angles[:, 1::2])

        # since we split up the tensor using the second dimension, here, we
        # concatenate them back along the second dimension

        # we want to add another dimension so when we add this positional
        # information to our word embedding, it is added to all training
        # examples
        return torch.cat((sines, cosines), dim=1)[None, :].to(device)

    def forward(self, batch_word_embeddings):
        """
        Add the current word embeddings to the computed positional encoding
        formula to incorporate information regarding the words' relative and
        absolute positions within the sequences.

        Input:
            batch_word_embedding: A matrix of word embeddings for each training
            example in the batch

            The shape of the input should be (batch_size, seq_len, d_model)

        Output:
            The word embeddings added with the computed positional encoding.
        """
        seq_len = batch_word_embeddings.size(1)

        # if we already precomputed the positional encoding as described above,
        # then just add it to the word embeddings
        try:
            return batch_word_embeddings + \
                   self.precomputed_pos_encoding[:, :seq_len, :]

        # if not, we need to compute the positional encoding
        except TypeError:
            device = batch_word_embeddings.device
            positional_encoding = self.get_positional_encoding(
                seq_len, device=device)
            return batch_word_embeddings + positional_encoding
