import torch


def create_padding_mask(inputs_batch, pad_token=0, device=torch.device("cpu")):
    """
    Return a boolean (byte) matrix representing, at each token, whether or not that token is a padded token.

    Inputs:
        inputs_batch: the input sentences that have been tokenized
        pad_token: the index of the padding token in the vocabulary

    Output:
        A mask which indicates which tokens are padding tokens and which token are not.
    """
    # like numpy, torch has a convenient way of doing element-wise comparison to a scalar, and the result of the
    # comparison would be a matrix of the same shape where each element is a byte indicating whether that comparison
    # is true for the element in the original matrix
    return (inputs_batch == pad_token).to(device)


def create_look_ahead_mask(inputs_batch, padding_mask=None, pad_token=0, device=torch.device("cpu")):
    """
    Return a boolean (byte) mask indicating, for each token, the subsequent tokens that should be ignored, as we wish
    to create the real decoding environment where the model does not have access to future tokens while decoding.

    Inputs:
        inputs_batch: the input sentences that have been tokenized
        pad_token: the index of the padding token in the vocabulary

    Output:
        A mask which indicates future tokens that should be ignored during decoding
    """
    seq_len = inputs_batch.size(1)
    square_matrix = torch.ones((seq_len, seq_len), dtype=torch.uint8).to(device)

    look_ahead_mask = torch.triu(square_matrix, diagonal=1)

    # we need to combine the above look ahead mask with the corresponding padding mask for the input sentences:
    # this is because if a token is either a padding token or a future token, then it needs to be ignored in our
    # attention mechanism -- essentially, we take the maximum of the above look ahead mask and the padding mask
    # because a 1 in either mask indicates that the token is either padding or future, and therefore we must ignore it

    # we want to do a max operation between every row in a with the first row in b and group this together to indicate
    # the padding & look ahead mask of the first training sentence, and we group every row in a with the second row in b
    # and so on -- therefore, we want to expand the padding mask to have another dimension in the middle

    # we also know that during our attention mechanism, we split our tensors into a 4-dimensional tensor, and therefore,
    # our mask also needs to have 4 dimensions, so we add a total of 2 dimensions
    if padding_mask is None:
        padding_mask = create_padding_mask(inputs_batch, pad_token)[:, None, None, :]

    return torch.max(look_ahead_mask, padding_mask)
