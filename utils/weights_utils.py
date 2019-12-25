import torch.nn as nn


def initialize_xavier_weights(layer):
    """
    Implement the xavier weights initialization for the layer.

    Input:
        layer: the affine transformation layer that requires weight initializations
    """
    nn.init.xavier_uniform_(layer.weight)

    # if the layer has a bias (the +b) part, initialize them to be all 0's
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)
