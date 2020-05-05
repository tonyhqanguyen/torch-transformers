import torch
import torch.nn as nn
from .utils import initialize_xavier_weights


class FeedForward(nn.Module):
    """
    A fully connected feed forward layer that is attached at the end of every
    attention module. This feed-forward network is applied to each position
    separately and identically.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Inputs:
            d_model: the dimensionality of the outputs of the architecture
                <hyperparameter>
            d_ff: the dimensionality of the hidden state of the feed forward
                network <hyperparameter>
            dropout: the rate of dropout for regularization
        """
        super(FeedForward, self).__init__()

        # the feed forward network consists of 2 transformation, essentially,
        # the formula which we wish to implement is FF(x) = ReLU(xW1 + b1)W2
        # + b2 -- so we first apply a linear transformation, apply the ReLU
        # activation function, and then apply a second linear transformation
        first_transformation = nn.Linear(d_model, d_ff)
        second_transformation = nn.Linear(d_ff, d_model)
        initialize_xavier_weights(first_transformation)
        initialize_xavier_weights(second_transformation)

        # dropout for regularization
        dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            first_transformation,
            nn.ReLU(inplace=True),
            second_transformation,
            dropout
        )

    def forward(self, attention_word_embeddings):
        """
        Apply the feed forward network onto the given inputs.

        Input:
            attention_word_embeddings: the resulting matrix from performing the
                multi-head mechanism in both the encoder
            and decoder.

            The shape of this matrix should be (batch_size, seq_len, d_model)

        Outputs:
            The computed values passed through the feed forward layer.
        """
        # we have declared the sequential module that is the core operations we
        # require to perform the feed forward action, and so we simply need to
        # pply the sequential network onto our inputs
        return self.feed_forward(attention_word_embeddings)


class LayerNormalization(nn.Module):
    """
    The module which implements layer normalization. The intuition behind layer
    normalization extends that of batch normalization. The distribution of
    training data points can and will vary with every batch, and therefore,
    causes the neural network to continually adapt to these differences to learn
    the data well. With more and more layers in the neural network, higher order
    interactions can increase the loss as training progresses unless a very
    small learning rate is used. Thus, we wish to normalize the mean and
    variance of each batch to 0 and 1 respectively so that the neural network
    has less complications to deal with, and a higher learning rate can be used
    to speed up training.

    The limitation of batch normalization is that this means that we cannot use
    a small batch size because then the normalization would not be effective.
    This means that there is a lower limit to the batch size, and thus increases
    memory consumption during training. We also know that small batch size is
    not a good setting for online learning, or algorithms that are very
    sensitive to noises. Instead of normalizing all activations within a batch
    to follow the same mean and variance, we normalize it along all feature
    dimensions.

    We essentially need to learn 2 parameters, gamma and beta, as essentially,
    we are performing a covariate shift.
    """

    # noinspection PyArgumentList
    def __init__(self, d_model, epsilon=1e-6):
        """
        Inputs:
            d_model: the dimensionality of the outputs <hyperparameter>
            epsilon: a float which is used for numerical stability in
                the case that variance is 0
        """
        super(LayerNormalization, self).__init__()

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, activations):
        """
        Inputs:
            activations: the activations outputted by each module of the neural
                network that needs to be normalized.

            The shape of the activations should have shape
            (batch_size, seq_len, d_model)

        Outputs:
            The shifted normalized activations for the layer requiring
            normalization.
        """
        # we want to get the mean and variance of each feature within this
        # batch, since the input has shape batch_size, seq_len, d_model,
        # the last dimension is the dimension of the features for each word;
        # we also set keepdims to True so we can perform subtraction between
        # the original activations tensor to the mean tensor
        feature_mean = activations.mean(dim=-1, keepdims=True)

        # we compute the variance
        feature_variance = ((activations - feature_mean) ** 2).\
            mean(dim=-1, keepdims=True)

        # we now center the activation
        centered_activations = activations - feature_mean

        # each normalized activation is subtracted by its corresponding mean
        # (mean of the feature dimension it is from) and divided by standard
        # deviation -- here the standard deviation is computed as variance
        # + epsilon square rooted because we want to avoid having the standard
        # deviation being zero as we cannot divide by 0

        feature_st_dev = (feature_variance + self.epsilon).sqrt()
        normalized_activations = centered_activations / feature_st_dev

        return self.gamma * normalized_activations + self.beta
