import torch.optim as optim
import warnings


class TransformerLRScheduler(optim.lr_scheduler._LRScheduler):
    """
    Vary the learning rate over the course of the training as proposed by the
    authors of the paper. The formula is as follows:
    lr(step_num) = d_model ** -0.5 * min(
    step_num ** -0.5, step_num * warmup_steps ** -1.5)

    Note that this module will step the optimizer automatically so there is no
    need to step the optimizer separately. Although there is a choice to not do
    this so your own stepping mechanism can be done (step only after a certain
    number of iterations for an example). By default, the scheduler will step
    the optimizer.
    """

    def __init__(self, warmup_steps, d_model, optimizer, last_step=-1,
                 step_optimizer=True):
        """
        Initializes an instance of the Transformer LR Scheduler.

        Inputs:
            optimizer: the optimizer instance that is used for training whose
                learning rates
            warmup_steps: the number of steps to warm up with to avoid early
                overfitting.
            d_model: the dimensionality of our outputs
            last_step: the step number of the previous step, PyTorch's base LR
                Scheduler names this last_epoch which is unfortunate naming
                since it isn't the epoch number that we are using here, but the
                training step number
        """
        super(TransformerLRScheduler, self).__init__(optimizer, last_step)

        # we're calling our last_step argumnet last_epoch here to be consistent
        # with PyTorch's implementation

        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.step_optimizer = step_optimizer

        if not self.step_optimizer:
            warnings.warn("It is specified that the LR Scheduler should not "
                          "step the optimizer automatically, ensure to step it "
                          "yourself.")

    def get_lr(self):
        """
        This method computes the next learning rate value for the next training
        step. This is called in the super class' step method which actually
        steps the learning rate.

        Note that we are returning a list of learning rate for all the
        parameters that need to be trained for the model instead of just one
        value.

        Output:
            A list of size of the length of self.optimizer.param_groups
            containing the next learning rate
        """
        # noinspection PyUnresolvedReferences
        step_num = self.last_epoch + 1
        normal_lr = step_num ** -0.5
        in_warmup = step_num * self.warmup_steps ** -1.5

        min_val = normal_lr if normal_lr <= in_warmup else in_warmup

        # noinspection PyUnresolvedReferences
        num_param_groups = len(self.optimizer.param_groups)
        return [self.d_model ** -0.5 * min_val] * num_param_groups

    def step(self, epoch=None):
        """
        Step the scheduler and also the optimizer if necessary.

        Input:
            epoch: please always step with the value being None (or don't pass
                in anything it will be defaulted to None) the parameter is given
                to be consistent with the interface given by PyTorch but isn't
                used at all; if you set this to any value, PyTorch will give you
                a warning and attempt to obtain a closed form learning rate
                which we do not have a method for
        """
        if self.step_optimizer:
            # noinspection PyUnresolvedReferences
            self.optimizer.step()

        super(TransformerLRScheduler, self).step(epoch)
