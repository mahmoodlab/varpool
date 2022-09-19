import numpy as np


class GradAccum:
    """
    Calculates loss function divosor for adjusting the loss when training with gradient accumulation.
    This object handles the case when number of batches is not divisible by the gradient accumultation and/or the case when the number of samples is not divisible by the batch size.
    
    Note this object assumes the loss function already averages over batches and that all but possibly the last batch has the same batch size. 

    Be careful about this when using nn.DataParallel 
    TODO: @iain think this through!
    
    This is also a small issue when using loss functions that have class weights, see https://github.com/pytorch/pytorch/issues/72047

    Parameters
    ----------
    loader: torch DataLoader
        The data loader for the batches.
        
    grad_accum: int, None
        Number of gradient accumulation steps. 
    
    
    Example
    -------
    GA = GradAccum(loader=loader, grad_accum=grad_accum)
    
    model.zero_grad()
    for batch_idx, (x, y_true) in enumerate(loader):
    
        
        y_pred = model(x)
        loss = loss_func(y_true, y_pred)  # assume this averages over batch
    
        # adjust loss divisor
        loss_div, update_params = GA.get_loss_div(batch_idx)
        loss = loss / loss_div

        loss.backward()

        # step after gradeint accumulation
        if update_params:
            optimizer.step()
            optimizer.zero_grad()
    """

    def __init__(self, loader, grad_accum):

        # no grad accum if <= 1
        grad_accum = 1 if (grad_accum is not None and grad_accum <= 1)\
            else grad_accum

        #########
        # Setup #
        #########

        n_batches = len(loader)
        n_samples = len(loader.dataset)
        batch_size = loader.batch_size

        # adjust for dropping last batch
        if loader.drop_last and (n_samples % batch_size == 0):
            # we see exactly full batches
            n_samples = n_batches * batch_size

        # number of gradient accumulation batches
        # i.e. the effective number of batches
        n_grad_accum_batches = int(np.ceil(n_batches / grad_accum))

        # number of batches the last grad accum batch will have
        if n_batches % grad_accum == 0:
            n_batchs_in_last_ga_batch = grad_accum
        else:
            n_batchs_in_last_ga_batch = n_batches % grad_accum

        # number of samples in last batch
        if n_samples % batch_size == 0:
            n_samples_last_batch = batch_size
        else:
            n_samples_last_batch = n_samples % batch_size

        # the last grad accum batch may see
        # a different number of grad accum batches
        # and/or an uneven batch
        n_samples_in_last_grad_accum_batch = \
            (n_batchs_in_last_ga_batch - 1) * batch_size +\
            n_samples_last_batch

        # for the final gradient update batch we dont necessarily check
        # batch_idx % grad_accum == 0
        final_ga_batch_update_crit = n_batches % grad_accum

        # store data we need
        self.grad_accum = grad_accum
        self.batch_size = batch_size
        self.n_grad_accum_batches = n_grad_accum_batches
        self.n_samples_in_last_grad_accum_batch = \
            n_samples_in_last_grad_accum_batch

        self.final_ga_batch_update_crit = final_ga_batch_update_crit

    def get_loss_div(self, batch_idx):
        """
        Gets the quantity we should divide loss function by to account for gradient accumulation

        Parameters
        ----------
        batch_idx: int
            The current batch index.

        Output
        ------
        loss_div, update_params

        loss_div: float
            The loss divisor that accounts for gradient accumulation.

        update_params: bool
            Whether or not to update parameters with a gradient step on this batch_idx.
        """

        if self.grad_accum is None:
            return 1.

        # which gradient accumulation batch we are on
        grad_accum_batch_idx = batch_idx // self.grad_accum

        if (grad_accum_batch_idx + 1) < self.n_grad_accum_batches:
            # NOT on last grad accum batch
            # current div is batch_size and we want to
            # change it to batch_size * grad_accum
            loss_div = self.grad_accum

            # update parameters every grad_accum times
            update_params = (batch_idx + 1) % self.grad_accum == 0

        else:
            # ON last grad accum batch
            # current div is batch_size and we want to
            # change it to n_samples_in_last_grad_accum_batch
            loss_div = self.batch_size / self.n_samples_in_last_grad_accum_batch

            # update parameters on very last batch index
            update_params = (batch_idx + 1) % self.grad_accum == \
                self.final_ga_batch_update_crit

        return loss_div, update_params
