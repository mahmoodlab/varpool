import numpy as np
import os
import torch
from copy import deepcopy


class EarlyStopper:
    """
    Checks early stopping criteria and saves a model checkpoint anytime a record is set. Note the model checkpoints are saved everytime a record is set i.e. at the beginning of the patience period.

    Parameters
    ----------
    save_dir: str
        Directory to where the model checkpoints are saved.

    min_epoch: int
        Dont check early stopping before this epoch. Record model checkpoints will still be saved before min_epoch.

    patience: int
        How many calls to wait after last time validation loss improved to decide whether or not to stop. Note this corresponds to the number of calls to EarlyStopper e.g. if we only check the validation score every K epochs then patience_epochs = K * patience.

    patience_min_improve: float
        The minimum improvement over the previous best score for us to reset the patience coutner. E.g. if the metric is very slowly improving we might want to stop training.

    abs_scale: bool
        Whether or not the patience_min_improve should be put on absolue (new - prev) or relative scale (new - prev)/prev.

    min_good: float
        Want to minimize the score metric (e.g. validation loss).

    verbose: bool
        Whether or not to print progress.

    Attributes
    ----------
    patience_counter_: int
        The current patience counter.

    best_score_: float
        The best observed score so far.

    epochs_with_records_: list of in
        The epochs where a record was set.
    """
    def __init__(self, save_dir, min_epoch=20, patience=10, min_good=True,
                 patience_min_improve=0, abs_scale=True, verbose=True):

        self.save_dir = save_dir

        self.min_epoch = min_epoch
        self.patience = patience

        self.patience_min_improve = patience_min_improve
        self.abs_scale = abs_scale

        self.verbose = verbose
        self.min_good = min_good

        self._reset_tracking()

    def __call__(self, model, score, epoch, ckpt_name='checkpoint.pt'):
        """
        Check early stopping criterion.

        Parametres
        ----------
        model:
            The model to maybe save.

        score: float
            The metric we are scoring e.g. validation loss.

        epoch: int
            Which epoch just finished. Assumes zero indexed i.e. epoch=0 means we just finished the first epoch.

        ckpt_name: str
            The name of the checkpoint file.

        Output
        -----
        stop_early: bool
        """

        #################################################
        # Check if  new record + save record checkpoint #
        #################################################
        prev_best = deepcopy(self.best_score_)

        # check if this score is a record
        if (self.min_good and score < self.best_score_) or \
                ((not self.min_good) and score > self.best_score_):

            self.best_score_ = score
            is_record = True

            # always save a record to disk
            os.makedirs(self.save_dir, exist_ok=True)
            fpath = os.path.join(self.save_dir, ckpt_name)
            torch.save(model.state_dict(), fpath)

            if self.verbose:
                print('New record set on epoch {} at {:1.5f}'
                      ' (previously was {:1.5f})'.
                      format(epoch, self.best_score_, prev_best))

            self.epochs_with_records_.append(epoch)

        else:
            is_record = False

        ########################
        # Check early stopping #
        ########################
        stop_early = False

        if (epoch + 1) >= self.min_epoch:
            # either increase or reset the counter since we are beyond the min epoch

            if is_record:  # +1 for zero indexing
                # if we are passed the warm up period and just set a record

                # check if this is impressive record
                if abs(prev_best) == np.inf:
                    # if the previous record was infintee this
                    # was auotmatically an imporessive record
                    is_impressive_record = True

                else:
                    # compute difference on absolute or relative scale
                    abs_diff = abs(score - prev_best)
                    if self.abs_scale:
                        diff_to_check = abs_diff
                    else:
                        epsilon = np.finfo(float).eps
                        diff_to_check = abs_diff / (abs(prev_best) + epsilon)

                    if diff_to_check >= self.patience_min_improve:
                        is_impressive_record = True
                    else:
                        is_impressive_record = False
            else:
                # this was not a record
                is_impressive_record = False

            # reset the patience counter for impressive records
            # otherwise increase counter
            if is_impressive_record:
                self.patience_counter_ = 0
            else:
                self.patience_counter_ += 1

                if self.verbose:
                    print("Early stopping counter {}/{}".
                          format(self.patience_counter_, self.patience))

            # if we have met our patience level we should stop!
            if self.patience_counter_ >= self.patience:
                stop_early = True

        return stop_early

    def _reset_tracking(self):
        """
        resets the tracked data
        """
        self.patience_counter_ = 0

        if self.min_good:
            self.best_score_ = np.Inf
        else:
            self.best_score_ = -np.Inf

        self.epochs_with_records_ = []
