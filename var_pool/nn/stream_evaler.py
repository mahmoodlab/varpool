import numpy as np
from scipy.special import softmax, expit
import torch
from warnings import warn

from sklearn.metrics import roc_auc_score,\
    accuracy_score, balanced_accuracy_score, f1_score  # classification_report
from sksurv.metrics import concordance_index_censored

from var_pool.utils import get_counts_and_props, get_traceback
from var_pool.nn.CoxLoss import CoxLoss
from var_pool.nn.SurvRankingLoss import SurvRankingLoss


class BaseStreamEvaler:
    """
    Base class for evaulating supervised learning metrics when the predictions are computed in batches.
    """

    def reset_tracking(self):
        """Resets the tracked data"""
        self.tracking_ = {'z': [], 'y_true': []}

    def log(self, z, y_true):
        assert z.ndim == 2
        assert z.shape[0] == y_true.shape[0]

        # move to numpy
        if torch.is_tensor(z):
            z = z.detach().cpu().numpy()
            y_true = y_true.detach().cpu().numpy()

        # get estimated probability/class label
        y_pred, prob_pred = pred_clf(z)

        self.tracking_['z'].append(z)
        self.tracking_['y_true'].append(y_true)

    def get_tracked_data(self):
        """
        Gets the tracked predictions and true response data i.e. concatenates all the backed data.

        Output
        ------
        z: array-like, (n_samples_tracked, n_out)
            The predictions for each sample tracked thus far.

        y_true: array-like, (n_samples_tracked, n_response)
            The responses for each sample tracked thus far.
        """
        z = np.concatenate(self.tracking_['z'])
        y_true = np.concatenate(self.tracking_['y_true'])

        z = safe_to_vec(z)
        y_true = safe_to_vec(y_true)

        return z, y_true

    def save_tracked_data(self, fpath, sample_ids=None):
        """
        Saves the tracked z and y_true data to disk.

        Parameters
        ----------
        fpath: str
            File path to save.
        """
        z, y_true = self.get_tracked_data()
        to_save = {'z': z, 'y_true': y_true}

        if sample_ids is not None:
            to_save['sample_ids'] = sample_ids

        np.savez(file=fpath, **to_save)

    def get_metrics(self):
        """
        Gets a variety of metrics after all the samples have been logged.

        Output
        ------
        metrics: dict of floats
        """
        raise NotImplementedError("Subclass should overwrite.")


def safe_to_vec(a):
    """
    Ensures a numpy array that should be a vector is always a vector
    """
    a = np.array(a)
    if a.ndim == 2 and a.shape[1] == 1:
        return a.reshape(-1)
    else:
        return a


class ClfEvaler(BaseStreamEvaler):
    """
    Evaluates classification metrics when the predictions are computed in batches.

    Parameters
    ----------
    class_names: None, list of str
        (Optional) The names of each class.
    """
    def __init__(self, class_names=None):
        self.class_names = class_names
        self.reset_tracking()

    def get_metrics(self):
        z, y_true = self.get_tracked_data()

        # comptue predictions
        y_pred, prob_pred = pred_clf(z)

        clf_report = {}
        clf_report['acc'] = accuracy_score(y_true=y_true,
                                           y_pred=y_pred)
        clf_report['bal_acc'] = balanced_accuracy_score(y_true=y_true,
                                                        y_pred=y_pred)

        clf_report['f1'] = f1_score(y_true=y_true,
                                    y_pred=y_pred,
                                    average='macro')

        # clf_report = classification_report(y_true=y_true,
        #                                    y_pred=y_pred,
        #                                    target_names=self.class_names,
        #                                    output_dict=True)

        # add auc score
        clf_report['auc'] = roc_auc_score(y_true=y_true, y_score=prob_pred,
                                          average='macro',
                                          multi_class='ovr')

        # prediction counts for each class
        if self.class_names is not None:
            n_classes = len(self.class_names)
        else:
            # try to guess
            n_classes = max(max(y_pred), max(y_true)) + 1

        counts, props = get_counts_and_props(y=y_pred,
                                             n_classes=n_classes,
                                             class_names=self.class_names)

        # add to clf_report
        for cl_name, prop in props.items():
            key = 'pred_prop__{}'.format(cl_name)
            clf_report[key] = prop

        return clf_report


class SurvMetricsMixin:
    """
    Mixin for computing survival metrics.

    Parameters
    ----------
    tied_tol: float
        The tolerance value for considering ties.  See sksurv.metrics. concordance_index_censored.

    train_times: None, array-like of floats, (n_samples_train, )
        (Optional) The training data survival times. Used for cumulative_dynamic_auc. If not provided this will not be computed.

    train_events: None, array-like of bools, (n_samples_train, )
        The training data event indicators. Used for cumulative_dynamic_auc. If not provided this will not be computed.

    Attributes
    ----------
    eval_times_
    """

    def _set_eval_times(self):
        """
        Sets the evaulation times for cumulative_dynamic_auc from the train times. Not self.train_times must be already set.
        """
        # this is the default from
        # https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html#Time-dependent-Area-under-the-ROC
        # TODO: maybe make this customizable

        if self.train_times is not None:
            quantiles = np.linspace(start=5, stop=81, num=15)
            self.eval_times_ = np.percentile(a=self.train_times, q=quantiles)

    def compute_surv_metrics(self, pred_risk_score, events, times):
        """
        Gets the c index. Safely handles NaNs.

        Parameters
        ----------
        pred_risk_score: array-like, (n_samples, )
            The predictted risk score.

        events: array-like of bools, (n_samples, )
            The event indicators

        times: array-like,e (n_samples, )
            The observed survival times

        Output
        ------
        out: dict
            Various survival prediction metrics.
        """
        out = {}

        try:
            out['c_index'] = \
                concordance_index_censored(event_indicator=events,
                                           event_time=times,
                                           estimate=pred_risk_score,
                                           tied_tol=self.tied_tol)[0]

        except Exception as e:
            warn("Could not compute c-index, got traceback:\n\n{}".
                 format(get_traceback(e)))
            out['c_index'] = np.nan

        # TODO: figure out where the nans are coming from
        # if self.train_times is not None:

        #     survival_train = Surv.from_arrays(event=self.train_events,
        #                                       time=self.train_times)

        #     survival_test = Surv.from_arrays(event=events,
        #                                      time=times)

        #     out['cum_dynamic_auc'] =\
        #         cumulative_dynamic_auc(survival_train=survival_train,
        #                                survival_test=survival_test,
        #                                estimate=pred_risk_score,
        #                                times=self.eval_times_,
        #                                tied_tol=self.tied_tol)[1]

        return out


class DiscreteSurvivalEvaler(BaseStreamEvaler, SurvMetricsMixin):
    """
    Evaulation object handling batch predictions for discrete survival models. Computes the concordance index when the predictions are computed in batches.

    See https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html.

    Parameters
    ----------
    tied_tol: float
        The tolerance value for considering ties.  See sksurv.metrics. concordance_index_censored.

    References
    ----------
    Pölsterl, S., 2020. scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn. J. Mach. Learn. Res., 21(212), pp.1-6.

    Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L., Rosati, R.A, "Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors", Statistics in Medicine, 15(4), 361-87, 1996.
    """
    def __init__(self, tied_tol=1e-8, train_times=None, train_events=None):
        self.tied_tol = tied_tol
        self.reset_tracking()

        self.train_times = train_times
        self.train_events = train_events
        self._set_eval_times()

    def get_metrics(self):
        z, y_true = self.get_tracked_data()

        # time_bin_true = y_true[:, 0]
        censor = y_true[:, 1].astype(bool)
        event_indicator = ~censor
        survival_time_true = y_true[:, 2]

        # compute prpedicted risk
        pred_risk = pred_discr_surv(z)

        out = self.compute_surv_metrics(pred_risk_score=pred_risk,
                                        events=event_indicator,
                                        times=survival_time_true)
        return out


class CoxSurvivalEvaler(BaseStreamEvaler, SurvMetricsMixin):
    """
    Evaulation object handling batch predictions for cox survival models. Computes the concordance index and the cox loss function.

    See https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html.

    Parameters
    ----------
    tied_tol: float
        The tolerance value for considering ties.  See sksurv.metrics. concordance_index_censored.

    References
    ----------
    Pölsterl, S., 2020. scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn. J. Mach. Learn. Res., 21(212), pp.1-6.

    Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L., Rosati, R.A, "Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors", Statistics in Medicine, 15(4), 361-87, 1996.
    """
    def __init__(self, tied_tol=1e-8, train_times=None, train_events=None):
        self.tied_tol = tied_tol
        self.reset_tracking()

        self.train_times = train_times
        self.train_events = train_events
        self._set_eval_times()

    def get_metrics(self):
        z, y_true = self.get_tracked_data()

        censor = y_true[:, 0].astype(bool)
        event_indicator = ~censor
        survival_time_true = y_true[:, 1]

        out = self.compute_surv_metrics(pred_risk_score=z,
                                        events=event_indicator,
                                        times=survival_time_true)

        out['cox_loss'] = get_cox_loss(z=z, y=y_true)

        return out


class RankSurvivalEvaler(BaseStreamEvaler, SurvMetricsMixin):
    """
    Evaulation object handling batch predictions for the survival ranking loss.

    See https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html.

    Parameters
    ----------
    tied_tol: float
        The tolerance value for considering ties.  See sksurv.metrics. concordance_index_censored.

    phi: str
        phi argument for survival rank loss. See SurvRankingLoss's documentation.

    References
    ----------
    Pölsterl, S., 2020. scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn. J. Mach. Learn. Res., 21(212), pp.1-6.

    Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L., Rosati, R.A, "Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors", Statistics in Medicine, 15(4), 361-87, 1996.
    """
    def __init__(self, phi, tied_tol=1e-8, train_times=None, train_events=None):
        self.phi = phi
        self.tied_tol = tied_tol
        self.reset_tracking()

        self.train_times = train_times
        self.train_events = train_events
        self._set_eval_times()

    def get_metrics(self):
        z, y_true = self.get_tracked_data()

        censor = y_true[:, 0].astype(bool)
        event_indicator = ~censor
        survival_time_true = y_true[:, 1]

        out = self.compute_surv_metrics(pred_risk_score=z,
                                        events=event_indicator,
                                        times=survival_time_true)

        out['rank_loss'] = get_rank_loss(z=z, y=y_true, phi=self.phi)

        return out


def pred_clf(z):
    """
    Gets classification predictions from the z input.

    Parameters
    ----------
    z: shape (n_samples, n_classes)
        The unnormalized class scores.

    Output
    ------
    y_pred, prob_pred

    y_pred: shape (n_samples, )
        The predicted class label indices.

    prob_pred: shape (n_samples, n_classes) or (n_samples, )
        The predicted probabilities for each class. Returns

    """
    assert z.ndim == 2

    prob = softmax(z, axis=1)
    y = prob.argmax(axis=1)

    if prob.shape[1] == 2:
        prob = prob[:, 1]

    return y, prob


def pred_discr_surv(z):
    """
    Gets risk score predictions from the z input for the discrete survival loss.

    Parameters
    ----------
    z: shape (n_samples, n_bins)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).

    Output
    ------
    risk_scores: shape (n_samples, )
        The predicted risk scores.
    """
    hazards = expit(z)
    S = np.cumprod(1 - hazards, axis=1)
    risk = -S.sum(axis=1)
    return risk


def get_cox_loss(z, y):
    """
    Returns the cox loss.

    Parameters
    ----------
    z: array-like, (n_samples, )

    y: array-like, (n_samples, 2)
        First column is censorship indicator.
        Second column is survival time.

    Output
    ------
    loss: float
    """

    with torch.no_grad():

        # format to torch
        z = torch.from_numpy(z)
        c_t = torch.from_numpy(y)

        # setup loss func
        loss_func = CoxLoss(reduction='mean')
        loss = loss_func(z, c_t)
        return loss.detach().cpu().numpy().item()


def get_rank_loss(z, y, phi):
    """
    Returns the ranking loss.

    Parameters
    ----------
    z: array-like, (n_samples, )

    y: array-like, (n_samples, 2)
        First column is censorship indicator.
        Second column is survival time.

    phi: str
        The phi argument for SurvRankingLoss.

    Output
    ------
    loss: float
    """

    with torch.no_grad():

        # format to torch
        z = torch.from_numpy(z)
        c_t = torch.from_numpy(y)

        # setup loss func
        loss_func = SurvRankingLoss(phi=phi, reduction='mean')

        loss = loss_func(z, c_t)
        return loss.detach().cpu().numpy().item()
