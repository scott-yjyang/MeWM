import numpy as np
import pandas as pd 
import sys
import os

import torch
import torch.nn as nn

# Import from sibling directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + ".")
from breslow_estimator import BreslowEstimator

### DeepSurv Loss. For 1-dim model output. 
# Cox models use the Breslow estimator to obtain indiv. risk curves from the baseline hazard of 
# the population at each time point multiplied by the individual log partial hazard (= the model 
# output, also called hazard ratio).
# For the loss, however, it is sufficient to input the model outputs.
# The loss is explained here https://k-d-w.org/blog/2019/07/survival-analysis-for-deep-learning

# Adapted from the auton-survival package
# https://github.com/autonlab/auton-survival/blob/master/auton_survival/models/cph/dcph_utilities.py

def partial_ll_loss(lrisks, tb, eb, eps=1e-3): 
    """
    Compute the partial log-likelihood loss.

    Args:
    lrisks: log risks (n, )
    tb: time of events in batch (n, )
    eb: event indicator in batch (n, )
    eps: small value for numerical stability"""
  
    def _reshape_tensor_with_nans(data):
        data = data.reshape(-1)
        return data[~torch.isnan(data)]
  
    tb = _reshape_tensor_with_nans(tb).detach().cpu()
    eb = _reshape_tensor_with_nans(eb).detach().cpu()

    # Use torch instead of numpy, stay on gpu
    tb = tb + eps*torch.rand(len(tb))
    sindex = torch.argsort(-tb)

    tb = tb[sindex]
    eb = eb[sindex]

    lrisks = lrisks[sindex]
    lrisksdenom = torch.logcumsumexp(lrisks, dim = 0)

    plls = lrisks - lrisksdenom
    pll = plls[eb == 1]

    pll = torch.sum(pll)

    return -pll

class NLLDeepSurvLoss(nn.Module):
    def __init__(self):
        super(NLLDeepSurvLoss, self).__init__()
    
    def forward(self, hazard_ratio, durations, events):
        loss = partial_ll_loss(hazard_ratio, durations, events)
        return loss

def predict_survival(breslow: BreslowEstimator, preds, times=None):
    """Predict survival function from model and data using Breslow estimator.
    
    Args:
        breslow: Breslow estimator 
            (output of fit_breslow() after training or init_breslow() after loading)
        preds: partial hazard predictions (n, )
        times: time points to include for interpolation (optional)
    """

    if isinstance(times, (int, float)): times = [times]

    preds = preds.detach().cpu().numpy()

    with np.errstate(divide='ignore', over='ignore'):
        unique_times = breslow.unique_times_

        raw_predictions = breslow.get_survival_function(preds)
        raw_predictions = np.array([pred.y for pred in raw_predictions])

        predictions = pd.DataFrame(data=raw_predictions, columns=unique_times)

        if times is None:
            return predictions
        else:
            predictions = __interpolate_missing_times(predictions.T, times)
            # Clip values between 0 and 1
            predictions[predictions > 1] = 1
            predictions[predictions < 0] = 0
            return predictions

def __interpolate_missing_times(survival_predictions, times):

    nans = np.full(survival_predictions.shape[1], np.nan)
    not_in_index = list(set(times) - set(survival_predictions.index))

    for idx in not_in_index:
        survival_predictions.loc[idx] = nans
    return survival_predictions.sort_index(axis=0).bfill().ffill().T[times].values