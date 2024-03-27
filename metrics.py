import numpy as np
import torch

class EvalMetric(object):
    """
    Base class for all evaluation metrics.

    Parameters:
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self,
                 name,
                 output_names=None,
                 label_names=None,
                 **kwargs):
        super(EvalMetric, self).__init__()
        self.name = str(name)
        self.output_names = output_names
        self.label_names = label_names
        self._has_global_stats = kwargs.pop("has_global_stats", False)
        self._kwargs = kwargs
        self.reset()

    def reset(self):
        """
        Resets the internal evaluation result to initial state.
        """
        self.num_inst = 0
        self.sum_metric = 0.0
        self.global_num_inst = 0
        self.global_sum_metric = 0.0

    def update(self, labels, preds):
        """
        Updates the internal evaluation result.

        Parameters:
        ----------
        labels : torch.Tensor
            The labels of the data.
        preds : torch.Tensor
            Predicted values.
        """
        raise NotImplementedError()
    def get(self):
        """
        Gets the current evaluation result.

        Returns:
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num_inst == 0:
            return self.name, float("nan")
        else:
            return self.name, self.sum_metric / self.num_inst


class Accuracy(EvalMetric):
    """
    Computes accuracy classification score.

    Parameters:
    ----------
    axis : int, default 1
        The axis that represents classes
    name : str, default 'accuracy'
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self,
                 axis=1,
                 name="accuracy",
                 output_names=None,
                 label_names=None):
        super(Accuracy, self).__init__(
            name,
            axis=axis,
            output_names=output_names,
            label_names=label_names,
            has_global_stats=True)
        self.axis = axis

    def update(self, labels, preds):
        """
        Updates the internal evaluation result.

        Parameters:
        ----------
        labels : torch.Tensor
            The labels of the data with class indices as values, one per sample.
        preds : torch.Tensor
            Prediction values for samples. Each prediction value can either be the class index,
            or a vector of likelihoods for all classes.
        """
        assert (len(labels) == len(preds))
        with torch.no_grad():
            if preds.shape != labels.shape:
                pred_label = torch.argmax(preds, dim=self.axis)
            else:
                pred_label = preds
            pred_label = pred_label.cpu().numpy().astype(np.int32)
            label = labels.cpu().numpy().astype(np.int32)

            label = label.flat
            pred_label = pred_label.flat

            num_correct = (pred_label == label).sum()
            self.sum_metric += num_correct
            self.global_sum_metric += num_correct
            self.num_inst += len(pred_label)
            self.global_num_inst += len(pred_label)

class Top1Error(Accuracy):
    """
    Computes top-1 error (inverted accuracy classification score).

    Parameters:
    ----------
    axis : int, default 1
        The axis that represents classes.
    name : str, default 'top_1_error'
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self,
                 axis=1,
                 name="top_1_error",
                 output_names=None,
                 label_names=None):
        super(Top1Error, self).__init__(
            axis=axis,
            name=name,
            output_names=output_names,
            label_names=label_names)

    def get(self):
        """
        Gets the current evaluation result.

        Returns:
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num_inst == 0:
            return self.name, float("nan")
        else:
            return self.name, 1.0 - self.sum_metric / self.num_inst
