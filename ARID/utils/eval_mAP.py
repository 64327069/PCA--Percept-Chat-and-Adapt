# This code is originally from the official Youtube-8M repo
# https://github.com/google/youtube-8m/
# Small modification from Youtube-8M Code

import numpy as np
import os

import heapq
import random
import numbers
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")



class AveragePrecisionCalculator(object):
  """Calculate the average precision and average precision at n."""

  def __init__(self, top_n=None):
    """Construct an AveragePrecisionCalculator to calculate average precision.
    This class is used to calculate the average precision for a single label.
    Args:
      top_n: A positive Integer specifying the average precision at n, or None
        to use all provided data points.
    Raises:
      ValueError: An error occurred when the top_n is not a positive integer.
    """
    if not ((isinstance(top_n, int) and top_n >= 0) or top_n is None):
      raise ValueError("top_n must be a positive integer or None.")

    self._top_n = top_n  # average precision at n
    self._total_positives = 0  # total number of positives have seen
    self._heap = []  # max heap of (prediction, actual)

  @property
  def heap_size(self):
    """Gets the heap size maintained in the class."""
    return len(self._heap)

  @property
  def num_accumulated_positives(self):
    """Gets the number of positive samples that have been accumulated."""
    return self._total_positives

  def accumulate(self, predictions, actuals, num_positives=None):
    """Accumulate the predictions and their ground truth labels.
    After the function call, we may call peek_ap_at_n to actually calculate
    the average precision.
    Note predictions and actuals must have the same shape.
    Args:
      predictions: a list storing the prediction scores.
      actuals: a list storing the ground truth labels. Any value larger than 0
        will be treated as positives, otherwise as negatives. num_positives = If
        the 'predictions' and 'actuals' inputs aren't complete, then it's
        possible some true positives were missed in them. In that case, you can
        provide 'num_positives' in order to accurately track recall.
    Raises:
      ValueError: An error occurred when the format of the input is not the
      numpy 1-D array or the shape of predictions and actuals does not match.
    """
    if len(predictions) != len(actuals):
      raise ValueError("the shape of predictions and actuals does not match.")

    if num_positives is not None:
      if not isinstance(num_positives, numbers.Number) or num_positives < 0:
        raise ValueError(
            "'num_positives' was provided but it was a negative number.")

    if num_positives is not None:
      self._total_positives += num_positives
    else:
      self._total_positives += np.size(
          np.where(np.array(actuals) > 1e-5))
    topk = self._top_n
    heap = self._heap

    for i in range(np.size(predictions)):
      if topk is None or len(heap) < topk:
        heapq.heappush(heap, (predictions[i], actuals[i]))
      else:
        if predictions[i] > heap[0][0]:  # heap[0] is the smallest
          heapq.heappop(heap)
          heapq.heappush(heap, (predictions[i], actuals[i]))

  def clear(self):
    """Clear the accumulated predictions."""
    self._heap = []
    self._total_positives = 0

  def peek_ap_at_n(self):
    """Peek the non-interpolated average precision at n.
    Returns:
      The non-interpolated average precision at n (default 0).
      If n is larger than the length of the ranked list,
      the average precision will be returned.
    """
    if self.heap_size <= 0:
      return 0
    predlists = np.array(list(zip(*self._heap)))

    ap = self.ap_at_n(predlists[0],
                      predlists[1],
                      n=self._top_n,
                      total_num_positives=self._total_positives)
    return ap

  @staticmethod
  def ap(predictions, actuals):
    """Calculate the non-interpolated average precision.
    Args:
      predictions: a numpy 1-D array storing the sparse prediction scores.
      actuals: a numpy 1-D array storing the ground truth labels. Any value
        larger than 0 will be treated as positives, otherwise as negatives.
    Returns:
      The non-interpolated average precision at n.
      If n is larger than the length of the ranked list,
      the average precision will be returned.
    Raises:
      ValueError: An error occurred when the format of the input is not the
      numpy 1-D array or the shape of predictions and actuals does not match.
    """
    return AveragePrecisionCalculator.ap_at_n(predictions, actuals, n=None)

  @staticmethod
  def ap_at_n(predictions, actuals, n=20, total_num_positives=None):
    """Calculate the non-interpolated average precision.
    Args:
      predictions: a numpy 1-D array storing the sparse prediction scores.
      actuals: a numpy 1-D array storing the ground truth labels. Any value
        larger than 0 will be treated as positives, otherwise as negatives.
      n: the top n items to be considered in ap@n.
      total_num_positives : (optionally) you can specify the number of total
        positive in the list. If specified, it will be used in calculation.
    Returns:
      The non-interpolated average precision at n.
      If n is larger than the length of the ranked list,
      the average precision will be returned.
    Raises:
      ValueError: An error occurred when
      1) the format of the input is not the numpy 1-D array;
      2) the shape of predictions and actuals does not match;
      3) the input n is not a positive integer.
    """
    if len(predictions) != len(actuals):
      raise ValueError("the shape of predictions and actuals does not match.")

    if n is not None:
      if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be 'None' or a positive integer."
                         " It was '%s'." % n)

    ap = 0.0
    if not isinstance(predictions,np.ndarray):
      predictions = np.array(predictions)
    if not isinstance(actuals,np.ndarray):
      actuals = np.array(actuals)

    # add a shuffler to avoid overestimating the ap
    predictions, actuals = AveragePrecisionCalculator._shuffle(
        predictions, actuals)
    sortidx = sorted(range(len(predictions)),
                     key=lambda k: predictions[k],
                     reverse=True)

    if total_num_positives is None:
      numpos = np.size(np.where(actuals > 0))
    else:
      numpos = total_num_positives

    if numpos == 0:
      return 0

    if n is not None:
      numpos = min(numpos, n)
    delta_recall = 1.0 / numpos
    poscount = 0.0

    # calculate the ap
    r = len(sortidx)
    if n is not None:
      r = min(r, n)
    for i in range(r):
      if actuals[sortidx[i]] > 0:
        poscount += 1
        ap += poscount / (i + 1) * delta_recall
    return ap

  @staticmethod
  def _shuffle(predictions, actuals):
    random.seed(0)
    suffidx = random.sample(range(len(predictions)), len(predictions))
    predictions = predictions[suffidx]
    actuals = actuals[suffidx]
    return predictions, actuals

  @staticmethod
  def _zero_one_normalize(predictions, epsilon=1e-7):
    """Normalize the predictions to the range between 0.0 and 1.0.
    For some predictions like SVM predictions, we need to normalize them before
    calculate the interpolated average precision. The normalization will not
    change the rank in the original list and thus won't change the average
    precision.
    Args:
      predictions: a numpy 1-D array storing the sparse prediction scores.
      epsilon: a small constant to avoid denominator being zero.
    Returns:
      The normalized prediction.
    """
    denominator = np.max(predictions) - np.min(predictions)
    ret = (predictions - np.min(predictions)) / np.max(
        denominator, epsilon)
    return ret
def index2onehot(index):
    onehot = [0]*17
    for i in index:
      onehot[i] = 1
    return onehot
def MAP(label_list,pre_list,num_cls = 17):
    #pre_dict = sorted(pre_dict)
    
    #assert operator.eq(list(label_dict.keys()),list(pre_dict.keys()))
    # if type(pre_list) != list:
    #   pre_list = pre_list.tolist()

    # if type(label_list) != list:
    #   label_list = label_list.tolist()
    

    #label_list = list(label_dict.values())
    # label_list = list(map(index2onehot,label_list))
    
    label_arrary = np.array(label_list)

    #pre_list = list(pre_dict.values())
    pre_arrary = np.array(pre_list)
    
    calculator = AveragePrecisionCalculator()
    p_list = np.vsplit(pre_arrary.T,num_cls)
    p_list = list(map(lambda x: x.squeeze(),p_list))
    a_list = np.vsplit(label_arrary.T,num_cls)
    a_list = list(map(lambda x: x.squeeze(),a_list))
    ap_list = list(map(lambda p,a: calculator.ap(p,a), p_list, a_list))

    cls_label_list = ['ZC', 'CJ', 'CK', 'ZW', 'JG', 'SG', 'PL', 'BX', 'CR', 'FZ', 'FS', 'AJ', 'CQ', 'SL', 'QF', 'TJ', 'TL']
    '''
    print('AP:')
    for i, cls in enumerate(cls_label_list):
      print('{}:{:.3f}'.format(cls,ap_list[i]))
    print('mAP:{:.3f}'.format(np.mean(ap_list)))
    '''
    return np.mean(ap_list)

