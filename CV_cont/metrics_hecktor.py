# Third party inports
import tensorflow as tf
import keras.backend as K
import numpy as np
import pdb

from lifelines.utils import concordance_index





def Dice_coeff(y_true, y_pred, epsilon=1e-5):
    
    # https://github.com/milesial/Pytorch-UNet/blob/master/utils/dice_score.py
    
    # Determine axes for summing.
    ndims = len(y_pred.get_shape().as_list()) - 2
    vol_axes = list(range(1, ndims+1))

    # Intersection and union
    inter = 2.0 * tf.reduce_sum(y_true * y_pred, axis=vol_axes)
    sets_sum = tf.reduce_sum(y_true, axis=vol_axes) + tf.reduce_sum(y_pred, axis=vol_axes)

    # Avoid division by zero by replacing 0 in sets_sum with inter.
    sets_sum_safe = tf.where(tf.equal(sets_sum, 0.0), inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum_safe + epsilon)
    
    # Create sample mask: skip samples where y_true is all zeros.
    mask = tf.reduce_sum(y_true, axis=vol_axes) > 0
    mask = tf.cast(mask, tf.float32)
    
    dice_val = tf.reduce_sum(dice * mask) / (tf.reduce_sum(mask) + 1e-8)

    return dice_val





def Accuracy(y_true, y_pred):
    """
    Computes accuracy ignoring unlabeled samples (where y_true == 0).
    y_true: Tensor of shape (batch_size, 1), values in {0, 1, 2}
    y_pred: Tensor of shape (batch_size, 1), sigmoid output in [0, 1]
    """
    
    # Unlabeled sample ---> y_true=0.
    # Mask where y_true != 0.
    mask = tf.not_equal(y_true, 0)
    mask = tf.cast(mask, tf.float32)

    # Convert y_true: 1 → 0, 2 → 1
    y_true_label = tf.where(tf.equal(y_true, 1), tf.zeros_like(y_true), tf.ones_like(y_true))
    y_true_label = tf.cast(y_true_label, tf.float32)

    # Predicted label: threshold at 0.5.
    y_pred_label = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)

    # Correct predictions.
    correct = tf.cast(tf.equal(y_pred_label, y_true_label), tf.float32)

    accuracy = tf.reduce_sum(correct * mask) / (tf.reduce_sum(mask) + 1e-8)
    
    return accuracy





def Balanced_accuracy(y_true, y_pred, threshold=0.5):
    """
    Balanced Accuracy metric for binary classification in TensorFlow 1.15.

    y_true: tensor of shape (batch_size, 1), values: 0 (ignore), 1 (class 0), 2 (class 1)
    y_pred: tensor of shape (batch_size, 1), sigmoid outputs
    """
    # Mask to ignore unlabeled samples (y_true == 0)
    mask = tf.not_equal(y_true, 0)
    mask = tf.cast(mask, tf.float32)

    # Convert labels: 1 -> 0 (class 0), 2 -> 1 (class 1)
    y_true_label = tf.where(tf.equal(y_true, 1), tf.zeros_like(y_true), tf.ones_like(y_true))
    y_true_label = tf.cast(y_true_label, tf.float32)

    # Apply threshold to predictions
    y_pred_label = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)

    # Flatten
    y_true_bin = tf.reshape(y_true_label, [-1])
    y_pred_bin = tf.reshape(y_pred_label, [-1])
    mask = tf.reshape(mask, [-1])

    # Apply mask
    y_true_bin = y_true_bin * mask
    y_pred_bin = y_pred_bin * mask

    # Count TP, TN, FP, FN
    TP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_bin, 1), tf.equal(y_pred_bin, 1)), tf.float32))
    TN = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_bin, 0), tf.equal(y_pred_bin, 0)), tf.float32))
    P = tf.reduce_sum(tf.cast(tf.equal(y_true_bin, 1), tf.float32))
    N = tf.reduce_sum(tf.cast(tf.equal(y_true_bin, 0), tf.float32))

    # Avoid division by zero
    TPR = TP / (P + 1e-8)
    TNR = TN / (N + 1e-8)

    # Balanced Accuracy
    balanced_acc = (TPR + TNR) / 2.0

    return balanced_acc





def Cindex(y_true, y_pred):
    '''
    C-index score for risk prediction.
    y_pred is the predicted risk from trained model.
    y_true is event indicator, event=0 means censored
    Survival time is not required as input
    Samples should be sorted with increasing survival time 
    '''
    
    risk = y_pred
    event = tf.cast(y_true, risk.dtype)
    
    g = tf.subtract(risk, risk[:,0])
    g = tf.cast(g == 0.0, risk.dtype) * 0.5 + tf.cast(g > 0.0, risk.dtype)

    
    f = tf.matmul(event, tf.cast(tf.transpose(event)>-1, risk.dtype)) 
    f = tf.linalg.band_part(f, 0, -1) - tf.linalg.band_part(f, 0, 0)

    top = tf.reduce_sum(tf.multiply(g, f))
    bottom = tf.reduce_sum(f)
    
    
    cindex = top/bottom
    
    
    cindex = tf.cond(tf.equal(bottom, 0),
                    lambda: tf.constant(0.0, dtype=cindex.dtype),
                    lambda: cindex)

    return cindex