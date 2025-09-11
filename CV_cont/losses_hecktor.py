# Third party inports
import tensorflow as tf
import keras.backend as K
import numpy as np
import pdb

import metrics_hecktor





def Dice_coeff_loss(y_true, y_pred, epsilon=1e-5):
    """
    https://github.com/milesial/Pytorch-UNet/blob/master/utils/dice_score.py
    Masked Dice Loss: skips samples where y_true is all zeros.
    """
    # Determine axes for summing.
    ndims = len(y_pred.get_shape().as_list()) - 2
    # excludes batch and channel.
    vol_axes = list(range(1, ndims + 1)) 

    # Intersection and union per sample.
    inter = 2.0 * tf.reduce_sum(y_true * y_pred, axis=vol_axes)
    sets_sum = tf.reduce_sum(y_true, axis=vol_axes) + tf.reduce_sum(y_pred, axis=vol_axes) 
    
    # Replace zero denominator to avoid division by zero.
    sets_sum_safe = tf.where(tf.equal(sets_sum, 0.0), inter, sets_sum)
    
    dice = (inter + epsilon) / (sets_sum_safe + epsilon)

    # Create sample mask: skip samples where y_true is all zeros.
    mask = tf.reduce_sum(y_true, axis=vol_axes) > 0
    mask = tf.cast(mask, tf.float32)

    # Apply mask.
    masked_dice = tf.reduce_sum(dice * mask) / (tf.reduce_sum(mask) + 1e-8)

    seg_loss = 1.0 - masked_dice
    
    return seg_loss





def Focal_loss(y_true, y_pred, alpha=0.25, gamma=2, epsilon=1e-5):
    """
    Masked Focal Loss: skips samples where y_true is all zeros.
    """

    # Clamp prediction to avoid log(0).
    y_pred_clamp = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    logits = tf.math.log(y_pred_clamp / (1 - y_pred_clamp))

    weight_a = alpha * tf.pow((1 - y_pred_clamp), gamma) * y_true
    weight_b = (1 - alpha) * tf.pow(y_pred_clamp, gamma) * (1 - y_true)
    focal_val = tf.math.log1p(tf.exp(-logits)) * (weight_a + weight_b) + logits * weight_b
    
    ndims = len(y_true.get_shape().as_list()) - 2
    vol_axes = list(range(1, ndims + 1))  

    # Mean over spatial dims.
    focal_per_sample = tf.reduce_mean(focal_val, axis=vol_axes)  

    # Mask: exclude samples with empty ground truth
    mask = tf.reduce_sum(y_true, axis=vol_axes) > 0  # shape = [B, 1]
    mask = tf.cast(mask, tf.float32)

    # Apply mask and normalize
    seg_loss = tf.reduce_sum(focal_per_sample * mask) / (tf.reduce_sum(mask) + 1e-8)

    return seg_loss





def Seg_loss(y_true, y_pred):

    return Dice_coeff_loss(y_true, y_pred) + Focal_loss(y_true, y_pred)





def Binary_focal_crossentropy(y_true, y_pred, alpha=0.25, gamma=2, epsilon=1e-5):
    """
    https://github.com/umbertogriffo/focal-loss-keras/blob/master/src/loss_function/losses.py
    y_true: true labels tensor (batch_size, 1)
    y_pred: predicted labels tensor (batch_size, 1), output of sigmoid activation
    """
    
    # Unlabeled sample ---> y_true=0.
    # Mask where y_true != 0.
    mask = tf.not_equal(y_true, 0)
    mask = tf.cast(mask, tf.float32)

    # The original y_true contain 1 for class0 and 2 for class1.
    # Convert 1 → 0, 2 → 1.
    y_true_label = tf.where(tf.equal(y_true, 1), tf.zeros_like(y_true), tf.ones_like(y_true))
    y_true_label = tf.cast(y_true_label, tf.float32)

    # Avoid log(0) by clipping predictions.
    y_pred_label = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    # Compute p_t.
    p_t = tf.where(tf.equal(y_true_label, 1), y_pred_label, 1 - y_pred_label)

    # Compute alpha_t.
    alpha_factor = tf.ones_like(y_true_label) * alpha
    alpha_t = tf.where(tf.equal(y_true_label, 1), alpha_factor, 1 - alpha_factor)
    
    # Calculate cross entropy.
    cross_entropy = -tf.math.log(p_t)

    # Compute focal weight.
    weight = alpha_t * tf.pow(1 - p_t, gamma)

    # Compute focal loss.
    loss = weight * cross_entropy
    
    # Apply mask.
    loss = tf.reshape(loss, [-1])
    mask = tf.reshape(mask, [-1])

    classification_loss = tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)
    
    return classification_loss




def Cox_loss(y_true, y_pred):
    '''
    Calculate the average Cox negative partial log-likelihood.
    y_pred is the predicted risk from trained model.
    y_true is event indicator, event=0 means censored
    Survival time is not required as input
    Samples should be sorted with increasing survival time 
    '''
    
    risk = y_pred
    event = tf.cast(y_true[:,1:], dtype=risk.dtype)
    
    risk_exp = tf.exp(risk)
    risk_exp_cumsum = tf.cumsum(risk_exp, reverse=True)
    likelihood = risk - tf.math.log(risk_exp_cumsum)
    uncensored_likelihood = tf.multiply(likelihood, event)
    
    n_observed = tf.reduce_sum(event)
    cox_loss = -tf.reduce_sum(uncensored_likelihood)/n_observed
    
    cox_loss = tf.cond(tf.equal(n_observed, 0),
                       lambda: tf.constant(0.0, dtype=cox_loss.dtype),
                       lambda: cox_loss)

    return cox_loss


def SurvivalContrastiveLoss(features, survival_times, event_indicators, margin=1.0, temperature=0.1):
    """
    Contrastive loss for survival analysis that encourages similar survival times
    to have similar representations and different survival times to be separated.
    """
    """
    Args:
        features: Extracted features [batch_size, feature_dim]
        survival_times: Survival times [batch_size]
        event_indicators: Event indicators [batch_size]
    """
    batch_size = features.size(0)
    
    # Normalize features
    features = tf.math.l2_normalize(features, dim=1)
    
    # Compute pairwise distances
    #distance_matrix = torch.cdist(features, features, p=2)
    distance_matrix = tf.norm(tf.expand_dims(features, 1) - tf.expand_dims(features, 0), ord=2, axis=-1)
    
    # Create similarity targets based on survival times
    time_diff_matrix = tf.math.abs(survival_times.unsqueeze(0) - survival_times.unsqueeze(1))
    
    # Similar pairs: small time differences and both have events
    event_matrix = event_indicators.unsqueeze(0) * event_indicators.unsqueeze(1)
    similar_mask = (time_diff_matrix < tf.median(time_diff_matrix)) & (event_matrix > 0)
    
    # Dissimilar pairs: large time differences
    #quant = torch.quantile(time_diff_matrix, 0.75)
    quant = tf.keras.ops.quantile(time_diff_matrix, 0.75) #q=0.75
    dissimilar_mask = time_diff_matrix > quant
    
    # Remove diagonal
    #eye_mask = torch.eye(batch_size, device=features.device).bool()
    #tf.eye(num_rows,num_columns=None,batch_shape=None,dtype=tf.dtypes.float32,name=None)
    #torch.eye(n, m=None, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    eye_mask = tf.eye(batch_size, device=features.device).bool()
    similar_mask = similar_mask & ~eye_mask
    dissimilar_mask = dissimilar_mask & ~eye_mask
    
    #loss = torch.tensor(0.0, device=features.device, requires_grad=True)
    loss = tf.Variable(0.0, dtype=tf.float32, trainable=True)
    
    #if similar_mask.sum() > 0:
    #    # Similar pairs should be close
    #    similar_distances = distance_matrix[similar_mask]
    #    similar_loss = similar_distances.mean()
    #    loss = loss + similar_loss
    
    #if dissimilar_mask.sum() > 0:
    #    # Dissimilar pairs should be far apart
    #    dissimilar_distances = distance_matrix[dissimilar_mask]
    #    dissimilar_loss = torch.clamp(margin - dissimilar_distances, min=0).mean()
    #    loss = loss + dissimilar_loss
    
    if tf.reduce_sum(tf.cast(similar_mask, tf.float32)) > 0:
        # Similar pairs should be close
        similar_distances = tf.boolean_mask(distance_matrix, similar_mask)
        similar_loss = tf.reduce_mean(similar_distances)
        loss = loss + similar_loss
    
    if tf.reduce_sum(tf.cast(dissimilar_mask, tf.float32)) > 0:
        # Dissimilar pairs should be far apart
        dissimilar_distances = tf.boolean_mask(distance_matrix, dissimilar_mask)
        dissimilar_loss = tf.reduce_mean(tf.nn.relu(margin - dissimilar_distances))
        loss = loss + dissimilar_loss
    return loss


def Total_Loss(y_true, y_pred):
    loss_cox = Cox_loss(y_true, y_pred)
    contrastive_loss = SurvivalContrastiveLoss(features, survival_times, event_indicators)









    