import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def weighted_cross_entropy(y_true, y_pred, pos_weight=7):
    losses = y_true * -K.log(y_pred) * pos_weight + (1-y_true) * -K.log(1-y_pred)
    losses = K.clip(losses, 0.0, 11)
    return K.mean(losses)


def balanced_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        background_weights = weights[:,0] ** (1-y_true)
        signal_weights = weights[:,1] ** (y_true)
        CE = K.binary_crossentropy(y_true, y_pred)
        losses = background_weights * signal_weights * CE
        losses = K.clip(losses, 0.0, 11.0)
        return K.mean(losses, axis=-1)
    return weighted_loss


def soft_crossentropy(preds, labels, weights=None, name='loss'):
    
    with tf.name_scope(name) as scope:
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels, name='unweighted_loss')

        if weights is None:
            loss = tf.reduce_mean(loss, name='final_loss')
            return loss

        class_weights = tf.constant(weights, name='class_weights')
        sample_weights = tf.reduce_sum(class_weights*labels, axis=1, name='sample_weights')
        loss = tf.reduce_mean(sample_weights*loss, name='weighted_loss')
        return loss


def RACL_losses(y_true, y_pred, masks, opt):
    ae_label, oe_label, sc_label = y_true
    ae_pred, oe_pred, sc_pred = y_pred
    word_mask, sentiment_mask = masks

    # Format predictions
    ae_pred = tf.cast(ae_pred, tf.float32, name='ae_pred')
    oe_pred = tf.cast(oe_pred, tf.float32, name='oe_pred')
    sc_pred = tf.cast(sc_pred, tf.float32, name='sc_pred')
    word_mask = tf.cast(word_mask, tf.float32, name='word_mask')
    sentiment_mask = tf.cast(sentiment_mask, tf.float32, name='sentiment_mask')

    # Convert values to probabilities
    ae_prob = tf.nn.softmax(ae_pred, axis=-1, name='ae_prob')
    oe_prob = tf.nn.softmax(oe_pred, axis=-1, name='oe_prob')
    # sc_prob = tf.nn.softmax(sc_pred, axis=-1, name='sc_prob')

    # Define shapes
    batch_size = ae_pred.shape[0]
    output_shape = [-1, opt.n_classes]
    mask_shape = [1, 1, opt.n_classes]

    # Mask AE, OE, SC Predictions
    word_mask = tf.tile(tf.expand_dims(word_mask, axis=-1), mask_shape)
    ae_pred = tf.reshape(word_mask*ae_pred, output_shape, name='ae_pred_masked')
    oe_pred = tf.reshape(word_mask*oe_pred, output_shape, name='oe_pred_masked')

    sentiment_mask = tf.tile(tf.expand_dims(sentiment_mask, axis=-1), mask_shape)
    sc_pred = tf.reshape(sentiment_mask*sc_pred, output_shape, name='sc_pred_masked')

    # Relation R4 (only in Training)
    # In training / validation, sentiment masks are set to 1.0 only for aspect terms.
    # In testing, sentiment masks are set to 1.0 for all words (except padded ones).

    # Format Labels
    ae_label = tf.cast(ae_label, tf.float32, name='ae_label')
    oe_label = tf.cast(oe_label, tf.float32, name='oe_label')
    sc_label = tf.cast(sc_label, tf.float32, name='sc_label')
    ae_label = tf.reshape(ae_label, output_shape, name='ae_label_flat')
    oe_label = tf.reshape(oe_label, output_shape, name='oe_label_flat')
    sc_label = tf.reshape(sc_label, output_shape, name='sc_label_flat')

    # AE & OE Regularization cost - only get Beginning [1] and Inside [2] values
    ae_cost = tf.reduce_sum(ae_prob[:,:,1:], axis=-1, name='ae_cost')
    oe_cost = tf.reduce_sum(oe_prob[:,:,1:], axis=-1, name='oe_cost')
    total_cost = ae_cost + oe_cost - 1.
    total_cost = tf.maximum(0., total_cost, name='total_cost')
    reg_cost = tf.reduce_sum(total_cost) / tf.reduce_sum(word_mask)
    reg_cost = tf.identity(reg_cost, name='regularization_cost') 

    # Categorical Cross-Entropy for AE, OE, SC
    ae_loss = soft_crossentropy(ae_pred, ae_label, opt.class_weights, name='aspect')
    oe_loss = soft_crossentropy(oe_pred, oe_label, opt.class_weights, name='opinion')
    sc_loss = soft_crossentropy(sc_pred, sc_label, opt.class_weights, name='sentiment')

    loss = opt.aspect_weight * ae_loss + \
            opt.opinion_weight * oe_loss + \
            opt.sentiment_weight * sc_loss + \
            opt.regularization_weight * reg_cost
    loss = tf.identity(loss, name='overall_loss')
    return loss, ae_loss, oe_loss, sc_loss, reg_cost


def RACL_losses_wrapper(opt):
    def call_RACL_losses(y_true, y_pred):
        # Split predictions into word_mask, sentiment_mask, aspect_pred, opinion_pred, sentiment_pred
        word_mask, sentiment_mask, *predictions = tf.split(y_pred, [1,1,3,3,3], axis=-1)
        word_mask = tf.squeeze(word_mask, axis=-1)
        sentiment_mask = tf.squeeze(sentiment_mask, axis=-1)

        # Split labels into aspect_label, opinion_label, sentiment_label
        labels = tf.split(y_true, [1,1,1], axis=-1)

        # Call RACL_losses
        losses = RACL_losses(predictions, labels, [word_mask, sentiment_mask], opt)
        return losses[0]
    return call_RACL_losses
            


