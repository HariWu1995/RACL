import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
try:
    import tensorflow.contrib as tfc
except ModuleNotFoundError:
    pass


def _bernoulli(shape, mean):
    return tf.nn.relu(
        tf.sign(mean-tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32))
    )


class DropBlock2D(tf.keras.layers.Layer):
    
    # Adopted from https://github.com/DHZS/tf-dropblock 

    def __init__(self, keep_prob, block_size, scale=True, **kwargs):
        super(DropBlock2D, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 4
        _, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
        p1 = (self.block_size-1) // 2
        p0 = (self.block_size-1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]
        self.set_keep_prob()
        super(DropBlock2D, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(self.scale,
                             true_fn=lambda: output * tf.to_float(tf.size(mask)) / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output

        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs,
                         false_fn=drop)
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        w, h = tf.to_float(self.w), tf.to_float(self.h)
        dropout_rate = 1. - self.keep_prob
        self.gamma = dropout_rate*(w*h) / (self.block_size**2) / ((w-self.block_size+1)*(h-self.block_size+1))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                       self.h-self.block_size+1,
                                       self.w-self.block_size+1,
                                       self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        # mask = tf.distributions.Bernoulli(probs=self.gamma).sample(sampling_mask_shape)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], padding='SAME')
        mask = 1 - mask
        return mask


def RACL_block(aspect_input, opinion_input, context_input, context_query, word_mask, mask, position_att, opt, initializer, dropblock_layers, is_training):
    DropBlock_aspect, DropBlock_opinion, DropBlock_sentiment = dropblock_layers

    # Private Feature Extraction
    aspect_conv = tf.layers.conv1d(aspect_input, opt.n_filters, opt.kernel_size, padding='SAME', activation=tf.nn.relu, name='aspect_conv')
    opinion_conv = tf.layers.conv1d(opinion_input, opt.n_filters, opt.kernel_size, padding='SAME', activation=tf.nn.relu, name='opinion_conv')
    context_conv = tf.layers.conv1d(context_input, opt.emb_dim, opt.kernel_size, padding='SAME', activation=tf.nn.relu, name='context_conv')

    # Normalize
    aspect_conv_norm = tf.nn.l2_normalize(aspect_conv, axis=-1, name='aspect_conv_norm')
    opinion_conv_norm = tf.nn.l2_normalize(opinion_conv, axis=-1, name='opinion_conv_norm')    
    context_conv_norm = tf.nn.l2_normalize(context_conv, axis=-1, name='context_conv_norm')
    # aspect_conv = tfc.layers.batch_norm(aspect_conv, scale=True, center=True, is_training=is_training, trainable=True, scope='aspect_conv_batchnormed')
    # opinion_conv = tfc.layers.batch_norm(opinion_conv, scale=True, center=True, is_training=is_training, trainable=True, scope='opinion_conv_batchnormed')
    # context_conv = tfc.layers.batch_norm(context_conv, scale=True, center=True, is_training=is_training, trainable=True, scope='context_conv_batchnormed')

    # Relation R1    
    aspect_see_opinion = tf.matmul(aspect_conv_norm, opinion_conv_norm, adjoint_b=True)
    aspect_att_opinion = softmask_2d(aspect_see_opinion, word_mask)
    aspect_inter = tf.concat([aspect_conv, tf.matmul(aspect_att_opinion, opinion_conv)], axis=-1, name='aspect_inter')

    opinion_see_aspect = tf.matmul(opinion_conv_norm, aspect_conv_norm, adjoint_b=True)
    opinion_att_aspect = softmask_2d(opinion_see_aspect, word_mask)
    opinion_inter = tf.concat([opinion_conv, tf.matmul(opinion_att_aspect, aspect_conv)], axis=-1, name='opinion_inter')

    # AE & OE Prediction
    aspect_pred = tfc.layers.fully_connected(aspect_inter, opt.n_classes, activation_fn=None, weights_initializer=initializer, biases_initializer=initializer, scope='aspect_pred')
    opinion_pred = tfc.layers.fully_connected(opinion_inter, opt.n_classes, activation_fn=None, weights_initializer=initializer, biases_initializer=initializer, scope='opinion_pred')

    # OE Confidence - a slight difference from the original paper.
    # For propagating R3, we calculate the confidence of each candidate opinion word.
    # Only when a word satisfies the condition Prob[B,I] > Prob[O] in OE, it can be propagated to SC.
    opinion_condition = 1 - 2.*tf.nn.softmax(opinion_pred, axis=-1)[:,:,0]
    opinion_confidence = tf.maximum(0., opinion_condition, name='opinion_confidence')
    opinion_propagate = tf.tile(tf.expand_dims(opinion_confidence, axis=1), [1, opt.max_sentence_len, 1])
    opinion_propagate = opinion_propagate * mask * position_att

    # SC Aspect-Context Attention
    word_see_context = tf.matmul((context_query), context_conv_norm, adjoint_b=True)  * position_att
    word_att_context = softmask_2d(word_see_context, word_mask, scale=True)

    # Relation R2 & R3
    word_att_context += aspect_att_opinion + opinion_propagate
    context_inter = (context_query + tf.matmul(word_att_context, context_conv)) # query + value
                
    # SC Prediction
    sentiment_pred = tfc.layers.fully_connected(context_inter, opt.n_classes, activation_fn=None, \
                                                weights_initializer=initializer, biases_initializer=initializer, scope='sentiment_pred')

    # We use DropBlock to enhance the learning of the private features for AE, OE & SC.
    # For more details, refer to 
    #   http://papers.nips.cc/paper/8271-dropblock-a-regularization-method-for-convolutional-networks for more details.
    aspect_inter = tf.squeeze(DropBlock_aspect(inputs=tf.expand_dims(aspect_inter, axis=-1), training=is_training), axis=-1, name=f'aspect_inter')
    opinion_inter = tf.squeeze(DropBlock_opinion(inputs=tf.expand_dims(opinion_inter, axis=-1), training=is_training), axis=-1, name=f'opinion_inter')
    context_conv = tf.squeeze(DropBlock_sentiment(inputs=tf.expand_dims(context_conv, axis=-1), training=is_training), axis=-1, name=f'context_conv')
    
    return (aspect_pred, opinion_pred, sentiment_pred), (aspect_inter, opinion_inter, context_inter, context_conv)


def softmask_2d(x, mask, scale: bool=False):
    if scale:
        dim = tf.shape(x)[-1]
        max_x = tf.reduce_max(x, axis=-1, keepdims=True)
        max_x = tf.tile(max_x, [1, 1, dim])
        x -= max_x
    length = tf.shape(mask)[1]
    mask_d1 = tf.tile(tf.expand_dims(mask, 1), [1, length, 1])
    y = tf.multiply(tf.exp(x), mask_d1)
    sum_y = tf.reduce_sum(y, axis=-1, keepdims=True)
    att = y / (sum_y+1e-10)

    mask_d2 = tf.tile(tf.expand_dims(mask, 2), [1, 1, length])
    att *= mask_d2
    return att


if __name__ == "__main__":
    x = tf.keras.Input(shape=(12, 10, 100))
    y = DropBlock2D(0.5, 1)(x)
    model = tf.keras.Model(inputs=[x], outputs=[y], name='test')
    model.summary()