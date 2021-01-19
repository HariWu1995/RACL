import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, InputLayer, Layer, Embedding, 
    Conv1D, Conv2D, Dropout, Dense, 
    Dot, Concatenate, Average, Add,
    Lambda, Reshape, 
    Softmax, Maximum, Minimum,
)
from tensorflow.keras.activations import softmax, sigmoid, relu
# from tensorflow.keras.initializers import Identity, GlorotNormal, GlorotUniform


def _bernoulli(shape, mean):
    return tf.nn.relu(
        tf.sign(mean-tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32))
    )


def to_float(tensor):
    return tf.cast(tensor, tf.float32)


################################## 
# Keras Layer - for RACL_v2_K.py #
##################################

class DropBlock2D(Layer):
    
    # Adopted from https://github.com/DHZS/tf-dropblock 

    def __init__(self, keep_prob: float=1.0, block_size: int=1, scale: bool=True, **kwargs):
        super(DropBlock2D, self).__init__(**kwargs)
        self.keep_prob = keep_prob
        self.block_size = block_size
        self.scale = tf.constant(scale, dtype=tf.bool)

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

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                        self.h-self.block_size+1,
                                        self.w-self.block_size+1,
                                        self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], padding='SAME')
        mask = 1 - mask
        return mask

    def call(self, inputs, training=None, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            mask_size = to_float(tf.size(mask))
            inputs_masked = inputs * mask
            output = tf.cond(self.scale,
                             true_fn=lambda: inputs_masked*mask_size/tf.reduce_sum(mask),
                             false_fn=lambda: inputs_masked)
            return output

        if training is None:
            training = K.learning_phase()
        output = tf.cond(
            tf.logical_or(tf.logical_not(training), 
                          tf.equal(self.keep_prob, 1.0)), 
            true_fn=lambda: inputs, 
            false_fn=drop
        )
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        w, h = to_float(self.w), to_float(self.h)
        dropout_rate = 1. - self.keep_prob
        self.gamma = dropout_rate*(w*h) / (self.block_size**2) / ((w-self.block_size+1)*(h-self.block_size+1))


class L2Norm(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(L2Norm, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True

    def call(self, inputs, mask=None):
        # return K.l2_normalize(inputs, axis=self.axis)
        return tf.math.l2_normalize(inputs, axis=self.axis)


class SoftMask2D(Layer):
    def __init__(self, scale: bool=False, **kwargs):
        super(SoftMask2D, self).__init__(**kwargs)
        self.scale = scale
        self.supports_masking = True

    def call(self, inputs):
        x, mask = inputs
        if self.scale:
            dim = tf.shape(x)[-1]
            max_x = tf.math.reduce_max(x, axis=-1, keepdims=True, name='max_x')
            max_x = tf.tile(max_x, [1, 1, dim], name='max_x_tiled')
            x = tf.math.subtract(x, max_x, name='x_scaled')
        length = tf.shape(mask)[1]
        mask_d1 = tf.tile(tf.expand_dims(mask, axis=1), [1, length, 1], name='mask_d1')
        y = tf.math.multiply(tf.exp(x), mask_d1, name='y')
        sum_y = tf.math.reduce_sum(y, axis=-1, keepdims=True, name='sum_y')
        att = tf.math.divide(y, sum_y+K.epsilon(), name='att')

        mask_d2 = tf.tile(tf.expand_dims(mask, axis=2), [1, 1, length], name='mask_d2')
        att = tf.math.multiply(att, mask_d2, name='att_masked')
        return att


class ExpandDim(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ExpandDim, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = False

    def call(self, inputs): 
        return tf.expand_dims(inputs, axis=self.axis)


class Squeeze(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(Squeeze, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = False

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)


class ReduceDim(Layer):
    def __init__(self, method: str='mean', axis=None, **kwargs):
        super(ReduceDim, self).__init__(**kwargs)
        self.axis = axis
        self.method = method.lower()
        self.supports_masking = False

    def call(self, inputs):
        if self.method == 'sum':
            return tf.math.reduce_sum(inputs, axis=self.axis)
        elif self.method == 'mean':
            return tf.math.reduce_mean(inputs, axis=self.axis)
        elif self.method == 'max':
            return tf.math.reduce_max(inputs, axis=self.axis)
        elif self.method == 'min':
            return tf.math.reduce_min(inputs, axis=self.axis)
        elif self.method == 'std':
            return tf.math.reduce_std(inputs, axis=self.axis)
        elif self.method == 'variance':
            return tf.math.reduce_variance(inputs, axis=self.axis)
        else:
            raise ValueError(f'method={self.method} has been implemented yet!')


class MatMul(Layer):
    def __init__(self, adjoint_a=False, adjoint_b=False, 
                    transpose_a=False, transpose_b=False,     
                    a_is_sparse=False, b_is_sparse=False, **kwargs):
        super(MatMul, self).__init__(**kwargs)
        self.adjoint_a, self.adjoint_b = adjoint_a, adjoint_b
        self.transpose_a, self.transpose_b = transpose_a, transpose_b
        self.a_is_sparse, self.b_is_sparse = a_is_sparse, b_is_sparse        
        self.supports_masking = False

    def call(self, inputs):
        args = {
            'a': inputs[0], 'b': inputs[1],
            'adjoint_a': self.adjoint_a, 'adjoint_b': self.adjoint_b,
            'transpose_a': self.transpose_a, 'transpose_b': self.transpose_b,
            'a_is_sparse': self.a_is_sparse, 'b_is_sparse': self.b_is_sparse,
        }
        return tf.linalg.matmul(**args)


class RACL_Block(Model):
    def __init__(self, opt, Normalizer, DropBlocks, TileBlock,
                conv_args: dict, dense_args: dict, block_id: int, **kwargs):
        super(RACL_Block, self).__init__(**kwargs)
        self._name = f'RACL_Block_{block_id}'
        self.opt = opt
        self.block_id = block_id

        self.Tile = TileBlock
        self.Normalizer = Normalizer
        self.DropBlock_aspect, self.DropBlock_opinion, self.DropBlock_context = DropBlocks

        self.Aspect_Extractor = Conv1D(filters=self.opt.n_filters, name=f'Aspect_Conv-{block_id}', **conv_args)
        self.Opinion_Extractor = Conv1D(filters=self.opt.n_filters, name=f'Opinion_Conv-{block_id}', **conv_args)
        self.Context_Extractor = Conv1D(filters=self.opt.emb_dim, name=f'Context_Conv-{block_id}', **conv_args)

        self.Aspect_Classifier = Dense(name=f'Aspect_Classifier-{block_id}', **dense_args)
        self.Opinion_Classifier = Dense(name=f'Opinion_Classifier-{block_id}', **dense_args)
        self.Sentiment_Classifier = Dense(name=f'Sentiment_Classifier-{block_id}', **dense_args)

    def call(self, inputs):
        aspect_input, opinion_input, context_input, context_query, word_mask, position_att = inputs
        i = self.block_id
        
        # Extract Private Features for each task
        aspect_conv = self.Aspect_Extractor(aspect_input)
        opinion_conv = self.Opinion_Extractor(opinion_input)
        context_conv = self.Context_Extractor(context_input)

        # Normalize
        aspect_conv_norm = self.Normalizer(aspect_conv)
        opinion_conv_norm = self.Normalizer(opinion_conv)
        context_conv_norm = self.Normalizer(context_conv)

        # Relation R1
        aspect_see_opinion = MatMul(adjoint_b=True, name=f'aspect_see_opinion-{i}')([aspect_conv_norm, opinion_conv_norm])
        aspect_attend_opinion = SoftMask2D(name=f'aspect_attend_opinion-{i}')([aspect_see_opinion, word_mask])
        aspect_weigh_opinion = MatMul(name=f'aspect_weigh_opinion-{i}')([aspect_attend_opinion, opinion_conv])
        aspect_interact = Concatenate(axis=-1, name=f'aspect_interact-{i}')([aspect_conv, aspect_weigh_opinion])

        opinion_see_aspect = MatMul(adjoint_b=True, name=f'opinion_see_aspect-{i}')([opinion_conv_norm, aspect_conv_norm])
        opinion_attend_aspect = SoftMask2D(name=f'opinion_attend_aspect-{i}')([opinion_see_aspect, word_mask])
        opinion_weigh_aspect = MatMul(name=f'opinion_weigh_aspect-{i}')([opinion_attend_aspect, aspect_conv])
        opinion_interact = Concatenate(axis=-1, name=f'opinion_interact-{i}')([opinion_conv, opinion_weigh_aspect])

        # AE & OE Prediction
        aspect_pred = self.Aspect_Classifier(aspect_interact)
        opinion_pred = self.Opinion_Classifier(opinion_interact)

        # OE Confidence - a slight difference from the original paper.
        # For propagating R3, we calculate the confidence of each candidate opinion word.
        # Only when a word satisfies the condition Prob[B,I] > Prob[O] in OE, it can be propagated to SC.
        opinion_condition = Lambda(lambda x: 1-2.*tf.nn.softmax(x, axis=-1)[:,:,0], name=f'opinion_condition-{i}')(opinion_pred)
        opinion_confidence = Lambda(lambda x: tf.math.maximum(0., x), name=f'opinion_confidence-{i}')(opinion_condition)
        mask = self.Tile(word_mask)
        opinion_propagated = self.Tile(opinion_confidence)
        opinion_propagated = MatMul(name=f'opinion_propagated_masked-{i}')([opinion_propagated, mask])
        opinion_propagated = MatMul(name=f'opinion_propagated-{i}')([opinion_propagated, position_att])

        # SC Aspect-Context Attention
        word_see_context = MatMul(adjoint_b=True, name=f'word_see_context-{i}')([(context_query), context_conv_norm])
        word_see_context = MatMul(name=f'word_see_context_masked-{i}')([word_see_context, position_att])
        word_attend_context = SoftMask2D(scale=True, name=f'word_attend_context-{i}')([word_see_context, word_mask])

        # Relation R2 & R3
        word_attend_context += aspect_attend_opinion + opinion_propagated
        word_weigh_context = MatMul(name=f'word_weigh_context-{i}')([word_attend_context, context_conv])
        context_interact = context_query + word_weigh_context

        # SC Prediction
        sentiment_pred = self.Sentiment_Classifier(context_interact)

        # We use DropBlock to enhance the learning of the private features for AE, OE & SC.
        # For more details, refer to 
        #   http://papers.nips.cc/paper/8271-dropblock-a-regularization-method-for-convolutional-networks for more details.
        aspect_interact = ExpandDim(axis=-1)(aspect_interact)
        aspect_interact = self.DropBlock_aspect(aspect_interact)
        aspect_interact = Squeeze(axis=-1)(aspect_interact)

        opinion_interact = ExpandDim(axis=-1)(opinion_interact)
        opinion_interact = self.DropBlock_opinion(opinion_interact)
        opinion_interact = Squeeze(axis=-1)(opinion_interact)
        
        context_conv = ExpandDim(axis=-1)(context_conv)
        context_conv = self.DropBlock_context(context_conv)
        context_conv = Squeeze(axis=-1)(context_conv)

        return [(aspect_pred, opinion_pred, sentiment_pred), 
                (aspect_interact, opinion_interact, context_interact, context_conv)]


def dropoutize_embeddings(opt, layer_name: str='embeddings_dropout', model_name: str='dropoutize_embeddings'):
    model = Sequential(name=model_name)
    model.add(Dropout(rate=1-opt.keep_prob_1, 
                      input_shape=(opt.max_sentence_len, opt.emb_dim), 
                      seed=opt.random_seed,
                      name=layer_name))
    word_inputs, word_embeddings = model.inputs, model.outputs
    return word_inputs[0], word_embeddings[0]


def create_embeddings(inputs, opt, embedding_dim: int, layer_prefix: str='', pretrained_embeddings=None):
    embedding_args = {
        'input_dim': opt.vocab_size+1, # MUST: +1
        'output_dim': embedding_dim,
        'name': layer_prefix+'_embeddings',
        'trainable': False # finetune only happens after warm-up
    }
    if pretrained_embeddings is not None \
        and pretrained_embeddings.shape==(embedding_args['input_dim'], embedding_args['output_dim']):
        embedding_args['weights'] = [pretrained_embeddings]
    embeddings = Embedding(**embedding_args)(inputs)
    embeddings = Dropout(1-opt.keep_prob_1, name=layer_prefix+'_embeddings_dropout')(embeddings)
    return embeddings


#################################################################### 
# Tensorflow functions - for RACL_v2.py.tf: bugs are still unfixed #
####################################################################
 
def softmask_2d(x, mask, scale: bool=False, scope: str='softmask_2d'):
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        if scale:
            dim = tf.shape(x)[-1]
            max_x = tf.math.reduce_max(x, axis=-1, keepdims=True, name='max_x')
            max_x = tf.tile(max_x, [1, 1, dim], name='max_x_tiled')
            x = tf.math.subtract(x, max_x, name='x_scaled')
        length = tf.shape(mask)[1]
        mask_d1 = tf.tile(tf.expand_dims(mask, axis=1), [1, length, 1], name='mask_d1')
        y = tf.math.multiply(tf.math.exp(x), mask_d1, name='y')
        sum_y = tf.math.reduce_sum(y, axis=-1, keepdims=True, name='sum_y')
        att = tf.math.divide(y, sum_y+K.epsilon(), name='att')

        mask_d2 = tf.tile(tf.expand_dims(mask, axis=2), [1, 1, length], name='mask_d2')
        att = tf.math.multiply(att, mask_d2, name='att_masked')
    return att


def fully_connected(x, n_outputs, activation_fn=sigmoid, initializer=None, scope: str='fully_connected'):
    """
    Densely connected NN layer op.
    Arguments:
        inputs: `tf.Tensor` or `tf.SparseTensor`. Inputs for operation.
        kernel: `tf.Variable`. Matrix kernel.
        bias: (Optional) `tf.Variable`. Bias to add to outputs.
        activation: (Optional) 1-argument callable. Activation function to apply to
        outputs.
        dtype: (Optional) `tf.DType`. Dtype to cast `inputs` to.
    Returns:
        `tf.Tensor`. Output of dense connection.
    """
    input_shape = x.shape
    if initializer is None:
        initializer = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)
    W_init = initializer((input_shape[-1], n_outputs))
    b_init = initializer((n_outputs,))

    with tf.name_scope(scope):
        W = tf.Variable(W_init, dtype=tf.float32, trainable=True)
        b = tf.Variable(b_init, dtype=tf.float32, trainable=True)
        y = tf.linalg.matmul(x, W)
        y = tf.nn.bias_add(y, b)
        if activation_fn is not None:
            y = activation_fn(y)
    return y


def conv1d(x, n_filters: int, kernel_size: int=3, stride: int=1, padding: str='SAME', activation_fn=relu, initializer=None, scope: str='conv1d'):
    N, W, C_in = x.shape
    if padding.upper() == 'SAME':
        C_out = C_in * stride
    elif padding.upper() == 'VALID':
        C_out = (C_in-1) * stride + kernel_size
    else:
        raise ValueError('padding must be either SAME or VALID')
    if initializer is None:
        initializer = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)
    filter_shape = (n_filters, C_in, C_out)
    filters_init = initializer(filter_shape)
    with tf.name_scope(scope):
        filters = tf.Variable(filters_init, dtype=tf.float32, trainable=True)
        y = tf.nn.conv1d(x, filters, stride, padding, name='conv1d')
        if activation_fn is not None:
            y = activation_fn(y)
    return y


def dropblock2d(inputs, keep_prob: float=1.0, block_size: int=1, 
                scale: bool=True, is_training: bool=True, scope: str='dropblock2d'):
    
    def _create_mask(inputs, block_size, keep_prob):
        B, H, W, C = tf.shape(inputs)
        H_, W_ = to_float(H), to_float(W)
        dropout_rate = 1 - keep_prob
        
        # pad the mask
        p1 = (block_size-1) // 2
        p0 = (block_size-1) - p1
        padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]

        gamma = dropout_rate*(W_*H_) / (block_size**2) / ((W_-block_size+1)*(H_-block_size+1))
        sampling_mask_shape = tf.stack([B, H-block_size+1, W-block_size+1, C], name='sampling_mask_shape')
        mask = _bernoulli(sampling_mask_shape, gamma)
        mask = tf.pad(mask, padding, name='mask_padded')
        mask = tf.nn.max_pool(mask, [1, block_size, block_size, 1], [1, 1, 1, 1], padding='SAME', name='mask_pooled')
        mask = tf.math.subtract(1.0, mask, name='mask')
        return mask

    def _drop(inputs):
        mask = _create_mask(inputs, block_size, keep_prob)
        mask_size = to_float(tf.size(mask))
        inputs_masked = inputs * mask
        inputs_scaled = inputs_masked * mask_size / tf.reduce_sum(mask)
        output = tf.cond(scale, true_fn=lambda: inputs_scaled, false_fn=lambda: inputs_masked)
        return output

    scale = tf.constant(scale, dtype=tf.bool)
    output = tf.cond(tf.logical_or(tf.logical_not(is_training), tf.equal(keep_prob, 1.0)), 
                     true_fn=lambda: inputs, 
                     false_fn=lambda: _drop(inputs))
    return output


if __name__ == "__main__":
    x = tf.keras.Input(shape=(12, 10, 100))
    y = DropBlock2D(0.5, 1, name='DropBlock2D')(x)
    model = tf.keras.Model(inputs=[x], outputs=[y], name='test')
    model.summary()











