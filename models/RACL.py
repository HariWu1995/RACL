"""
RACL - Relation-Aware Collaborative Learning
            for Unified Aspect-based Sentiment Analysis

    # Sentence Representation
        A. Word Embeddings - Pretrained GloVe (word-level and ngram-level)

        B. Features:
            0. Sharing Feature: Dropout -> CNN -> Dropout

            1. Aspect Extraction [AE]: CNN

            2. Opinion Extraction [OE]: CNN

            3. Sentiment Classification [SC]: CNN

        C. Relations:
            1. R1 - relation between AE and OE: Attention

            2. R2 - relation between SC and R1: Attention

            3. R3 - relation between SC and OE: Attention

            4. R4 - relation between SC and AE: Attention

            5. Opinion Propagation: SoftMax -> Clip
    
    # Predictions + Loss Function
        A. Aspect: Fully-Connected -> SoftMax -> Cross-Entropy

        B. Opinion: Fully-Connected -> SoftMax -> Cross-Entropy

        C. Sentiment: Fully-Connected -> SoftMax -> Cross-Entropy

        D. Loss = Weighted_Sum(Aspect_Loss, Opinion_Loss, Sentiment_Loss, Regularization_Loss)
"""
import os
import sys

work_dir = os.getcwd()
root_dir = os.path.dirname(work_dir)
model_dir = os.path.join(root_dir, 'src', 'models')
ckpts_dir = os.path.join(root_dir, 'src', 'checkpoints')
boards_dir = os.path.join(root_dir, 'src', 'tensorboards')
logs_dir = os.path.join(root_dir, 'src', 'logs')
viz_dir = os.path.join(root_dir, 'src', 'training_histories')

sys.path.append(work_dir)
sys.path.append(root_dir)
sys.path.append(model_dir)

import time
import logging
from tqdm import tqdm as print_progress

import math
import numpy as np
import tensorflow as tf
if not tf.executing_eagerly():
    tf.enable_eager_execution()

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, InputLayer, Embedding, 
    Conv1D, Conv2D, Dropout, Dense, 
    Dot, Concatenate, Average, Add, Multiply,
    Lambda, Reshape, 
    Softmax, Maximum, Minimum,
)
try:
    from tensorflow.keras.activations import softmax, sigmoid
    from tensorflow.keras.initializers import Identity, GlorotNormal, GlorotUniform
    from tensorflow.keras.optimizers import Adam, Nadam, Adagrad, Adadelta, RMSprop, SGD
    from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, LearningRateScheduler, EarlyStopping, ModelCheckpoint
    from tensorflow.keras.losses import CategoricalCrossentropy
    from tensorflow.keras.utils import plot_model
except ImportError as e:
    pass

from utils import *
from losses import *
from metrics import *
from lr_schedulers import *
from models.racl.layers import *


class RACL(object):

    def __init__(self, opt):
        self.opt = opt
        self.mode = 'train' if self.opt.is_training else 'predict'
        if opt.random_type == 'uniform':
            self.initializer = GlorotUniform(seed=self.opt.random_seed)
        else:
            self.initializer = GlorotNormal(seed=self.opt.random_seed)

        if self.opt.is_training:
            # Build logger
            info = ''
            for arg in vars(opt):
                if arg in ['word_dict', 'w2v', 'w2v_domain']:
                    continue
                info += ('>>> {0}: {1}\n'.format(arg, getattr(opt, arg)))

            log_dir = os.path.join(logs_dir, self.opt.task)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            filename = os.path.join(log_dir, f'{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.txt')
            self.logger = logging.getLogger(filename)
            self.logger.setLevel(logging.DEBUG)
            # self.logger.propagate = False

            self.logger.addHandler(logging.StreamHandler())
            self.logger.addHandler(logging.FileHandler(filename, 'a'))

            self.logger.info('{:-^80}\n{}\n'.format('Parameters', info))

        # Build checkpoint & tensorboard
        self.ckpt_dir = os.path.join(ckpts_dir, self.opt.task)
        self.board_dir = os.path.join(boards_dir, self.opt.task)
        self.viz_dir = os.path.join(viz_dir, self.opt.task)

        # Build model
        inputs, embeddings, position_att, word_mask, sentiment_mask = self.build_input_block()
        predictions = list(self.build_RACL_block(embeddings, position_att, word_mask))
        
        if self.opt.is_training:
            model_inputs = [inputs, word_mask, sentiment_mask, position_att]
            model_outputs = predictions + [word_mask, sentiment_mask]
            self.model = RACLModel(inputs=model_inputs, outputs=model_outputs, name='RACL')
        else:
            predictions_as_prob = self.build_output_block(predictions)
            self.model = RACLModel(inputs=[inputs, word_mask, position_att], 
                                   outputs=predictions_as_prob, name='RACL')
            self.load_weights(opt.weights_path)

        # self.model.summary()
        # self.visualize_architecture()
        model_summary = log_summary(self.model)
        self.logger.info(model_summary)

    def visualize_architecture(self):
        plot_model(self.model, to_file=f'{self.opt.model}_{self.mode}.png', dpi=128, show_shapes=True, show_layer_names=True)

    def build_input_block(self):
        # Inputs for Word in Global and Domain
        if self.opt.feed_text_embeddings:
            inputs, embeddings = dropoutize_embeddings(self.opt)
            inputs._name = 'embeddings_concat'
        else:
            inputs = Input(shape=(self.opt.max_sentence_len,), name='sentence_input')
            word_embeddings = create_embeddings(inputs, self.opt, self.opt.global_dim, 'word', self.opt.w2v)
            domain_embeddings = create_embeddings(inputs, self.opt, self.opt.domain_dim, 'domain', self.opt.w2v_domain)
            embeddings = Concatenate(axis=-1, name='embeddings_concat')([word_embeddings, domain_embeddings])

        # Inputs for Masking
        position_att = Input(shape=(self.opt.max_sentence_len, self.opt.max_sentence_len), name='position_att')
        word_mask = Input(shape=(self.opt.max_sentence_len,), name='word_mask')
        sentiment_mask = Input(shape=(self.opt.max_sentence_len,), name='sentiment_mask')
        return inputs, embeddings, position_att, word_mask, sentiment_mask
        
    def build_RACL_block(self, embeddings, position_att, word_mask):
        # Preprocessing
        inputs = Dropout(rate=1-self.opt.keep_prob_1, name='inputs_dropout')(embeddings)

        # Shared Features
        conv_args = {'kernel_size': 1, 'strides': 1, 'padding': 'same', 'activation': 'relu', }
        Feature_Extractor = Conv1D(filters=self.opt.emb_dim, name='shared_features', **conv_args)
        shared_features = Feature_Extractor(inputs)
        shared_features = Dropout(rate=1-self.opt.keep_prob_1, name='shared_features_dropout')(shared_features)

        # Define repeatable layers in RACL interactions
        DropBlock_aspect = DropBlock2D(keep_prob=self.opt.keep_prob_2, block_size=3, name='DropBlock2D_aspect')
        DropBlock_opinion = DropBlock2D(keep_prob=self.opt.keep_prob_2, block_size=3, name='DropBlock2D_opinion')
        DropBlock_context = DropBlock2D(keep_prob=self.opt.keep_prob_2, block_size=3, name='DropBlock2D_context')

        L2Normalize = L2Norm()
        Tile = Lambda(lambda x: tf.tile(tf.expand_dims(x, axis=1), [1, self.opt.max_sentence_len, 1]), name='Tiler-in-RACL')

        # We found that the SC task is more difficult than the AE and OE tasks.
        # Hence, we augment it with a memory-like mechanism by updating the aspect query with the retrieved contexts.
        # For more details about the memory network, refer to 
        #       https://www.aclweb.org/anthology/D16-1021/ .
        aspect_inputs, opinion_inputs, context_inputs = [shared_features], [shared_features], [shared_features]
        aspect_preds, opinion_preds, sentiment_preds = [], [], []
        context_queries = [shared_features]
        
        conv_args['kernel_size'] = self.opt.kernel_size
        dense_args = {'units': self.opt.n_classes, 'kernel_initializer': self.initializer}

        for interact_i in range(self.opt.n_interactions):            
            racl_block = RACL_Block(self.opt, L2Normalize, [DropBlock_aspect, DropBlock_opinion, DropBlock_context], Tile,
                                    conv_args, dense_args, block_id=interact_i)
            output_preds, output_interacts = racl_block([aspect_inputs[-1], opinion_inputs[-1], context_inputs[-1], context_queries[-1], word_mask, position_att])
            aspect_pred, opinion_pred, sentiment_pred = output_preds
            aspect_interact, opinion_interact, context_interact, context_conv = output_interacts

            # Stacking
            aspect_preds.append(ExpandDim(axis=-1, name=f'aspect_pred-{interact_i}')(aspect_pred))
            opinion_preds.append(ExpandDim(axis=-1, name=f'opinion_pred-{interact_i}')(opinion_pred))
            sentiment_preds.append(ExpandDim(axis=-1, name=f'sentiment_pred-{interact_i}')(sentiment_pred))

            aspect_inputs.append(aspect_interact)
            opinion_inputs.append(opinion_interact)
            context_inputs.append(context_conv)
            context_queries.append(context_interact) # update query

        # Multi-layer Short-cut
        aspect_preds = Concatenate(axis=-1, name='aspect_preds')(aspect_preds)
        opinion_preds = Concatenate(axis=-1, name='opinion_preds')(opinion_preds)
        sentiment_preds = Concatenate(axis=-1, name='sentiment_preds')(sentiment_preds)
        aspect_pred = ReduceDim('mean', axis=-1, name='AE_pred')(aspect_preds)
        opinion_pred = ReduceDim('mean', axis=-1, name='OE_pred')(opinion_preds)
        sentiment_pred = ReduceDim('mean', axis=-1, name='SC_pred')(sentiment_preds)
        return aspect_pred, opinion_pred, sentiment_pred

    def build_output_block(self, preds):
        aspect_pred, opinion_pred, sentiment_pred = preds

        # Scale probability
        aspect_prob = Softmax(axis=-1, name='aspect_prob')(aspect_pred)
        opinion_prob = Softmax(axis=-1, name='opinion_prob')(opinion_pred)
        sentiment_prob = Softmax(axis=-1, name='sentiment_prob')(sentiment_pred)
        return aspect_prob, opinion_prob, sentiment_prob

    def train(self, is_tuning: bool=False):
        from data_generator import DataGenerator

        optimizer_option = self.opt.optimizer.lower()
        if optimizer_option == 'adam':
            from tensorflow.keras.optimizers import Adam as Optimizer
        elif optimizer_option == 'nadam':
            from tensorflow.keras.optimizers import Nadam as Optimizer
        elif optimizer_option == 'adagrad':
            from tensorflow.keras.optimizers import Adagrad as Optimizer
        elif optimizer_option == 'adadelta':
            from tensorflow.keras.optimizers import Adadelta as Optimizer
        elif optimizer_option == 'rmsprop':
            from tensorflow.keras.optimizers import RMSprop as Optimizer
        else:
            from tensorflow.keras.optimizers import SGD as Optimizer

        # Load generators
        train_set = DataGenerator(self.opt.train_path, self.opt, is_shuffle=True)
        val_set = DataGenerator(self.opt.val_path, self.opt, is_shuffle=False, load_all=True)
        test_set = DataGenerator(self.opt.test_path, self.opt, is_shuffle=False, load_all=True)
        n_trains, n_vals, n_tests = len(train_set), len(val_set), len(test_set)

        # Training Process
        self.model.set_opt(self.opt)
        self.model.compile(
            # loss=RACL_losses_wrapper(self.opt),
            # metrics=ABSA_scores(self.opt, include_opinion=self.opt.include_opinion, dtype=float),
            optimizer=Optimizer(learning_rate=self.opt.lr),
        )
        ABSA_Callback = ABSA_scores(val_set[0], self.logger, include_opinion=self.opt.include_opinion)
        training_history = self.model.fit(
            train_set, steps_per_epoch=n_trains, 
            validation_data=val_set, validation_steps=n_vals,
            epochs=self.opt.n_epochs, verbose=True,
            callbacks=[
                # ReduceLROnPlateau(monitor='val_loss', factor=0.69, patience=5, min_lr=1e-7, verbose=1),
                CyclicLR(mode='exponential'),
                TensorBoard(self.board_dir),
                ModelCheckpoint(os.path.join(self.ckpt_dir, 'RACL-{epoch:03d}.h5'), save_weights_only=True, monitor='val_loss', verbose=1),
                ABSA_Callback,
                EarlyStopping(monitor="val_loss", patience=11, restore_best_weights=True, verbose=1)
            ]
        )
        history_fig = plot_history(training_history)
        history_fig.savefig(os.path.join(self.viz_dir, 'training_history.png'))
        self.model.save_weights(os.path.join(self.ckpt_dir, 'RACL-best-loss.h5'))
        self.model.load_weights(os.path.join(self.ckpt_dir, f'RACL-{ABSA_Callback.max_dev_index:03d}.h5'))
        self.model.save_weights(os.path.join(self.ckpt_dir, 'RACL-best-score.h5'))

        # Testing Process
        Xs, Ys_true = test_set[0]
        *Ys_pred, word_mask, _ = self.model.predict(Xs, batch_size=self.opt.batch_size)
        scores = evaluate_absa(Ys_true[0], Ys_pred[0],
                               Ys_true[1], Ys_pred[1],
                               Ys_true[2], Ys_pred[2],
                               word_mask, include_opinion=self.opt.include_opinion)
        display_text = f'\n\nTesting\n\topinion_f1={scores[0]:.7f}, aspect_f1={scores[1]:.7f}, sentiment_acc={scores[2]:.7f}, sentiment_f1={scores[3]:.7f}, ABSA_f1={scores[4]:.7f}'
        self.logger.info(display_text)

        if is_tuning:
            # Update global and domain-specific word embeddings
            self.w2v = self.model.get_layer('word_embeddings').get_weights()
            self.w2v_domain = self.model.get_layer('domain_embeddings').get_weights()
            return scores

    def predict(self, sentence, word_mask, position_att):
        ae_pred, oe_pred, sc_pred = self.model.predict([sentence, word_mask, position_att])
        return ae_pred, oe_pred, sc_pred

    def load_weights(self, weights_path):
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"weights_path:\n\t{weights_path}\ndoesn't exist!")
        try:
            self.model.load_weights(weights_path)
        except Exception as e:
            print(e)


class RACLModel(Model):

    def set_opt(self, opt):
        self.opt = opt

    def train_step(self, data):
        Xs, Ys_true = data

        with tf.GradientTape() as tape:
            # Forward pass
            *Ys_pred, word_mask, sentiment_mask = self(Xs, training=True)  

            # Compute the loss value. Loss function is configured in `compile()`.
            losses = RACL_losses(Ys_true, Ys_pred, [word_mask, sentiment_mask], self.opt)

        # Backward progagation - Compute gradients & Update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(losses[0], trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {
            'OE_loss': losses[1], 'AE_loss': losses[2], 'SC_loss': losses[3], 
            'Reg_cost': losses[4], 'loss': losses[0], 
        }

    def test_step(self, data):
        # Unpack the data
        Xs, Ys_true = data

        # Compute predictions
        *Ys_pred, word_mask, sentiment_mask = self(Xs, training=False)

        # Compute the loss value. Loss function is configured in `compile()`.
        losses = RACL_losses(Ys_true, Ys_pred, [word_mask, sentiment_mask], self.opt)
        return {
            'OE_loss': losses[1], 'AE_loss': losses[2], 'SC_loss': losses[3], 
            'Reg_cost': losses[4], 'loss': losses[0], 
        }


            



















