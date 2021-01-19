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

sys.path.append(work_dir)
sys.path.append(root_dir)
sys.path.append(model_dir)

import time
import logging

import math
import numpy as np
import tensorflow as tf
try:
    import tensorflow.contrib as tfc
except ModuleNotFoundError:
    pass

from utils import *
from metrics import *
from models.racl.layers_v1 import *


class RACL(object):

    def __init__(self, opt, word_embedding: np.array, domain_embedding: np.array, word_dict: dict, mode: str='train'):
        with tf.name_scope('parameters'):
            self.opt = opt
            self.mode = 'test' if any(m in mode.lower() for m in ['test', 'infer', 'predict']) else 'train'
            self.w2v = word_embedding
            self.w2v_domain = domain_embedding
            self.word_id_mapping = word_dict
            self.initializer = tf.random_uniform_initializer(minval=-0.01, maxval=0.01, seed=self.opt.random_seed)

        if self.mode == 'train':
            # Build logger
            info = ''
            for arg in vars(opt):
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

        # Build model
        inputs_embedding = self.build_input_block()
        with tf.name_scope('RACL'):
            preds = self.build_RACL_block(inputs_embedding)
            self.ae_pred, self.oe_pred, self.sc_pred = preds
        with tf.name_scope('outputs'):
            probs = self.build_output_block(*preds)
            self.ae_prob, self.oe_prob, self.sc_prob = probs

        # Create session
        self.saver = tf.train.Saver(max_to_keep=11)
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.sess = tf.Session()
        if self.mode == 'test':
            self.load_weights(opt.weights_path)

    def build_input_block(self):
        with tf.name_scope('embeddings'):
            is_finetuned = bool(self.opt.finetune_embeddings)
            self.word_embedding = tf.Variable(self.w2v, dtype=tf.float32, name='word_embedding', trainable=is_finetuned)
            self.domain_embedding = tf.Variable(self.w2v_domain, dtype=tf.float32, name='domain_embedding', trainable=is_finetuned)

        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.int32, [None, self.opt.max_sentence_len], name='sentence_x')
            inputs_word = tf.nn.embedding_lookup(self.word_embedding, self.x, name='word_embedding')
            inputs_domain = tf.nn.embedding_lookup(self.domain_embedding, self.x, name='domain_embedding')
            inputs_embedding = tf.concat([inputs_word, inputs_domain], axis=-1, name='embedding_concat')

            self.position_att = tf.placeholder(tf.float32, [None, self.opt.max_sentence_len, self.opt.max_sentence_len], name='position_attention')
            self.word_mask = tf.placeholder(tf.float32, [None, self.opt.max_sentence_len], name='word_mask')
            self.sentiment_mask = tf.placeholder(tf.float32, [None, self.opt.max_sentence_len], name='sentiment_mask')
            self.is_training = tf.placeholder(tf.bool, name='is_training')

        with tf.name_scope('dropout'):
            self.keep_prob1 = tf.placeholder(tf.float32, name='keep_prob_1')
            self.keep_prob2 = tf.placeholder(tf.float32, name='keep_prob_2')
            self.DropBlock_aspect = DropBlock2D(keep_prob=self.keep_prob2, block_size=3)
            self.DropBlock_opinion = DropBlock2D(keep_prob=self.keep_prob2, block_size=3)
            self.DropBlock_sentiment = DropBlock2D(keep_prob=self.keep_prob2, block_size=3)
        
        return inputs_embedding

    def build_RACL_block(self, inputs_embedding):
        inputs_embedding = tf.nn.dropout(inputs_embedding, keep_prob=self.keep_prob1, name='inputs_dropout')

        # Shared Feature
        shared_input = tf.layers.conv1d(inputs_embedding, self.opt.emb_dim, 1, padding='SAME', activation=tf.nn.relu, name='shared_features')
        # shared_input = tfc.layers.batch_norm(shared_input, scale=True, center=True, is_training=self.is_training, trainable=True, scope='shared_features_batchnormed')
        shared_input = tf.nn.dropout(shared_input, keep_prob=self.keep_prob1)

        # mask = tf.tile(tf.expand_dims(self.word_mask, axis=-1), [1, 1, self.opt.n_filters])
        mask = tf.tile(tf.expand_dims(self.word_mask, axis=1), [1, self.opt.max_sentence_len, 1])

        # Private Feature
        aspect_inputs, opinion_inputs, context_inputs = [shared_input], [shared_input], [shared_input]
        aspect_preds, opinion_preds, sentiment_preds = [], [], []

        # We found that the SC task is more difficult than the AE and OE tasks.
        # Hence, we augment it with a memory-like mechanism by updating the aspect query with the retrieved contexts.
        # For more details about the memory network, refer to 
        #       https://www.aclweb.org/anthology/D16-1021/ .
        context_queries = [shared_input]
        dropblock_layers = [self.DropBlock_aspect, self.DropBlock_opinion, self.DropBlock_sentiment]
        for interact in range(self.opt.n_interactions):
            with tf.variable_scope(f'interaction_{interact}', reuse=tf.AUTO_REUSE):
                RACL_inputs = [
                    aspect_inputs[-1], opinion_inputs[-1], context_inputs[-1], context_queries[-1], 
                    self.word_mask, mask, self.position_att, self.opt, self.initializer, 
                    dropblock_layers, self.is_training
                ]
                preds, interactions = RACL_block(*RACL_inputs)
                aspect_pred, opinion_pred, sentiment_pred = preds
                aspect_inter, opinion_inter, context_inter, context_conv = interactions
                
                # Stacking
                aspect_preds.append(tf.expand_dims(aspect_pred, axis=-1))
                opinion_preds.append(tf.expand_dims(opinion_pred, axis=-1))
                sentiment_preds.append(tf.expand_dims(sentiment_pred, axis=-1))

                aspect_inputs.append(aspect_inter)
                opinion_inputs.append(opinion_inter)
                context_inputs.append(context_conv)
                context_queries.append(context_inter) # update query

        # Multi-layer Short-cut
        ae_pred = tf.reduce_mean(tf.concat(aspect_preds, axis=-1), axis=-1, name='ae_pred')
        oe_pred = tf.reduce_mean(tf.concat(opinion_preds, axis=-1), axis=-1, name='oe_pred')
        sc_pred = tf.reduce_mean(tf.concat(sentiment_preds, axis=-1), axis=-1, name='sc_pred')

        return ae_pred, oe_pred, sc_pred

    def build_output_block(self, ae_pred, oe_pred, sc_pred):
        
        # Scale probability
        ae_prob = tf.nn.softmax(ae_pred, axis=-1, name='ae_prob')
        oe_prob = tf.nn.softmax(oe_pred, axis=-1, name='oe_prob')
        sc_prob = tf.nn.softmax(sc_pred, axis=-1, name='sc_prob')
        return ae_prob, oe_prob, sc_prob

    def build_loss_block(self):

        aspect_pred, opinion_pred, sentiment_pred = self.ae_pred, self.oe_pred, self.sc_pred
        aspect_prob, opinion_prob, _ = self.build_output_block(aspect_pred, opinion_pred, sentiment_pred)

        output_shape = [-1, self.opt.n_classes]
        mask_shape = [1, 1, self.opt.n_classes]

        with tf.name_scope('predictions'):
            # Mask AE & OE Probabilities
            word_mask = tf.tile(tf.expand_dims(self.word_mask, axis=-1), mask_shape)
            aspect_pred = tf.reshape(word_mask*aspect_pred, output_shape, name='aspect_pred_masked')
            opinion_pred = tf.reshape(word_mask*opinion_pred, output_shape, name='opinion_pred_masked')

            # Relation R4 (only in Training)
            # In training / validation, sentiment masks are set to 1.0 only for aspect terms.
            # In testing, sentiment masks are set to 1.0 for all words (except padding 1s).

            # Mask SC Probabilities
            sentiment_mask = tf.tile(tf.expand_dims(self.sentiment_mask, axis=-1), mask_shape)
            sentiment_mask = tf.cast(sentiment_mask, tf.float32)
            sentiment_pred = tf.reshape(sentiment_mask*sentiment_pred, output_shape, name='sentiment_pred_masked')

        with tf.name_scope('labels'):
            input_shape = [None, self.opt.max_sentence_len, self.opt.n_classes]
            self.aspect_y = tf.placeholder(tf.int32, input_shape, name='aspect_y')
            self.opinion_y = tf.placeholder(tf.int32, input_shape, name='opinion_y')
            self.sentiment_y = tf.placeholder(tf.int32, input_shape, name='sentiment_y')

            # Format ground-truths
            self.aspect_label = tf.cast(self.aspect_y, tf.float32)
            self.opinion_label = tf.cast(self.opinion_y, tf.float32)
            self.sentiment_label = tf.cast(self.sentiment_y, tf.float32)
            aspect_label = tf.reshape(self.aspect_label, output_shape, name='aspect_label')
            opinion_label = tf.reshape(self.opinion_label, output_shape, name='opinion_label')
            sentiment_label = tf.reshape(self.sentiment_label, output_shape, name='sentiment_label')

        with tf.name_scope('losses'):
            # AE & OE Regularization cost - only get Beginning [1] and Inside [2] values
            ae_cost = tf.reduce_sum(aspect_prob[:,:,1:], axis=-1, name='ae_cost')
            oe_cost = tf.reduce_sum(opinion_prob[:,:,1:], axis=-1, name='oe_cost')
            total_cost = ae_cost + oe_cost - 1.
            total_cost = tf.maximum(0., total_cost, name='total_cost')
            reg_cost = tf.reduce_sum(total_cost) / tf.reduce_sum(self.word_mask)
            reg_cost = tf.identity(reg_cost, name='regularization_cost')

            # Calculate costs
            aspect_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=aspect_pred, labels=aspect_label), name='aspect_loss')
            opinion_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=opinion_pred, labels=opinion_label), name='opinion_loss')
            sentiment_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=sentiment_pred, labels=sentiment_label), name='sentiment_loss')

            loss = self.opt.aspect_weight * aspect_loss + \
                    self.opt.opinion_weight * opinion_loss + \
                    self.opt.sentiment_weight * sentiment_loss + \
                    self.opt.regularization_weight * reg_cost
            loss = tf.identity(loss, name='overall_loss')
            losses = [loss, aspect_loss, opinion_loss, sentiment_loss, reg_cost]
        return losses
        
    def train(self, is_tuning: bool=False):
        batch_size = tf.shape(self.x)[0]
        output_shape = [batch_size, self.opt.max_sentence_len, self.opt.n_classes]

        with tf.name_scope('losses'):
            self.losses = self.build_loss_block()

        with tf.name_scope('train'):
            trainvars = tf.trainable_variables()
            total_params = count_parameter(trainvars)
            self.logger.info('>>> total parameter: {}'.format(total_params))

            global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.opt.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.opt.learning_rate)
            elif self.opt.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.opt.learning_rate)
            else:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.opt.learning_rate)
            train_operator = optimizer.minimize(self.losses[0], global_step=global_step)

        with tf.name_scope('predictions'):
            aspect_pred = tf.reshape(self.ae_pred, output_shape, name='aspect_pred')
            opinion_pred = tf.reshape(self.oe_pred, output_shape, name='opinion_pred')
            sentiment_pred = tf.reshape(self.sc_pred, output_shape, name='sentiment_pred')

        with tf.name_scope('ground-truths'):
            aspect_true = tf.reshape(self.aspect_label, output_shape, name='aspect_label')
            opinion_true = tf.reshape(self.opinion_label, output_shape, name='opinion_label')
            sentiment_true = tf.reshape(self.sentiment_label, output_shape, name='sentiment_label')
        
        # Read datasets
        train_set = read_data(self.opt.train_path, self.word_id_mapping, self.opt.max_sentence_len)
        val_set = read_data(self.opt.val_path, self.word_id_mapping, self.opt.max_sentence_len, is_testing=True)
        test_set = read_data(self.opt.test_path, self.word_id_mapping, self.opt.max_sentence_len, is_testing=True)
        n_trains, n_vals, n_tests = len(train_set), len(val_set), len(test_set)

        # Load pretrained weights or Initialize new weights
        pretrained_model_loaded = False
        if self.opt.load_pretrained and not is_tuning:
            pretrained_model_loaded = self.load_weights()
            
        if not pretrained_model_loaded:
            glovars = tf.global_variables_initializer()
            self.sess.run(glovars)
        
        # Save model architecture
        writer = tf.summary.FileWriter(logdir=self.board_dir, graph=self.sess.graph)
        writer.flush()

        # Training Process
        train_ops = self.losses + [global_step, train_operator]
        test_ops = [aspect_true, aspect_pred, opinion_true, opinion_pred, sentiment_true, sentiment_pred, self.word_mask]
        val_ops = self.losses + test_ops
        val_metrics, val_losses = [], []
        for ep in range(self.opt.n_epochs):
            self.logger.info('\n'+'-'*69)
            self.logger.info(f'Epoch {ep:03d} / {self.opt.n_epochs:03d}')

            # Train
            tr_loss, tr_a_loss, tr_o_loss, tr_s_loss, tr_r_loss = 0., 0., 0., 0., 0.
            epoch_start = time.time()
            for train_batch, n_samples in self.get_batch_data(train_set, self.opt.batch_size, self.opt.kp1, self.opt.kp2, True, is_shuffle=True):
                e_loss, ae_loss, oe_loss, sc_loss, reg_loss, step, _ = self.sess.run(
                    train_ops, feed_dict=train_batch
                )
                tr_loss += e_loss * n_samples
                tr_a_loss += ae_loss * n_samples
                tr_o_loss += oe_loss * n_samples
                tr_s_loss += sc_loss * n_samples
                tr_r_loss += reg_loss * n_samples
            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start
            epoch_time = 'Epoch Time: {:.0f}m {:.0f}s'.format(epoch_time//60, epoch_time%60)
            # print(epoch_time)
            self.logger.info('\t'*2+epoch_time)
            self.logger.info('Train:\nloss={:.3f}, aspect_loss={:.3f}, opinion_loss={:.3f}, sentiment_loss={:.3f}, reg_loss={:.3f}, step={}'.
                format(tr_loss/n_trains, tr_a_loss/n_trains, tr_o_loss/n_trains, tr_s_loss/n_trains, tr_r_loss/n_trains, step))

            if ep < self.opt.warmup_epochs:
                val_metrics.append(-1.0)
                val_losses.append(1e11)
                continue

            # Validate
            val_loss, val_a_loss, val_o_loss, val_s_loss, val_r_loss = 0., 0., 0., 0., 0.
            val_a_preds, val_a_labels = [], []
            val_o_preds, val_o_labels = [], []
            val_s_preds, val_s_labels = [], []
            val_mask = []
            for val_batch, n_samples_ in self.get_batch_data(val_set):
                e_loss_, ae_loss_, oe_loss_, sc_loss_, reg_loss_, \
                ae_true, ae_pred, oe_true, oe_pred, sc_true, sc_pred, mask_ = self.sess.run(
                    val_ops, feed_dict=val_batch
                )
                val_loss += e_loss_ * n_samples_
                val_a_loss += ae_loss_ * n_samples_
                val_o_loss += oe_loss_ * n_samples_
                val_s_loss += sc_loss_ * n_samples_
                val_r_loss += reg_loss_ * n_samples_

                # concatenate ground-truths and predictions for evaluation in next step
                val_a_preds.extend(ae_pred)
                val_o_preds.extend(oe_pred)
                val_s_preds.extend(sc_pred)
                val_a_labels.extend(ae_true)
                val_o_labels.extend(oe_true)
                val_s_labels.extend(sc_true)
                val_mask.extend(mask_)

            val_losses.append(val_loss)
            
            # Evaluation
            val_scores = evaluate_absa(val_a_labels, val_a_preds, val_o_labels, val_o_preds, val_s_labels, val_s_preds, val_mask)
            val_opinion_f1, val_aspect_f1, val_sentiment_acc, val_sentiment_f1, val_ABSA_f1 = val_scores
            val_metrics.append(val_ABSA_f1)

            self.logger.info('Validation:\nloss={:.3f}, aspect_loss={:.3f}, opinion_loss={:.3f}, sentiment_loss={:.3f}, reg_loss={:.3f}'.
                format(val_loss/n_vals, val_a_loss/n_vals, val_o_loss/n_vals, val_s_loss/n_vals, val_r_loss/n_vals))
            self.logger.info('aspect_f1={:.3f}, opinion_f1={:.3f}, sentiment_acc={:.3f}, sentiment_f1={:.3f}, ABSA_f1={:.3f}'.
                format(val_aspect_f1, val_opinion_f1, val_sentiment_acc, val_sentiment_f1, val_ABSA_f1))
            self.logger.info('Best Metrics at Epoch: {} - Best Loss at Epoch: {}'.
                format(np.argmax(val_metrics), np.argmin(val_losses)))

            # Only save best model(s)
            if ep >= self.opt.warmup_epochs and not is_tuning:
                best_epochs = [np.argmax(val_metrics), np.argmin(val_losses)]
                if ep in best_epochs:
                    self.saver.save(self.sess, os.path.join(self.ckpt_dir, 'RACL.ckpt'), global_step=ep)

        # Testing Process
        self.logger.info('\n{:-^69}'.format('Training Completed'))
        test_a_preds, test_a_labels = [], []
        test_o_preds, test_o_labels = [], []
        test_s_preds, test_s_labels = [], []
        test_mask = []
        for test_batch, _ in self.get_batch_data(test_set):
            ae_true, ae_pred, oe_true, oe_pred, sc_true, sc_pred, mask_ = self.sess.run(
                test_ops, feed_dict=test_batch
            )

            # concatenate ground-truths and predictions for evaluation in next step
            test_a_preds.extend(ae_pred)
            test_o_preds.extend(oe_pred)
            test_s_preds.extend(sc_pred)
            test_a_labels.extend(ae_true)
            test_o_labels.extend(oe_true)
            test_s_labels.extend(sc_true)
            test_mask.extend(mask_)

        # Evaluation
        test_scores = evaluate_absa(test_a_labels, test_a_preds, test_o_labels, test_o_preds, test_s_labels, test_s_preds, test_mask)
        test_opinion_f1, test_aspect_f1, test_sentiment_acc, test_sentiment_f1, test_ABSA_f1 = test_scores
        self.logger.info('Test:\naspect_f1={:.3f}, opinion_f1={:.3f}, sentiment_acc={:.3f}, sentiment_f1={:.3f}, ABSA_f1={:.3f}'.
            format(test_aspect_f1, test_opinion_f1, test_sentiment_acc, test_sentiment_f1, test_ABSA_f1))

        # Update global and domain-specific word embeddings
        # self.w2v = self.word_embedding.eval(session=self.sess)
        # self.w2v_domain = self.domain_embedding.eval(session=self.sess)

        if is_tuning:
            return test_opinion_f1, test_aspect_f1, test_sentiment_f1, test_ABSA_f1

    def predict(self, sentences, words_mask, position_matrices):
        # batch_size = sentences.shape[0]
        # mask_shape = [1, 1, self.opt.n_classes]
        # output_shape = [batch_size, -1, self.opt.n_classes]
        aspect_pred, opinion_pred, sentiment_pred = self.sess.run(
            [self.aspect_value, self.opinion_value, self.sentiment_value], 
            feed_dict={
                self.x: sentences,
                self.position_att: position_matrices,
                self.word_mask: words_mask,
                self.sentiment_mask: words_mask,
                self.keep_prob1: 1.0,
                self.keep_prob2: 1.0,
                self.is_training: False,
            }
        )
        return aspect_pred, opinion_pred, sentiment_pred

    def load_weights(self, weights_path=None):
        try:
            if not weights_path:
                ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
                weights_path = ckpt.model_checkpoint_path
            print(f'Load model from {weights_path}')
            self.saver.restore(self.sess, weights_path)
            return True
        except Exception as e:
            print(e)
            return False

    def get_batch_data(self, dataset, batch_size: int=16, 
                       keep_prob1: float=1.0, keep_prob2: float=1.0, 
                       is_training: bool=False, is_shuffle: bool=False):
        length = len(dataset[0])
        all_index = np.arange(length)
        if is_shuffle:
            np.random.shuffle(all_index)
        n_batches = int(length/batch_size) + (1 if length % batch_size else 0)
        for i in range(n_batches):
            indices = all_index[i*batch_size:(i+1)*batch_size]
            batch_feed = {
                self.x: dataset[0][indices],
                self.aspect_y: dataset[1][indices],
                self.opinion_y: dataset[2][indices],
                self.sentiment_y: dataset[3][indices],
                self.word_mask: dataset[4][indices],
                self.sentiment_mask: dataset[5][indices],
                self.position_att: dataset[6][indices],
                self.keep_prob1: keep_prob1,
                self.keep_prob2: keep_prob2,
                self.is_training: is_training,
            }
            yield batch_feed, len(indices)

            


























