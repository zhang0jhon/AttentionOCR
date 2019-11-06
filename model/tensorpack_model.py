# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorpack import ModelDesc
from tensorpack.models import GlobalAvgPooling, l2_regularizer, regularize_cost
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary

from model.model import *

import sys
sys.path.append("..")

import config as cfg


def label_smoothing(inputs, epsilon=0.05):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.
    Args:
      inputs: A 2d tensor with shape of [N, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.
    '''
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)


class AttentionOCR(ModelDesc):
    """
    Attention based method for arbitrary-shaped text recognition.
    """
    def inputs(self):
        return [tf.TensorSpec([None, cfg.image_size, cfg.image_size, 3], tf.float32, 'image'),
                tf.TensorSpec([None, cfg.seq_len+1], tf.int32, 'label'),
                tf.TensorSpec([None, cfg.seq_len+1], tf.float32, 'mask'),
                tf.TensorSpec([None, 4], tf.float32, 'normalized_bbox'),
                tf.TensorSpec([], tf.bool, 'is_training'),
                tf.TensorSpec([], tf.float32, 'dropout_keep_prob')]

    def optimizer(self):
        global_step = tf.train.get_or_create_global_step()
        lr = tf.get_variable('learning_rate', initializer=cfg.learning_rate/100, trainable=False)
        lr = tf.train.cosine_decay(lr, global_step, cfg.num_epochs*cfg.steps_per_epoch, alpha=cfg.min_lr)
        tf.compat.v1.summary.scalar('learning_rate', lr)
        # add_moving_summary(lr)
        # opt = tf.train.MomentumOptimizer(lr, 0.9)
        opt = tf.train.AdamOptimizer(lr, 0.9, 0.999)
        return opt

    def get_inferene_tensor_names(self):
        inputs, outputs = [],[]
        if cfg.model_name == 'ocr':
            inputs, outputs = ['image', 'label', 'is_training', 'dropout_keep_prob'], ['sequence_preds', 'sequence_probs']
        elif cfg.model_name == 'ocr_with_normalized_bbox':
            inputs, outputs = ['image', 'label', 'normalized_bbox', 'is_training', 'dropout_keep_prob'], ['sequence_preds', 'sequence_probs']
        return inputs, outputs

    def build_graph(self, image, label, mask, normalized_bbox, is_training, dropout_keep_prob):
        if cfg.model_name == 'ocr':
            outputs, attentions = inception_padding_model(image, label, \
                wemb_size=cfg.wemb_size, seq_len=cfg.seq_len+1, num_classes=cfg.num_classes, \
                lstm_size=cfg.lstm_size, is_training=is_training, dropout_keep_prob=dropout_keep_prob, \
                weight_decay=cfg.weight_decay, name=cfg.name_scope, reuse=None)

        elif cfg.model_name == 'ocr_with_normalized_bbox':
            outputs, attentions = inception_model(image, label, normalized_bbox, \
                wemb_size=cfg.wemb_size, seq_len=cfg.seq_len+1, num_classes=cfg.num_classes, \
                lstm_size=cfg.lstm_size, is_training=is_training, dropout_keep_prob=dropout_keep_prob, \
                weight_decay=cfg.weight_decay, name=cfg.name_scope, reuse=None)

        def _step_loss(k, total_xen_loss):
            # cross_entropy_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(label[:,k], outputs[:,k,:])
            label_smoothed = label_smoothing(tf.one_hot(label[:,k], cfg.num_classes, axis=-1))
            cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs[:,k,:], labels=label_smoothed)
            cross_entropy_loss *= mask[:,k]
            return k+1, total_xen_loss + cross_entropy_loss

        _, cross_entropy_loss = tf.while_loop(
            cond=lambda k, *_: k < cfg.seq_len+1,
            body=_step_loss,
            loop_vars=(tf.constant(0, tf.int32), tf.constant(np.zeros(cfg.batch_size), tf.float32))
        )

        cross_entropy_loss = tf.reduce_sum(cross_entropy_loss) / tf.reduce_sum(mask)
        reg_loss = tf.add_n(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))

        total_loss = reg_loss + cross_entropy_loss

        # tensorboard summary
        tf.compat.v1.summary.image('input_image', image[0:1,:,:,:])
        for a in range(cfg.seq_len+1):
            tf.compat.v1.summary.image('attention_%d' % a, attentions[0:1,:,:,a:a+1])
        tf.compat.v1.summary.scalar('cross_entropy_loss', cross_entropy_loss)
        tf.compat.v1.summary.scalar('reg_loss', reg_loss)
        tf.compat.v1.summary.scalar('total_loss', total_loss)

        tf.compat.v1.summary.tensor_summary('mask', mask[0,:], summary_description='mask')
        tf.compat.v1.summary.text('mask', tf.as_string(mask[0,:]))
        tf.compat.v1.summary.tensor_summary('label', label[0,:], summary_description='label')
        tf.compat.v1.summary.text('label', tf.as_string(label[0,:]))

        logits = tf.nn.softmax(outputs, name='logits')
        preds = tf.argmax(logits, axis=-1, name='sequence_preds')
        probs = tf.reduce_max(logits, axis=-1, name='sequence_probs')
        tf.compat.v1.summary.tensor_summary('pred', preds[0,:], summary_description='preds')
        tf.compat.v1.summary.text('pred', tf.as_string(preds[0,:]))

        tf.compat.v1.summary.tensor_summary('prob', probs[0,:], summary_description='probs')
        tf.compat.v1.summary.text('prob', tf.as_string(probs[0,:]))

        add_param_summary(('.*', ['histogram']))   # monitor W  .*/W

        return total_loss