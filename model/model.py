#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.layers import xavier_initializer
from model.inception_v4 import *
import time

relu = tf.nn.relu
  
def conv(x, channels, kernel_size, stride, scope, normalizer_fn=slim.batch_norm, activation_fn=relu):
    return slim.conv2d(x, channels, kernel_size, stride, scope=scope, normalizer_fn=normalizer_fn, activation_fn=activation_fn)

def inception_padding_model(images, labels, wemb_size, seq_len, num_classes, lstm_size, is_training=True, 
        dropout_keep_prob=0.5, weight_decay=0.00004, final_endpoint='Mixed_6h', name='InceptionV4', reuse=None):
    """
    Core default tensorflow model for text recognition.
    Args:
        images: input images
        labels: input groundtruth labels 
        wemb_size: word embedding size
        seq_len: max sequence length for lstm with end of sequence
        num_classes: text label classes
        lstm_size: lstm size
        is_training: tensorflow placeholder
        dropout_keep_prob: tensorflow placeholder for dropout
        weight_decay: tensorflow model weight decay factor
        final_endpoint: final endpoint for CNN(InceptionV4)
        name: name scope
        reuse: reuse parameter

    Returns:
        output_array: (batch, seq_len, num_classes) logits
        attention_array: (batch, h, w, seq_len) attention feature map
    """
    with tf.compat.v1.variable_scope(name, reuse=reuse) as scope:
        regularizer = slim.l2_regularizer(weight_decay)
        with slim.arg_scope(inception_v4_arg_scope()):
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                net, end_points = inception_v4_base(images, final_endpoint=final_endpoint, scope=scope)  # Mixed_6h Mixed_7d
                # print(net.get_shape().as_list())
                cnn_feature = net

                output_array = tf.TensorArray(dtype=tf.float32, size=seq_len)
                batch, height, width, channel = cnn_feature.get_shape().as_list()
                attention_array = tf.TensorArray(dtype=tf.float32, size=seq_len)           
                
                with tf.compat.v1.variable_scope("attention_lstm"):
                    with tf.compat.v1.variable_scope('word_embedding'):
                        W_wemb = tf.compat.v1.get_variable('W_wemb', [num_classes, wemb_size], initializer=xavier_initializer())

                    with tf.compat.v1.variable_scope("feature_map_attention"):
                        with tf.compat.v1.variable_scope("init_mean"):
                            W_init_c = tf.compat.v1.get_variable('W_init_c', [channel, lstm_size], initializer=xavier_initializer(), regularizer=regularizer)
                            W_init_h = tf.compat.v1.get_variable('W_init_h', [channel, lstm_size], initializer=xavier_initializer(), regularizer=regularizer)
                        with tf.compat.v1.variable_scope("attention_x"):
                            W = tf.compat.v1.get_variable('W', [channel, channel], initializer=xavier_initializer(), regularizer=regularizer)
                        with tf.compat.v1.variable_scope("attention_h"):
                            W_h = tf.compat.v1.get_variable('W_h', [lstm_size, channel], initializer=xavier_initializer(), regularizer=regularizer)#tf.truncated_normal_initializer(stddev=0.1)
                        with tf.compat.v1.variable_scope("att"):
                            W_att = tf.compat.v1.get_variable('W_att', [channel, 1], initializer=xavier_initializer(), regularizer=regularizer)
                            b_att = tf.compat.v1.get_variable('b_att', [1], initializer=xavier_initializer())

                    with tf.compat.v1.variable_scope("softmax"):
                        softmax_w = tf.compat.v1.get_variable('softmax_w', [wemb_size, num_classes], initializer=xavier_initializer(), regularizer=regularizer)
                        softmax_b = tf.compat.v1.get_variable('softmax_b', [num_classes], initializer=xavier_initializer())#tf.constant_initializer(biivector)

                    with tf.compat.v1.variable_scope("attention_to_embedding"):
                        W_attention_wemb = tf.compat.v1.get_variable('W_attention_wemb', [channel, wemb_size], initializer=xavier_initializer(), regularizer=regularizer)
                        W_hidden_wemd = tf.compat.v1.get_variable('W_hidden_wemd', [lstm_size, wemb_size], initializer=xavier_initializer(), regularizer=regularizer)

                    with tf.compat.v1.variable_scope('lstm_cell'): #orthogonal_initializer  xavier_initializer()
                        lstm_W = tf.compat.v1.get_variable('lstm_W', [wemb_size, lstm_size*4], initializer=tf.initializers.orthogonal(), regularizer=regularizer)
                        lstm_U = tf.compat.v1.get_variable('lstm_U', [lstm_size, lstm_size*4], initializer=tf.initializers.orthogonal(), regularizer=regularizer)
                        lstm_Z = tf.compat.v1.get_variable('lstm_Z', [channel, lstm_size*4], initializer=tf.initializers.orthogonal(), regularizer=regularizer)
                        lstm_b = tf.compat.v1.get_variable('lstm_b', [lstm_size*4], initializer=xavier_initializer())
                        

                def _LSTMCell(wemb_prev, h_prev, attention_feature, c_prev, forget_bias=1., keep_prob=1.):
                    """
                    Image Caption Attention Mechanism LSTM, refer to https://arxiv.org/abs/1502.03044.
                    wemb_prev  :                   (batch, wemb_size)
                    h_prev  :                      (batch, lstm_size)
                    attention_feature  :           (batch, channel)
                    c_prev  :                      (batch, lstm_size)
                    """
                    pack = tf.add_n([tf.matmul(wemb_prev, lstm_W), tf.matmul(h_prev, lstm_U), tf.matmul(attention_feature, lstm_Z)])
                    pack_with_bias = tf.nn.bias_add(pack, lstm_b)   # (bsize, hid_dim * 4)
                    i, f, o, g = tf.split(pack_with_bias, num_or_size_splits=4, axis=1) # (bsize, hid_dim)
                    i = tf.sigmoid(i)
                    f = tf.sigmoid(f) # (f + forget_bias)
                    o = tf.sigmoid(o)
                    g = tf.tanh(g)
                    c = tf.add(tf.multiply(f, c_prev), tf.multiply(i, g))
                    h = tf.multiply(o, tf.tanh(c))

                    h = tf.nn.dropout(h, rate = 1-keep_prob)
                    c = tf.nn.dropout(c, rate = 1-keep_prob)
                    return h, c

                mean_inputs = tf.reduce_mean(cnn_feature, [1, 2])
                init_h = tf.tanh(tf.matmul(mean_inputs, W_init_h))
                init_c = tf.tanh(tf.matmul(mean_inputs, W_init_c))
                init_wemb = tf.zeros([tf.shape(mean_inputs)[0], wemb_size])

                def attention_lstm(i, cnn_feature, wemb_prev, hidden_state, cell_state, output_array, attention_array):
                    """
                    Loop body for AttentionOCR.
                    Args:
                        i: loop count
                        cnn_feature: cnn feature map with shape (batch, height, width, channel)
                        bboxes: groundtruth boxes for crop text region
                        wemb_prev: previous word embedding
                        hidden_state: prev lstm hidden state
                        cell_state: prev lstm cell state
                        output_array: softmax logit TensorArray at time step i 
                        attention_array: attention feature map TensorArray at time step i 
                    """
                    # Bahdanau/Additive Attention Mechanism, refer to https://arxiv.org/pdf/1409.0473.pdf.
                    attention_x = tf.reshape(cnn_feature, [-1, channel])
                    attention_x = tf.matmul(attention_x, W)
                    attention_x = tf.reshape(attention_x, [-1, height*width, channel])  #(batch, h*w, channel)
                    
                    # rnn hidden state transform  (batch, lstm_size) -> attention feature h (batch, channel) 
                    attention_h = tf.matmul(hidden_state, W_h)  #(batch, channel)
                    
                    # Score function for Attention Mechanism
                    att = tf.tanh(attention_x + tf.expand_dims(attention_h, axis=1)) #(batch, h*w, channel)

                    # Align function for Attention Mechanism  (batch, h*w, channel) -> (batch, h*w)
                    att = tf.reshape(att, [-1, channel])
                    att = tf.add(tf.matmul(att, W_att), b_att) 
                    att = tf.reshape(att, [-1, height*width])
                    alpha = tf.nn.softmax(att)      #(batch, h*w)

                    # store attention
                    attention_array = attention_array.write(i, alpha)

                    # attention context feature for lstm input
                    x = attention_x * tf.expand_dims(alpha, axis=2) #(batch, h*w, channel)
                    attention_feature = tf.reduce_sum(x, axis=1) # (batch, channel)

                    # compute new state by previous word embedding, attention feature and state
                    hidden_state, cell_state = _LSTMCell(wemb_prev, hidden_state, attention_feature, cell_state, keep_prob=dropout_keep_prob)
                    
                    # compute output feature for softmax by wemb_prev and transformed hidden state, attention feature
                    attention_wemb = tf.matmul(attention_feature, W_attention_wemb)
                    hidden_wemd = tf.matmul(hidden_state, W_hidden_wemd)

                    #output = tf.add_n([attention_wemb, hidden_wemd])
                    output = tf.add_n([attention_wemb, hidden_wemd, wemb_prev])

                    output = tf.matmul(output, softmax_w) + softmax_b

                    # compute word embedding in different behaviours between train(by groundtruth) and test(by max_prob_word)
                    wemb_prev = tf.cond(is_training, lambda: tf.nn.embedding_lookup(W_wemb, labels[:,i]), \
                            lambda: tf.nn.embedding_lookup(W_wemb, tf.argmax(tf.nn.softmax(output), 1)) )

                    output_array = output_array.write(i, output)

                    return i+1, cnn_feature, wemb_prev, hidden_state, cell_state, output_array, attention_array

                _, _, _, _, _, output_array, attention_array = tf.while_loop(cond=lambda i, *_: i < seq_len, body=attention_lstm, \
                        loop_vars=(tf.constant(0, tf.int32), cnn_feature, init_wemb, init_h, init_c, output_array, attention_array))#, shape_invariants=

                output_array = tf.transpose(output_array.stack(), [1, 0, 2]) # (batch, seq_len, num_classes)
                attention_array = tf.transpose(attention_array.stack(), [1, 2, 0]) # (batch, seq_len, h*w)
                attention_array = tf.reshape(attention_array, [-1, height, width, seq_len])

            return output_array, attention_array


def inception_model(images, labels, bboxes, wemb_size, seq_len, num_classes, lstm_size, is_training=True, 
        dropout_keep_prob=0.5, weight_decay=0.00004, final_endpoint='Mixed_6h', name='InceptionV4', reuse=None):
    """
    Core tensorflow model for text recognition.
    Args:
        images: input images
        labels: input groundtruth labels 
        bboxes: input groundtruth boxes for text region extract due to preprocess image padding
        wemb_size: word embedding size
        seq_len: max sequence length for lstm with end of sequence
        num_classes: text label classes
        lstm_size: lstm size
        is_training: tensorflow placeholder
        dropout_keep_prob: tensorflow placeholder for dropout
        weight_decay: tensorflow model weight decay factor
        final_endpoint: final endpoint for CNN(InceptionV4)
        name: name scope
        reuse: reuse parameter

    Returns:
        output_array: (batch, seq_len, num_classes) logits
        attention_array: (batch, h, w, seq_len) attention feature map
    """
    with tf.compat.v1.variable_scope(name, reuse=reuse) as scope:
        regularizer = slim.l2_regularizer(weight_decay)
        with slim.arg_scope(inception_v4_arg_scope()):
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                net, end_points = inception_v4_base(images, final_endpoint=final_endpoint, scope=scope)  # Mixed_6h Mixed_7d
                # 299 -> 149 -> 147 -> 73 -> 71 -> 35 -> 17 -> 8    8X+19   16X+27   32X+43
                # print(net.get_shape().as_list())
                cnn_feature = net

                output_array = tf.TensorArray(dtype=tf.float32, size=seq_len)
                batch, height, width, channel = cnn_feature.get_shape().as_list()
                attention_array = tf.TensorArray(dtype=tf.float32, size=seq_len)           
                
                with tf.compat.v1.variable_scope("attention_lstm"):
                    with tf.compat.v1.variable_scope('word_embedding'):
                        W_wemb = tf.compat.v1.get_variable('W_wemb', [num_classes, wemb_size], initializer=xavier_initializer())

                    with tf.compat.v1.variable_scope("feature_map_attention"):
                        with tf.compat.v1.variable_scope("init_mean"):
                            W_init_c = tf.compat.v1.get_variable('W_init_c', [channel, lstm_size], initializer=xavier_initializer(), regularizer=regularizer)
                            W_init_h = tf.compat.v1.get_variable('W_init_h', [channel, lstm_size], initializer=xavier_initializer(), regularizer=regularizer)
                        with tf.compat.v1.variable_scope("attention_x"):
                            W = tf.compat.v1.get_variable('W', [channel, channel], initializer=xavier_initializer(), regularizer=regularizer)
                        with tf.compat.v1.variable_scope("attention_h"):
                            W_h = tf.compat.v1.get_variable('W_h', [lstm_size, channel], initializer=xavier_initializer(), regularizer=regularizer)#tf.truncated_normal_initializer(stddev=0.1)
                        with tf.compat.v1.variable_scope("att"):
                            W_att = tf.compat.v1.get_variable('W_att', [channel, 1], initializer=xavier_initializer(), regularizer=regularizer)
                            b_att = tf.compat.v1.get_variable('b_att', [1], initializer=xavier_initializer())

                    with tf.compat.v1.variable_scope("softmax"):
                        softmax_w = tf.compat.v1.get_variable('softmax_w', [wemb_size, num_classes], initializer=xavier_initializer(), regularizer=regularizer)
                        softmax_b = tf.compat.v1.get_variable('softmax_b', [num_classes], initializer=xavier_initializer())#tf.constant_initializer(biivector)

                    with tf.compat.v1.variable_scope("attention_to_embedding"):
                        W_attention_wemb = tf.compat.v1.get_variable('W_attention_wemb', [channel, wemb_size], initializer=xavier_initializer(), regularizer=regularizer)
                        W_hidden_wemd = tf.compat.v1.get_variable('W_hidden_wemd', [lstm_size, wemb_size], initializer=xavier_initializer(), regularizer=regularizer)

                    with tf.compat.v1.variable_scope('lstm_cell'): #orthogonal_initializer  xavier_initializer()
                        lstm_W = tf.compat.v1.get_variable('lstm_W', [wemb_size, lstm_size*4], initializer=tf.initializers.orthogonal(), regularizer=regularizer)
                        lstm_U = tf.compat.v1.get_variable('lstm_U', [lstm_size, lstm_size*4], initializer=tf.initializers.orthogonal(), regularizer=regularizer)
                        lstm_Z = tf.compat.v1.get_variable('lstm_Z', [channel, lstm_size*4], initializer=tf.initializers.orthogonal(), regularizer=regularizer)
                        lstm_b = tf.compat.v1.get_variable('lstm_b', [lstm_size*4], initializer=xavier_initializer())
                        

                def _LSTMCell(wemb_prev, h_prev, attention_feature, c_prev, forget_bias=1., keep_prob=1.):
                    """
                    wemb_prev  :                   (batch, wemb_size)
                    h_prev  :                      (batch, lstm_size)
                    attention_feature  :           (batch, channel)
                    c_prev  :                      (batch, lstm_size)
                    """
                    pack = tf.add_n([tf.matmul(wemb_prev, lstm_W), tf.matmul(h_prev, lstm_U), tf.matmul(attention_feature, lstm_Z)])
                    pack_with_bias = tf.nn.bias_add(pack, lstm_b)   # (bsize, hid_dim * 4)
                    i, f, o, g = tf.split(pack_with_bias, num_or_size_splits=4, axis=1) # (bsize, hid_dim)
                    i = tf.sigmoid(i)
                    f = tf.sigmoid(f) # (f + forget_bias)
                    o = tf.sigmoid(o)
                    g = tf.tanh(g)
                    c = tf.add(tf.multiply(f, c_prev), tf.multiply(i, g))
                    h = tf.multiply(o, tf.tanh(c))

                    h = tf.nn.dropout(h, rate = 1-keep_prob)
                    c = tf.nn.dropout(c, rate = 1-keep_prob)
                    return h, c

                mean_inputs = tf.reduce_mean(cnn_feature, [1, 2])
                init_h = tf.tanh(tf.matmul(mean_inputs, W_init_h))
                init_c = tf.tanh(tf.matmul(mean_inputs, W_init_c))
                init_y = tf.zeros([tf.shape(mean_inputs)[0], wemb_size])

                def attention_lstm(i, cnn_feature, bboxes, wemb_prev, hidden_state, cell_state, output_array, attention_array):
                    """
                    Loop body for AttentionOCR.
                    Args:
                        i: loop count
                        cnn_feature: cnn feature map with shape (batch, height, width, channel)
                        bboxes: groundtruth boxes for crop text region
                        wemb_prev: previous word embedding
                        hidden_state: prev lstm hidden state
                        cell_state: prev lstm cell state
                        output_array: softmax logit TensorArray at time step i 
                        attention_array: attention feature map TensorArray at time step i 
                    """
                    def map_fn_for_attention(image, bbox, attention_h):
                        """
                        attention mechanism for each image
                        """
                        offset_height, offset_width, target_height, target_width = tf.cast(bbox[0]*height, tf.int32), \
                                tf.cast(bbox[1]*width, tf.int32), tf.cast(bbox[2]*height, tf.int32), tf.cast(bbox[3]*width, tf.int32)
                        image = tf.compat.v1.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
                        
                        # score function for context similarity
                        attention_x = tf.reshape(image, [-1, channel]) #(h*w, c)
                        attention_x = tf.matmul(attention_x, W)
                        
                        att = tf.tanh(attention_x + tf.expand_dims(attention_h, axis=0)) #(h*w, c)

                        # attention weight
                        att = tf.add(tf.matmul(att, W_att), b_att)  #(h*w, 1)
                        alpha = tf.nn.softmax(att, axis=0)

                        x = attention_x * alpha   #(h*w, c)
                        attention_feature = tf.reduce_sum(x, axis=0)   # (c)

                        alpha = tf.reshape(alpha, [target_height, target_width])
                        paddings = [[offset_height, height-target_height-offset_height], [offset_width, width-offset_width-target_width]]
                        alpha = tf.pad(alpha, paddings, 'CONSTANT')

                        return attention_feature, alpha


                    # rnn hidden state transform  (batch, lstm_size) -> attention feature h (batch, channel) 
                    attention_h = tf.matmul(hidden_state, W_h)  #(batch, channel)   attention feature in channel dimension

                    attention_feature, alpha = tf.map_fn(lambda x: map_fn_for_attention(x[0], x[1], x[2]), \
                            (cnn_feature, bboxes, attention_h), (tf.float32, tf.float32))

                    # store attention
                    attention_array = attention_array.write(i, tf.reshape(alpha, [-1, height*width]))


                    # compute new state by previous word embedding, attention feature and state
                    hidden_state, cell_state = _LSTMCell(wemb_prev, hidden_state, attention_feature, cell_state, keep_prob=dropout_keep_prob)
                    
                    # compute output feature for softmax by wemb_prev and transformed hidden state, attention feature
                    attention_wemb = tf.matmul(attention_feature, W_attention_wemb)
                    hidden_wemd = tf.matmul(hidden_state, W_hidden_wemd)

                    #output = tf.add_n([attention_wemb, hidden_wemd])
                    output = tf.add_n([attention_wemb, hidden_wemd, wemb_prev])

                    output = tf.matmul(output, softmax_w) + softmax_b

                    # compute word embedding in different behaviours between train(by groundtruth) and test(by max_prob_word)
                    wemb_prev = tf.cond(is_training, lambda: tf.nn.embedding_lookup(W_wemb, labels[:,i]), \
                            lambda: tf.nn.embedding_lookup(W_wemb, tf.argmax(tf.nn.softmax(output), 1)) )

                    output_array = output_array.write(i, output)

                    return i+1, cnn_feature, bboxes, wemb_prev, hidden_state, cell_state, output_array, attention_array

                _, _, _, _, _, _, output_array, attention_array = tf.while_loop(cond=lambda i, *_: i < seq_len, body=attention_lstm, \
                        loop_vars=(tf.constant(0, tf.int32), cnn_feature, bboxes, init_y, init_h, init_c, output_array, attention_array))#, shape_invariants=

                output_array = tf.transpose(output_array.stack(), [1, 0, 2]) # (batch, seq_len, num_classes)
                attention_array = tf.transpose(attention_array.stack(), [1, 2, 0]) # (batch, seq_len, h*w)
                attention_array = tf.reshape(attention_array, [-1, height, width, seq_len])

            return output_array, attention_array

            
if __name__=='__main__':
    os.environ['CUDA_VISIBILE_DEVICES']='9'
    
    size = 256
    input_placeholder = tf.compat.v1.placeholder(tf.float32, [1, size, size, 3]) 
    label_batch = tf.ones([1, 32], dtype=tf.int64)
    bbox_batch = tf.constant([[0., 0., 1., 1.]])
    is_training = tf.compat.v1.placeholder(tf.bool) 
    dropout_keep_prob = tf.compat.v1.placeholder(tf.float32)
    outputs, attentions = inception_model(input_placeholder, label_batch, bbox_batch, wemb_size=256, seq_len=32, num_classes=10000, \
            lstm_size=512, is_training = is_training, dropout_keep_prob=dropout_keep_prob)
    # outputs, attentions = inception_padding_model(input_placeholder, label_batch, wemb_size=256, seq_len=32, num_classes=10000, \
    #         lstm_size=512, is_training = is_training, dropout_keep_prob=dropout_keep_prob)
    print(outputs.get_shape().as_list())
    logits = tf.nn.softmax(outputs)
    preds = tf.argmax(logits, axis=-1)
    
    with tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True), allow_soft_placement=True)) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        count = 0
        for i in range(1000):
            before_op = time.time()
            results = sess.run([preds], feed_dict={input_placeholder: np.random.random_sample((1, size, size, 3)), is_training: False, dropout_keep_prob: 0.5})
            after_op = time.time()
            print(after_op - before_op)
            count += after_op - before_op
        print(count)