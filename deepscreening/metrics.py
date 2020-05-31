# coding: utf-8
import tensorflow as tf

def nearest_label_accuracy(y_true, y_pred):
    dot_product = tf.matmul(y_pred, tf.transpose(y_pred))
    square_norm = tf.linalg.tensor_diag_part(dot_product)
    distances  = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
    distances  = tf.maximum(distances, 0.0)

    distances += tf.linalg.tensor_diag(tf.reduce_max(distances, axis=1))
    nearest_labels = tf.argmin(distances, axis=1)

    return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.gather(y_true, nearest_labels)), tf.float32))
