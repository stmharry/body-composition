import functools
import tensorflow as tf

tf_v1 = tf.compat.v1


def _confusion_matrix(labels, predictions):
    conf_mat = tf.math.confusion_matrix(
        labels=labels, predictions=predictions, num_classes=2)
    tp = conf_mat[1, 1]
    fn = conf_mat[1, 0]
    fp = conf_mat[0, 1]
    tn = conf_mat[0, 0]
    return (tp, fn, fp, tn)


def accuracy(labels, predictions, name='accuracy'):
    (tp, fn, fp, tn) = _confusion_matrix(
        labels=labels, predictions=predictions)
    return tf.divide(tp + tn, tp + fn + fp + tn, name=name)


def precision(labels, predictions, name='precision'):
    (tp, fn, fp, tn) = _confusion_matrix(
        labels=labels, predictions=predictions)
    return tf.divide(tp, tp + fp, name=name)


def recall(labels, predictions, name='recall'):
    (tp, fn, fp, tn) = _confusion_matrix(
        labels=labels, predictions=predictions)
    return tf.divide(tp, tp + fn, name=name)


streaming_accuracy = functools.partial(
    tf_v1.metrics.accuracy,
    updates_collections=tf_v1.GraphKeys.UPDATE_OPS)

streaming_precision = functools.partial(
    tf_v1.metrics.precision,
    updates_collections=tf_v1.GraphKeys.UPDATE_OPS)

streaming_recall = functools.partial(
    tf_v1.metrics.recall,
    updates_collections=tf_v1.GraphKeys.UPDATE_OPS)

streaming_auc = functools.partial(
    tf_v1.metrics.auc,
    updates_collections=tf_v1.GraphKeys.UPDATE_OPS)
