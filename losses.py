from tensorflow.keras import backend as K


def precision(y_true, y_pred):
    y_true = K.permute_dimensions(y_true, (3, 1, 2, 0))
    y_pred = K.permute_dimensions(y_pred, (3, 1, 2, 0))

    y_true_current = K.batch_flatten(y_true)
    y_pred_current = K.batch_flatten(y_pred)

    true_pos = K.sum(y_true_current * y_pred_current, 1)
    false_pos = K.sum((1 - y_true_current) * y_pred_current, 1)
    precision = true_pos / (true_pos + false_pos + K.epsilon())

    return precision


def recall(y_true, y_pred):
    y_true = K.permute_dimensions(y_true, (3, 1, 2, 0))
    y_pred = K.permute_dimensions(y_pred, (3, 1, 2, 0))

    y_true_current = K.batch_flatten(y_true)
    y_pred_current = K.batch_flatten(y_pred)
    true_pos = K.sum(y_true_current * y_pred_current, 1)
    false_neg = K.sum(y_true_current * (1 - y_pred_current), 1)

    recall = true_pos / (true_pos + false_neg + K.epsilon())

    return recall


def TP(y_true, y_pred):
    y_true = K.permute_dimensions(y_true, (3, 1, 2, 0))
    y_pred = K.permute_dimensions(y_pred, (3, 1, 2, 0))

    y_true_current = K.batch_flatten(y_true)
    y_pred_current = K.batch_flatten(y_pred)
    true_pos = K.sum(y_true_current * y_pred_current, 1)

    return true_pos


def FP(y_true, y_pred):
    y_true = K.permute_dimensions(y_true, (3, 1, 2, 0))
    y_pred = K.permute_dimensions(y_pred, (3, 1, 2, 0))

    y_true_current = K.batch_flatten(y_true)
    y_pred_current = K.batch_flatten(y_pred)

    false_pos = K.sum((1 - y_true_current) * y_pred_current, 1)

    return false_pos


def FN(y_true, y_pred):
    y_true = K.permute_dimensions(y_true, (3, 1, 2, 0))
    y_pred = K.permute_dimensions(y_pred, (3, 1, 2, 0))

    y_true_current = K.batch_flatten(y_true)
    y_pred_current = K.batch_flatten(y_pred)

    false_neg = K.sum(y_true_current * (1 - y_pred_current), 1)

    return false_neg


def f1_score(y_true, y_pred, numLabels=5):
    precisions = K.sum(precision(y_true, y_pred)) / numLabels
    recalls = K.sum(recall(y_true, y_pred)) / numLabels

    f1_score = 2 * ((precisions * recalls) / (precisions + recalls + K.epsilon()))
    return f1_score


# Reference: https://github.com/nabsabraham/focal-tversky-unet/issues/3
def class_tversky(y_true, y_pred):
    smooth = 1

    y_true = K.permute_dimensions(y_true, (3, 1, 2, 0))
    y_pred = K.permute_dimensions(y_pred, (3, 1, 2, 0))

    y_true_pos = K.batch_flatten(y_true)
    y_pred_pos = K.batch_flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos, 1)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos), 1)
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos, 1)
    alpha = 0.25
    return (true_pos + smooth) / (
        true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth
    )


# channels sensitive loss function
def focal_tversky_loss(y_true, y_pred):
    pt_1 = class_tversky(y_true, y_pred)
    gamma = 0.75
    return K.sum(K.pow((1 - pt_1), gamma))
