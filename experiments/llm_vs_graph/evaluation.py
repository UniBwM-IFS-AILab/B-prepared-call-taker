import numpy as np

UNKNOWN, FALSE, TRUE = -1, 0, 1

def to_binary_true_matrix(y_state):
    # correctness view: only TRUE matters
    return (np.asarray(y_state) == TRUE).astype(np.uint8)

y_true = to_binary_true_matrix(y_true_state)   # shape: (n_dialogues, 70)
y_pred = to_binary_true_matrix(y_pred_state)   # shape: (n_dialogues, 70)


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, hamming_loss

exact_success = accuracy_score(y_true, y_pred)

required_recall = recall_score(
    y_true, y_pred,
    average="samples",
    zero_division=0.0,
)

positive_precision = precision_score(
    y_true, y_pred,
    average="samples",
    zero_division=0.0,
)

soft_f1 = f1_score(
    y_true, y_pred,
    average="samples",
    zero_division=0.0,
)

slot_error_rate = hamming_loss(y_true, y_pred)