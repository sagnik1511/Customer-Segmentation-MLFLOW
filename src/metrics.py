from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, precision_score, recall_score

metric_dict = {
    "f1_score": f1_score,
    "accuracy_score": accuracy_score,
    "cohen_kappa_score": cohen_kappa_score,
    "precision": precision_score,
    "recall": recall_score
}


def metric_values(x, y, model):
    actual = y
    pred = model.predict(x)
    op_dict = {
        m: -1 for m, _ in metric_dict.items()
    }
    for m in metric_dict.keys():
        if m == "f1_score" or m == "precision" or m == "recall":
            op_dict[m] = metric_dict[m](actual, pred, average="weighted")
        else:
            op_dict[m] = metric_dict[m](actual, pred)
    return op_dict



