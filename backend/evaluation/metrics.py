
def recall_at_k(results, relevant_pages, k):
    """
    Fraction of relevant pages retrieved within top K.
    """
    if not relevant_pages:
        return None

    retrieved_pages = {
        result["page_number"]
        for result in results[:k]
    }

    relevant_retrieved = retrieved_pages.intersection(
        relevant_pages
    )

    return len(relevant_retrieved) / len(relevant_pages)


def reciprocal_rank(results, relevant_pages):
    """
    Reciprocal rank of the first relevant result.
    """
    for rank, result in enumerate(results, start=1):

        if result["page_number"] in relevant_pages:
            return 1 / rank

    return 0.0


def hit_rate_at_k(results, relevant_pages, k):
    """
    Returns 1 if at least one relevant page appears in top K.
    """
    if not relevant_pages:
        return None

    retrieved_pages = {
        result["page_number"]
        for result in results[:k]
    }

    return int(bool(retrieved_pages.intersection(relevant_pages)))

def classification_metrics(y_true, y_pred, labels):

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths")

    if not labels:
        raise ValueError("labels cannot be empty")

    if any(label not in labels for label in y_true + y_pred):
        raise ValueError("Found an unknown classification label")

    confusion = {
        actual: {
            predicted: 0
            for predicted in labels
        }
        for actual in labels
    }

    for actual, predicted in zip(y_true, y_pred):
        confusion[actual][predicted] += 1

    results = {}

    for label in labels:

        tp = confusion[label][label]

        fp = sum(
            confusion[actual][label]
            for actual in labels
            if actual != label
        )

        fn = sum(
            confusion[label][predicted]
            for predicted in labels
            if predicted != label
        )

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    accuracy = (
        sum(confusion[label][label] for label in labels) / len(y_true)
        if y_true else 0
    )

    return {
        "confusion_matrix": confusion,
        "per_class": results,
        "accuracy": accuracy
    }