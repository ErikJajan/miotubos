"""Línea base clásica de contraste local y evaluación exacta por píxel."""

import numpy as np
from PIL import ImageFilter

VARIANTS = ("claro", "oscuro", "absoluto")


def score_maps(image):
    """Calcula contraste local sin estadísticas aprendidas de otras imágenes."""
    smooth = np.asarray(image.filter(ImageFilter.GaussianBlur(2)), dtype=np.int16)
    background = np.asarray(image.filter(ImageFilter.GaussianBlur(28)), dtype=np.int16)
    difference = smooth - background
    return {
        "claro": np.maximum(difference, 0).astype(np.uint8),
        "oscuro": np.maximum(-difference, 0).astype(np.uint8),
        "absoluto": np.abs(difference).astype(np.uint8),
    }


def score_histograms(root, records, variants=VARIANTS):
    from PIL import Image

    counts = {name: np.zeros((2, 256), dtype=np.int64) for name in variants}
    for record in records:
        with Image.open(root / record["image"]) as image:
            scores = score_maps(image)
        with Image.open(root / record["mask"]) as mask:
            positive = np.asarray(mask).ravel() > 0
        for name in variants:
            # Índices 0–255 para fondo y 256–511 para miotubo.
            bins = scores[name].ravel().astype(np.uint16)
            bins += positive.astype(np.uint16) * 256
            counts[name] += np.bincount(bins, minlength=512).reshape(2, 256)
    return counts


def curves(counts):
    """Curvas microagregadas. Los empates se evalúan juntos."""
    negatives, positives = counts
    if not negatives.sum() or not positives.sum():
        raise ValueError("La evaluación requiere ambas clases")
    fp = np.r_[0, negatives[::-1].cumsum()]
    tp = np.r_[0, positives[::-1].cumsum()]
    recall = tp / positives.sum()
    fpr = fp / negatives.sum()
    precision = np.ones(len(tp))
    np.divide(tp, tp + fp, out=precision, where=(tp + fp) > 0)
    return dict(
        thresholds=np.arange(256, -1, -1), recall=recall, precision=precision, fpr=fpr,
        predicted=tp + fp, average_precision=float(np.sum(np.diff(recall) * precision[1:])),
        roc_auc=float(np.trapezoid(recall, fpr)),
    )


def confusion(counts, threshold):
    """Filas reales y columnas predichas, en orden fondo, miotubo."""
    below = counts[:, :threshold].sum(axis=1)
    at_or_above = counts[:, threshold:].sum(axis=1)
    return np.column_stack((below, at_or_above))


def evaluate(counts, precision_target=0.80):
    curve = curves(counts)
    recall = curve["recall"]
    precision = curve["precision"]
    valid = np.flatnonzero((curve["predicted"] > 0) & (precision >= precision_target))
    # La curva recorre umbrales de mayor a menor. El primer empate es conservador.
    operating_index = valid[np.argmax(recall[valid])] if len(valid) else None
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(recall), where=(precision + recall) > 0)
    index = operating_index if operating_index is not None else np.argmax(f1)
    threshold = int(curve["thresholds"][index])
    matrix = confusion(counts, threshold)
    (_, fp), (fn, tp) = matrix
    return dict(
        pr_auc=curve["average_precision"], roc_auc=curve["roc_auc"],
        prevalence=float(counts[1].sum() / counts.sum()),
        precision_target=precision_target,
        recall_at_precision_080=float(recall[operating_index]) if operating_index is not None else None,
        threshold=threshold, precision=float(precision[index]), recall=float(recall[index]),
        predicted_positive=int(curve["predicted"][index]),
        criterion="precision >= 0.80" if operating_index is not None else "maximo Dice en validacion",
        dice=float(f1[index]), iou=float(tp / (tp + fp + fn)),
        confusion=matrix.tolist(),
        max_precision=float(precision[curve["predicted"] > 0].max()),
    )
