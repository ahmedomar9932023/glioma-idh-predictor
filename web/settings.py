from __future__ import annotations


DEFAULT_DECISION_THRESHOLD = 0.50
BORDERLINE_MARGIN = 0.10
HIGH_CONFIDENCE_DISTANCE = 0.35
MODERATE_CONFIDENCE_DISTANCE = 0.20
MIN_FEATURE_OVERLAP = 0.95


def confidence_level(probability: float, threshold: float = DEFAULT_DECISION_THRESHOLD) -> str:
    distance = abs(float(probability) - float(threshold))
    if distance >= HIGH_CONFIDENCE_DISTANCE:
        return "High confidence"
    if distance >= MODERATE_CONFIDENCE_DISTANCE:
        return "Moderate confidence"
    return "Borderline / uncertain"


def is_borderline(probability: float, threshold: float = DEFAULT_DECISION_THRESHOLD) -> bool:
    return abs(float(probability) - float(threshold)) < BORDERLINE_MARGIN


def predicted_status(probability: float, threshold: float = DEFAULT_DECISION_THRESHOLD) -> tuple[int, str]:
    status = int(float(probability) >= float(threshold))
    return status, "IDH-mutant" if status == 1 else "IDH-wildtype"


def interpretation_text(label: str, borderline: bool) -> str:
    if label == "IDH-mutant":
        base = "This sample is more consistent with an IDH-mutant glioma expression pattern."
    else:
        base = "This sample is more consistent with an IDH-wildtype glioma expression pattern."
    if borderline:
        return base + " This prediction is borderline and should be interpreted cautiously."
    return base

