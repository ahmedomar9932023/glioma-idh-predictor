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


def interpretation_text(
    label: str,
    probability: float,
    confidence: str,
    borderline: bool,
    driver_genes: list[str] | None = None,
) -> str:
    probability = float(probability)
    mutant_probability = f"{probability * 100:.1f}%"
    if label == "IDH-mutant":
        if confidence == "High confidence":
            base = (
                "This sample shows a strong IDH-mutant-like expression pattern. "
                f"The model assigned a high mutant probability ({mutant_probability}), suggesting that the "
                "gene-expression profile is closer to IDH-mutant gliomas."
            )
        elif confidence == "Moderate confidence":
            base = (
                "This sample is predicted as IDH-mutant with moderate confidence. "
                f"The mutant probability is {mutant_probability}, so the expression profile contains mutant-like "
                "features but should still be interpreted alongside other molecular findings."
            )
        else:
            base = (
                "This sample is predicted as IDH-mutant, but the confidence is limited. "
                f"The mutant probability is {mutant_probability}, which suggests some mutant-like transcriptional "
                "features without a strong separation from the decision boundary."
            )
    else:
        if confidence == "High confidence":
            base = (
                "This sample shows a strong IDH-wildtype-like expression pattern. "
                f"The model assigned a low mutant probability ({mutant_probability}), suggesting that the "
                "expression profile is closer to IDH-wildtype gliomas."
            )
        elif confidence == "Moderate confidence":
            base = (
                "This sample is predicted as IDH-wildtype with moderate confidence. "
                f"The mutant probability is {mutant_probability}, so the expression profile trends toward "
                "IDH-wildtype biology but is not maximally separated."
            )
        else:
            base = (
                "This sample is predicted as IDH-wildtype, but the confidence is limited. "
                f"The mutant probability is {mutant_probability}, so the model is less certain and the result "
                "should be reviewed with additional molecular or clinical context."
            )
    if borderline:
        base += " The score lies near the decision boundary, so this prediction should be interpreted cautiously."
    if driver_genes:
        base += " Globally influential genes aligned with this prediction include: " + ", ".join(driver_genes) + "."
    return base
