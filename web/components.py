from __future__ import annotations

from html import escape

import pandas as pd

from .metadata import model_info, top_genes


def pct(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    return f"{float(value) * 100:.1f}%"


def ready_html() -> str:
    return """
    <section class="empty-state">
      <div class="empty-icon">IDH</div>
      <h3>Ready for prediction</h3>
      <p>Upload a supported expression file or try the bundled demo file. Results, charts, and downloads will appear here after prediction.</p>
    </section>
    """


def loading_html() -> str:
    return """
    <section class="loading-state" aria-live="polite">
      <div class="loading-spinner" aria-hidden="true"></div>
      <h3>Running inference...</h3>
      <p>Processing expression data with the saved model pipeline.</p>
      <span class="elapsed-time">Processing... 0.0s</span>
    </section>
    """


def error_html(message: str) -> str:
    return f"""
    <section class="error-state">
      <div class="error-kicker">Input check</div>
      <h3>Could not run prediction</h3>
      <p>{escape(message)}</p>
    </section>
    """


def success_html(metadata: dict) -> str:
    return f"""
    <section class="success-state">
      <div class="success-kicker">Prediction complete</div>
      <h3>{metadata['n_samples']} sample{'s' if metadata['n_samples'] != 1 else ''} processed</h3>
      <p>Download the enriched prediction CSV from the export panel. Borderline predictions are flagged for cautious interpretation.</p>
    </section>
    """


def summary_cards_html(metadata: dict) -> str:
    mean_probability = metadata.get("mean_probability", 0)
    borderline = metadata.get("n_borderline", 0)
    avg_confidence = metadata.get("average_confidence", "Unavailable")
    single = metadata.get("n_samples", 0) == 1
    mode_note = "Single-case view" if single else "Batch summary view"
    return f"""
    <section class="summary-grid" aria-label="Prediction summary">
      <article class="metric-card metric-samples">
        <span class="metric-label">Total samples</span>
        <strong>{metadata['n_samples']}</strong>
        <small>{mode_note}</small>
      </article>
      <article class="metric-card metric-mutant">
        <span class="metric-label">IDH-mutant</span>
        <strong>{metadata['n_predicted_mutant']}</strong>
        <small>predicted positive</small>
      </article>
      <article class="metric-card metric-wildtype">
        <span class="metric-label">IDH-wildtype</span>
        <strong>{metadata['n_predicted_wildtype']}</strong>
        <small>predicted negative</small>
      </article>
      <article class="metric-card metric-coverage">
        <span class="metric-label">Gene coverage</span>
        <strong>{metadata['feature_overlap'] * 100:.1f}%</strong>
        <small>{metadata['n_present_features']:,}/{metadata['n_required_features']:,} model genes</small>
      </article>
      <article class="metric-card metric-probability">
        <span class="metric-label">Mean IDH probability</span>
        <strong>{mean_probability * 100:.1f}%</strong>
        <small>across uploaded samples</small>
      </article>
      <article class="metric-card metric-borderline">
        <span class="metric-label">Borderline cases</span>
        <strong>{borderline}</strong>
        <small>average confidence: {escape(avg_confidence)}</small>
      </article>
    </section>
    """


def single_case_html(predictions: pd.DataFrame) -> str:
    if len(predictions) != 1:
        return ""
    row = predictions.iloc[0]
    warning = (
        "<p class='borderline-warning'>This prediction is borderline and should be interpreted cautiously.</p>"
        if bool(row["borderline_flag"])
        else ""
    )
    return f"""
    <section class="single-case-card">
      <div>
        <span class="single-kicker">Single sample interpretation</span>
        <h3>{escape(str(row['SAMPLE_ID']))}</h3>
        <p>{escape(str(row['interpretation']))}</p>
        {warning}
      </div>
      <div class="single-meter">
        <strong>{float(row['IDH_mutation_probability']) * 100:.1f}%</strong>
        <span>IDH-mutant probability</span>
      </div>
    </section>
    """


def results_table_html(predictions: pd.DataFrame) -> str:
    if predictions.empty:
        return """
        <section class="results-empty">
          <div class="empty-icon">RS</div>
          <h3>Results will appear here</h3>
          <p>Upload an expression file and run prediction to populate this dashboard with sample-level IDH calls, confidence labels, probability bars, and interpretation text.</p>
        </section>
        """

    rows = []
    for _, row in predictions.iterrows():
        label = str(row["predicted_label"])
        klass = "mutant" if label == "IDH-mutant" else "wildtype"
        confidence = str(row["confidence_level"])
        conf_class = confidence.lower().split()[0].replace("/", "")
        probability = float(row["IDH_mutation_probability"])
        warning = " <span class='row-warning'>Borderline</span>" if bool(row["borderline_flag"]) else ""
        rows.append(
            "<tr>"
            f"<td class='sample-cell'>{escape(str(row['SAMPLE_ID']))}</td>"
            f"<td><span class='prediction-pill {klass}'>{escape(label)}</span>{warning}</td>"
            f"<td><span class='confidence-pill {conf_class}'>{escape(confidence)}</span></td>"
            f"<td><div class='probability-cell'><span>{probability * 100:.1f}%</span>"
            f"<div class='probability-track'><i style='width:{max(0, min(100, probability * 100)):.1f}%'></i></div></div></td>"
            f"<td class='interpretation-cell'>{escape(str(row['interpretation']))}</td>"
            "</tr>"
        )

    return f"""
    <div class="results-table-wrap">
      <table class="results-table">
        <thead>
          <tr>
            <th>SAMPLE_ID</th>
            <th>Prediction</th>
            <th>Confidence</th>
            <th>IDH-mutant probability</th>
            <th>Interpretation</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def top_genes_html(limit: int = 12) -> str:
    genes = top_genes(limit)
    if genes.empty:
        return "<section class='top-genes-empty'><p>Top-gene file is not available.</p></section>"
    rows = "".join(
        f"<tr><td>{escape(str(row.gene))}</td><td>{float(row.importance):.4f}</td><td>{float(row.signed_value):+.4f}</td></tr>"
        for row in genes.itertuples(index=False)
    )
    return f"""
    <div class="interpretability-note">
      <p>These are globally influential model features from the saved pipeline. They are not causal biomarkers and are not per-sample explanations.</p>
    </div>
    <div class="mini-table-wrap">
      <table class="mini-table">
        <thead><tr><th>Gene</th><th>Importance</th><th>Signed value</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """


def model_info_html() -> str:
    info = model_info()
    return f"""
    <div class="model-info-grid">
      <div><span>Model type</span><strong>{escape(str(info.get('model_type', 'Linear SVM')))}</strong></div>
      <div><span>Training samples</span><strong>{info.get('training_samples', 'Unavailable')}</strong></div>
      <div><span>Held-out test samples</span><strong>{info.get('test_samples', 'Unavailable')}</strong></div>
      <div><span>Genes/features</span><strong>{info.get('genes', 'Unavailable')}</strong></div>
      <div><span>Test ROC AUC</span><strong>{pct(info.get('roc_auc'))}</strong></div>
      <div><span>Test balanced accuracy</span><strong>{pct(info.get('balanced_accuracy'))}</strong></div>
      <div><span>Test F1</span><strong>{pct(info.get('f1'))}</strong></div>
    </div>
    <p class="model-disclaimer">This is a research model trained for IDH mutation-status prediction in glioma from gene expression data.</p>
    """
