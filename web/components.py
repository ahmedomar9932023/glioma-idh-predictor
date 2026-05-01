from __future__ import annotations

from html import escape

import pandas as pd

from .metadata import class_driver_genes, model_info, top_genes


def pct(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    return f"{float(value) * 100:.1f}%"


def ready_html() -> str:
    return """
    <section class="empty-state">
      <div class="empty-icon">IDH</div>
      <h3>Ready for analysis</h3>
      <p>Upload a compatible gene-expression matrix or open the bundled example dataset to begin.</p>
    </section>
    """


def loading_html() -> str:
    return """
    <section class="loading-state" aria-live="polite">
      <div class="loading-spinner" aria-hidden="true"></div>
      <h3>Processing expression data</h3>
      <p>Applying the preserved classification pipeline to the uploaded sample set.</p>
      <span class="elapsed-time">Processing...</span>
    </section>
    """


def filtered_empty_html() -> str:
    return """
    <section class="results-empty filtered-results-empty">
      <div class="empty-icon">FL</div>
      <h3>No matching samples</h3>
      <p>No rows matched the current display criteria.</p>
    </section>
    """


def error_html(message: str) -> str:
    return f"""
    <section class="error-state">
      <div class="error-kicker">Input review</div>
      <h3>Analysis could not be completed</h3>
      <p>{escape(message)}</p>
    </section>
    """


def success_html(metadata: dict) -> str:
    return f"""
    <section class="success-state">
      <div class="success-kicker">Analysis complete</div>
      <h3>{metadata['n_samples']} sample{'s' if metadata['n_samples'] != 1 else ''} processed</h3>
      <p>Result files are available in the exports panel. Borderline classifications are flagged for cautious interpretation.</p>
    </section>
    """


def summary_cards_html(metadata: dict) -> str:
    mean_probability = metadata.get("mean_probability", 0)
    borderline = metadata.get("n_borderline", 0)
    avg_confidence = metadata.get("average_confidence", "Unavailable")
    single = metadata.get("n_samples", 0) == 1
    mode_note = "Single-sample view" if single else "Cohort summary"
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
        <small>classified as mutant-like</small>
      </article>
      <article class="metric-card metric-wildtype">
        <span class="metric-label">IDH-wildtype</span>
        <strong>{metadata['n_predicted_wildtype']}</strong>
        <small>classified as wildtype-like</small>
      </article>
      <article class="metric-card metric-coverage">
        <span class="metric-label">Gene coverage</span>
        <strong>{metadata['feature_overlap'] * 100:.1f}%</strong>
        <small>{metadata['n_present_features']:,}/{metadata['n_required_features']:,} required genes detected</small>
      </article>
      <article class="metric-card metric-probability">
        <span class="metric-label">Mean IDH probability</span>
        <strong>{mean_probability * 100:.1f}%</strong>
        <small>across the current sample set</small>
      </article>
      <article class="metric-card metric-borderline">
        <span class="metric-label">Borderline cases</span>
        <strong>{borderline}</strong>
        <small>mean confidence: {escape(avg_confidence)}</small>
      </article>
    </section>
    """


def single_case_html(predictions: pd.DataFrame) -> str:
    if len(predictions) != 1:
        return ""
    row = predictions.iloc[0]
    warning = (
        "<p class='borderline-warning'>This result falls near the decision boundary and should be interpreted cautiously.</p>"
        if bool(row["borderline_flag"])
        else ""
    )
    return f"""
    <section class="single-case-card">
      <div>
        <span class="single-kicker">Single-sample interpretation</span>
        <h3>{escape(str(row['SAMPLE_ID']))}</h3>
        <p>{escape(str(row['interpretation']))}</p>
        {warning}
      </div>
      <div class="single-meter {'mutant' if row['predicted_label'] == 'IDH-mutant' else 'wildtype'}">
        <strong>{float(row['IDH_mutation_probability']) * 100:.1f}%</strong>
        <span>IDH-mutant probability</span>
      </div>
    </section>
    """


def pathway_overview_html(predictions: pd.DataFrame | None = None) -> str:
    predictions = predictions if predictions is not None else pd.DataFrame()
    state_class = "neutral"
    state_title = "Pathway learning card"
    state_text = (
        "Explore the canonical IDH-mutant cascade below. Once you run prediction, this card will add a class-aware "
        "research note linked to the uploaded sample or cohort."
    )
    driver_title = "Global genes to inspect"
    driver_note = (
        "These globally influential model genes can help frame the saved classifier, but they are not a substitute "
        "for direct mechanistic validation."
    )
    driver_genes: list[str] = []

    if not predictions.empty:
        if len(predictions) == 1:
            row = predictions.iloc[0]
            label = str(row["predicted_label"])
            confidence = str(row["confidence_level"])
            probability = float(row["IDH_mutation_probability"])
            if label == "IDH-mutant":
                state_class = "mutant"
                state_title = f"{escape(str(row['SAMPLE_ID']))}: mutant-leaning expression signal"
                state_text = (
                    f"The uploaded sample is predicted as IDH-mutant with {escape(confidence.lower())}. "
                    f"Its mutant probability is {probability * 100:.1f}%, so the pathway below is highlighted as the "
                    "more compatible transcriptional program."
                )
            else:
                state_class = "wildtype"
                state_title = f"{escape(str(row['SAMPLE_ID']))}: wildtype-leaning expression signal"
                state_text = (
                    f"The uploaded sample is predicted as IDH-wildtype with {escape(confidence.lower())}. "
                    f"Its mutant probability is {probability * 100:.1f}%, so the mutant pathway below should be read "
                    "as biological context rather than the dominant inferred program."
                )
            driver_title = "Class-aligned global driver genes"
            driver_genes = class_driver_genes(label, limit=5)
        else:
            mutant_count = int((predictions["predicted_label"] == "IDH-mutant").sum())
            wildtype_count = int((predictions["predicted_label"] == "IDH-wildtype").sum())
            mean_probability = float(predictions["IDH_mutation_probability"].mean())
            if mutant_count >= wildtype_count:
                state_class = "mutant"
                state_title = "Cohort signal leans toward IDH-mutant biology"
                state_text = (
                    f"{mutant_count} of {len(predictions)} samples were predicted IDH-mutant. "
                    f"The mean mutant probability across the batch is {mean_probability * 100:.1f}%."
                )
                driver_genes = class_driver_genes("IDH-mutant", limit=5)
            else:
                state_class = "wildtype"
                state_title = "Cohort signal leans toward IDH-wildtype biology"
                state_text = (
                    f"{wildtype_count} of {len(predictions)} samples were predicted IDH-wildtype. "
                    f"The mean mutant probability across the batch is {mean_probability * 100:.1f}%."
                )
                driver_genes = class_driver_genes("IDH-wildtype", limit=5)
            driver_title = "Class-aligned global driver genes"

    driver_markup = (
        "".join(f"<span>{escape(gene)}</span>" for gene in driver_genes)
        if driver_genes
        else "<span>IDH1</span><span>IDH2</span><span>MGMT</span><span>PDGFRA</span>"
    )

    return f"""
    <section class="pathway-knowledge {state_class}">
      <div class="pathway-overview-card">
        <div class="section-heading pathway-section-heading">
          <div class="section-icon">PW</div>
          <div>
            <h2>Interactive IDH pathway overview</h2>
            <p>Mutation, metabolism, epigenetics, transcription, and glioma phenotype in one view.</p>
          </div>
        </div>
        <div class="pathway-state-card {state_class}">
          <div>
            <span class="pathway-state-kicker">Current model context</span>
            <h3>{state_title}</h3>
            <p>{state_text}</p>
          </div>
        </div>
        <div class="pathway-flow-advanced">
          <details class="pathway-node mutation" open>
            <summary><span class="pathway-node-icon">A</span><span><b>IDH1 / IDH2 mutation</b><small>Early molecular event in diffuse glioma classification.</small></span></summary>
            <div class="pathway-detail">Mutations in IDH1 or IDH2 alter the enzyme's normal catalytic behavior and create a biologically distinct glioma subgroup.</div>
          </details>
          <div class="pathway-connector"></div>
          <details class="pathway-node enzyme" open>
            <summary><span class="pathway-node-icon">B</span><span><b>Mutant IDH enzyme activity</b><small>The altered enzyme gains a new metabolic reaction.</small></span></summary>
            <div class="pathway-detail">Instead of supporting normal isocitrate metabolism, mutant IDH converts alpha-ketoglutarate-related substrates into abnormal metabolic output.</div>
          </details>
          <div class="pathway-connector"></div>
          <details class="pathway-node metabolite" open>
            <summary><span class="pathway-node-icon">C</span><span><b>2-Hydroxyglutarate (2-HG) accumulation</b><small>An oncometabolite builds up inside the tumor.</small></span></summary>
            <div class="pathway-detail">2-HG is a hallmark consequence of mutant IDH and is often used conceptually to explain how metabolism can reshape cancer cell identity.</div>
          </details>
          <div class="pathway-connector"></div>
          <details class="pathway-node dioxygenase" open>
            <summary><span class="pathway-node-icon">D</span><span><b>Dioxygenase inhibition</b><small>Alpha-ketoglutarate-dependent enzymes become suppressed.</small></span></summary>
            <div class="pathway-detail">2-HG can interfere with enzymes involved in chromatin regulation and DNA demethylation, linking metabolism to epigenetic control.</div>
          </details>
          <div class="pathway-connector"></div>
          <details class="pathway-node epigenetic" open>
            <summary><span class="pathway-node-icon">E</span><span><b>DNA / histone hypermethylation</b><small>Epigenetic state shifts toward abnormal methylation patterns.</small></span></summary>
            <div class="pathway-detail">These methylation changes can alter developmental programs, cell-state identity, and broad transcriptional regulation in glioma.</div>
          </details>
          <div class="pathway-connector"></div>
          <details class="pathway-node expression" open>
            <summary><span class="pathway-node-icon">F</span><span><b>Altered gene expression</b><small>Downstream transcriptional programs change in measurable ways.</small></span></summary>
            <div class="pathway-detail">Expression-based machine-learning models take advantage of these broad transcriptional differences to distinguish mutant-like from wildtype-like tumors.</div>
          </details>
          <div class="pathway-connector"></div>
          <details class="pathway-node outcome" open>
            <summary><span class="pathway-node-icon">G</span><span><b>Glioma biology and outcome</b><small>Diagnosis, WHO class, prognosis, and management are affected.</small></span></summary>
            <div class="pathway-detail">IDH status helps define major glioma categories and is routinely interpreted together with histology and additional molecular findings.</div>
          </details>
        </div>
        <div class="pathway-driver-card">
          <div>
            <span class="pathway-state-kicker">Research interpretation</span>
            <h4>{driver_title}</h4>
            <p>{driver_note}</p>
          </div>
          <div class="driver-chip-row">{driver_markup}</div>
        </div>
      </div>
      <div class="pathway-mini-grid">
        <article class="mini-edu-card">
          <h3>What is IDH?</h3>
          <p>IDH1 and IDH2 encode metabolic enzymes. In glioma, their mutation status is a major molecular classifier rather than a cosmetic genomic detail.</p>
        </article>
        <article class="mini-edu-card">
          <h3>What is 2-HG?</h3>
          <p>2-hydroxyglutarate is an oncometabolite produced by mutant IDH enzymes. It links altered metabolism to chromatin and transcriptional change.</p>
        </article>
        <article class="mini-edu-card">
          <h3>What do epigenetic alterations mean?</h3>
          <p>Epigenetic changes affect how genes are regulated without changing the DNA sequence itself, often through methylation and chromatin remodeling.</p>
        </article>
        <article class="mini-edu-card">
          <h3>Why does IDH status matter?</h3>
          <p>IDH status is tied to WHO classification, prognosis, and how gliomas are biologically interpreted in modern neuro-oncology.</p>
        </article>
        <article class="mini-edu-card">
          <h3>Why can gene expression help prediction?</h3>
          <p>Gene-expression profiles capture the downstream consequences of mutation, metabolism, and epigenetics, giving the model a systems-level molecular signal.</p>
        </article>
      </div>
    </section>
    """


def results_table_html(predictions: pd.DataFrame) -> str:
    if predictions.empty:
        return """
        <section class="results-empty">
          <div class="empty-icon">RS</div>
          <h3>Results will appear here</h3>
          <p>Run classification on a compatible expression matrix to populate this table with sample-level IDH status, confidence score, and interpretive notes.</p>
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
            f"<div class='probability-track {klass}'><i style='width:{max(0, min(100, probability * 100)):.1f}%'></i></div></div></td>"
            f"<td class='interpretation-cell'>{escape(str(row['interpretation']))}</td>"
            "</tr>"
        )

    return f"""
    <section class="results-shell">
      <div class="results-toolbar-copy">
        <span class="results-toolbar-badge">Sample-level classification</span>
        <p>Scroll horizontally if needed. The table summarizes class assignment, confidence, probability, and interpretation for each sample.</p>
      </div>
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
    </section>
    """


def top_genes_html(limit: int = 12) -> str:
    genes = top_genes(limit)
    if genes.empty:
        return "<section class='top-genes-empty'><p>Global feature rankings are not available.</p></section>"
    highlight = genes.head(5)["gene"].astype(str).tolist()
    max_importance = max(float(genes["importance"].abs().max()), 1e-9)
    max_signed = max(float(genes["signed_value"].abs().max()), 1e-9)
    rows = "".join(
        (
            f"<tr class='gene-row {'positive' if float(row.signed_value) > 0 else 'negative' if float(row.signed_value) < 0 else 'neutral'}'>"
            f"<td class='gene-name-cell'><div class='gene-name-wrap'><span class='gene-dot {'positive' if float(row.signed_value) > 0 else 'negative' if float(row.signed_value) < 0 else 'neutral'}'></span><strong>{escape(str(row.gene))}</strong></div></td>"
            f"<td class='metric-cell metric-cell-importance'><div class='metric-stack'><span class='metric-number'>{float(row.importance):.4f}</span><div class='metric-bar-track'><i class='metric-bar importance' style='width:{(abs(float(row.importance)) / max_importance) * 100:.1f}%'></i></div></div></td>"
            f"<td class='metric-cell metric-cell-signed'><div class='metric-stack'><span class='metric-number signed {'positive' if float(row.signed_value) > 0 else 'negative' if float(row.signed_value) < 0 else 'neutral'}'>{float(row.signed_value):+.4f}</span><div class='metric-bar-track'><i class='metric-bar {'positive' if float(row.signed_value) > 0 else 'negative' if float(row.signed_value) < 0 else 'neutral'}' style='width:{(abs(float(row.signed_value)) / max_signed) * 100:.1f}%'></i></div></div></td>"
            f"<td class='direction-cell'><span class='direction-pill {'positive' if float(row.signed_value) > 0 else 'negative' if float(row.signed_value) < 0 else 'neutral'}'>{'Positive' if float(row.signed_value) > 0 else 'Negative' if float(row.signed_value) < 0 else 'Neutral'}</span></td>"
            "</tr>"
        )
        for row in genes.itertuples(index=False)
    )
    chips = "".join(f"<span>{escape(gene)}</span>" for gene in highlight)
    return f"""
    <div class="interpretability-note">
      <p>These globally weighted model features summarize the preserved classifier. They are not causal biomarkers and they are not sample-specific explanations.</p>
    </div>
    <div class="driver-chip-row driver-chip-row-large">
      {chips}
    </div>
    <div class="top-genes-toolbar">
      <span class="top-genes-badge">Global feature ranking</span>
      <p>Higher-magnitude values indicate stronger overall contribution within the preserved classification pipeline.</p>
    </div>
    <div class="mini-table-wrap top-genes-table-wrap">
      <table class="mini-table top-genes-table">
        <thead><tr><th>Gene</th><th>Importance</th><th>Contribution</th><th>Direction</th></tr></thead>
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
      <div><span>Genes / features</span><strong>{info.get('genes', 'Unavailable')}</strong></div>
      <div><span>Test ROC AUC</span><strong>{pct(info.get('roc_auc'))}</strong></div>
      <div><span>Test balanced accuracy</span><strong>{pct(info.get('balanced_accuracy'))}</strong></div>
      <div><span>Test F1</span><strong>{pct(info.get('f1'))}</strong></div>
    </div>
    <p class="model-disclaimer">This research model was trained to classify glioma IDH status from gene-expression data.</p>
    """
