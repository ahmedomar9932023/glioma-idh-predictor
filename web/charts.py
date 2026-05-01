from __future__ import annotations

from html import escape

import pandas as pd

from .metadata import class_driver_genes


def _batch_interpretation(predictions: pd.DataFrame) -> tuple[str, str, list[str]]:
    mutant = int((predictions["predicted_label"] == "IDH-mutant").sum())
    wildtype = int((predictions["predicted_label"] == "IDH-wildtype").sum())
    mean_probability = float(predictions["IDH_mutation_probability"].mean())
    if mutant >= wildtype:
        title = "Cohort interpretation leans toward IDH-mutant biology"
        body = (
            f"Most samples in this cohort were classified as IDH-mutant-like, with a mean mutant probability of "
            f"{mean_probability * 100:.1f}%. This pattern is consistent with transcriptomic signals linked to altered "
            "metabolism and downstream epigenetic remodeling."
        )
        genes = class_driver_genes("IDH-mutant", limit=5)
    else:
        title = "Cohort interpretation leans toward IDH-wildtype biology"
        body = (
            f"Most samples in this cohort were classified as IDH-wildtype-like, with a mean mutant probability of "
            f"{mean_probability * 100:.1f}%. This pattern suggests a sample set whose transcriptomic profile trends away "
            "from the canonical IDH-mutant metabolic and methylation program."
        )
        genes = class_driver_genes("IDH-wildtype", limit=5)
    return title, body, genes


def batch_charts_html(predictions: pd.DataFrame) -> str:
    if predictions.empty:
        return """
        <section class="chart-empty batch-dashboard-empty">
          <div class="empty-icon">DB</div>
          <h3>Cohort analytics will appear here</h3>
          <p>Run classification on one or more samples to populate this panel with cohort-level summaries and molecular interpretation.</p>
        </section>
        """

    total = max(len(predictions), 1)
    mutant = int((predictions["predicted_label"] == "IDH-mutant").sum())
    wildtype = int((predictions["predicted_label"] == "IDH-wildtype").sum())
    borderline_count = int(predictions["borderline_flag"].sum())
    mean_probability = float(predictions["IDH_mutation_probability"].mean())
    high_conf_count = int((predictions["confidence_level"] == "High confidence").sum())
    dominant_label = "IDH-mutant" if mutant >= wildtype else "IDH-wildtype"
    dominant_share = (max(mutant, wildtype) / total) * 100
    high_conf_pct = (high_conf_count / total) * 100

    recurring_genes = (
        predictions["predicted_label"]
        .map(lambda label: class_driver_genes(str(label), limit=3))
        .explode()
        .dropna()
        .astype(str)
        .value_counts()
        .head(5)
    )
    recurring_markup = (
        "".join(
            f"<span><strong>{escape(gene)}</strong><em>{count} sample-associated occurrences</em></span>"
            for gene, count in recurring_genes.items()
        )
        if not recurring_genes.empty
        else "<span><strong>Global driver genes unavailable</strong><em>No recurring cohort pattern was identified.</em></span>"
    )

    title, interpretation, genes = _batch_interpretation(predictions)
    gene_markup = "".join(f"<span>{escape(gene)}</span>" for gene in genes) if genes else "<span>Global genes unavailable</span>"

    insight_cards = f"""
      <div class="batch-insight-grid batch-insight-grid-compact">
        <article class="batch-insight-card">
          <span class="batch-insight-kicker">Dominant cohort class</span>
          <strong>{dominant_share:.1f}% {escape(dominant_label)}</strong>
          <p>The current cohort trends toward <b>{escape(dominant_label.lower())}</b>, providing an initial view of the dominant molecular profile.</p>
        </article>
        <article class="batch-insight-card">
          <span class="batch-insight-kicker">High-confidence fraction</span>
          <strong>{high_conf_pct:.1f}% of samples</strong>
          <p>Most classifications are {'well separated from' if high_conf_pct >= 60 else 'closer to'} the decision boundary.</p>
        </article>
        <article class="batch-insight-card">
          <span class="batch-insight-kicker">Borderline cases</span>
          <strong>{borderline_count} sample{'s' if borderline_count != 1 else ''}</strong>
          <p>{borderline_count} sample{'s fall' if borderline_count != 1 else ' falls'} near the uncertainty threshold and may warrant closer review.</p>
        </article>
      </div>
    """

    return f"""
    <section class="batch-analytics-shell">
      {insight_cards}

      <div class="batch-analytics-focus">
        <article class="batch-interpretation-card">
          <div class="batch-panel-heading">
            <h3>{escape(title)}</h3>
            <p>{escape(interpretation)}</p>
          </div>
          <div class="driver-chip-row batch-driver-chips">
            {gene_markup}
          </div>
        </article>

        <article class="batch-panel batch-recurring-panel">
          <div class="batch-panel-heading">
            <h3>Recurring molecular signals</h3>
            <p>Global driver genes that repeatedly align with the cohort's dominant class context.</p>
          </div>
          <div class="batch-recurring-gene-list">
            {recurring_markup}
          </div>
          <p class="batch-panel-footnote">
            Mean IDH-mutant probability across the cohort is <strong>{mean_probability * 100:.1f}%</strong>, which helps contextualize how strongly the sample set leans toward one molecular state.
          </p>
        </article>
      </div>
    </section>
    """
