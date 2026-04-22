from __future__ import annotations

import pandas as pd


def batch_charts_html(predictions: pd.DataFrame) -> str:
    if predictions.empty:
        return """
        <section class="chart-empty">
          <p>Charts will appear after a prediction run.</p>
        </section>
        """

    counts = predictions["predicted_label"].value_counts().to_dict()
    mutant = int(counts.get("IDH-mutant", 0))
    wildtype = int(counts.get("IDH-wildtype", 0))
    total = max(mutant + wildtype, 1)
    mutant_pct = mutant / total * 100
    wildtype_pct = wildtype / total * 100

    bins = pd.cut(
        predictions["IDH_mutation_probability"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0000001],
        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
        include_lowest=True,
    )
    hist = bins.value_counts().sort_index()
    max_count = max(int(hist.max()), 1)
    bars = "".join(
        f"""
        <div class="hist-bin">
          <div class="hist-bar" style="height:{(int(count) / max_count) * 100:.1f}%"></div>
          <span>{label}</span>
          <b>{int(count)}</b>
        </div>
        """
        for label, count in hist.items()
    )

    return f"""
    <section class="charts-grid">
      <article class="chart-card">
        <h3>Predicted class distribution</h3>
        <div class="stacked-bar" aria-label="Mutant and wildtype counts">
          <span class="stack-mutant" style="width:{mutant_pct:.1f}%"></span>
          <span class="stack-wildtype" style="width:{wildtype_pct:.1f}%"></span>
        </div>
        <div class="chart-legend">
          <span><i class="legend-mutant"></i>IDH-mutant: {mutant}</span>
          <span><i class="legend-wildtype"></i>IDH-wildtype: {wildtype}</span>
        </div>
      </article>
      <article class="chart-card">
        <h3>Probability distribution</h3>
        <div class="histogram">{bars}</div>
      </article>
    </section>
    """

