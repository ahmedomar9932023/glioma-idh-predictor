from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_PACKAGES = PROJECT_ROOT / ".venv_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import gradio as gr
import pandas as pd

from web.charts import batch_charts_html
from web.components import loading_html, model_info_html, ready_html, results_table_html, top_genes_html
from web.metadata import TOP_GENES_PATH
from web.predictor import filter_predictions, run_prediction_for_app
from web.settings import DEFAULT_DECISION_THRESHOLD


CSS_PATH = PROJECT_ROOT / "web" / "styles.css"
EXAMPLE_PATH = PROJECT_ROOT / "data" / "examples" / "sample_expression_input.csv"
APP_JS = """
() => {
  const applyTheme = (mode) => {
    const theme = mode === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", theme);
    document.body.setAttribute("data-theme", theme);
    localStorage.setItem("idh-theme", theme);
  };
  const saved = localStorage.getItem("idh-theme") || "light";
  applyTheme(saved);
  setTimeout(() => {
    const toggle = document.querySelector("#theme-toggle input");
    if (toggle) toggle.checked = saved === "dark";
  }, 250);
  let inferenceTimer = null;
  const startInferenceTimer = () => {
    const started = performance.now();
    if (inferenceTimer) clearInterval(inferenceTimer);
    inferenceTimer = setInterval(() => {
      const target = document.querySelector(".loading-state .elapsed-time");
      if (!document.querySelector(".loading-state")) {
        clearInterval(inferenceTimer);
        inferenceTimer = null;
        return;
      }
      if (target) {
        const seconds = ((performance.now() - started) / 1000).toFixed(1);
        target.textContent = `Processing... ${seconds}s`;
      }
    }, 100);
  };
  document.addEventListener("click", (event) => {
    if (event.target.closest("#predict-button")) startInferenceTimer();
  });
}
"""

THEME_TOGGLE_JS = """
(darkMode) => {
  const theme = darkMode ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", theme);
  document.body.setAttribute("data-theme", theme);
  localStorage.setItem("idh-theme", theme);
  return [];
}
"""


def load_css() -> str:
    return CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Glioma IDH Mutation Prediction",
        css=load_css(),
        js=APP_JS,
        fill_width=True,
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
    ) as demo:
        with gr.Column(elem_id="app-shell"):
            with gr.Row(elem_id="topbar"):
                gr.HTML(
                    """
                    <button class="brand-lockup" type="button" aria-label="Return to app home" onclick="window.location.reload()">
                      <div class="brand-mark">IDH</div>
                      <div>
                        <div class="brand-name">GliomaIDH AI</div>
                        <div class="brand-subtitle">Gene expression mutation-status prediction</div>
                      </div>
                      <span class="research-badge">Research Demo</span>
                    </button>
                    """
                )
                theme_toggle = gr.Checkbox(label="Toggle theme", value=False, elem_id="theme-toggle")

            gr.HTML(
                """
                <section class="hero-product">
                  <div class="hero-copy">
                    <div class="eyebrow">Expression-based molecular insight</div>
                    <h1>Predict IDH mutation status from glioma expression profiles.</h1>
                    <p>
                      Upload a cBioPortal-style matrix or sample-by-gene CSV and run the existing saved
                      machine-learning pipeline in a clean, deployment-ready interface.
                    </p>
                    <div class="hero-tags">
                      <span>Saved model pipeline</span>
                      <span>Confidence-aware output</span>
                      <span>CSV predictions</span>
                      <span>No retraining</span>
                    </div>
                  </div>
                  <div class="hero-orbit" aria-hidden="true">
                    <div class="orbit-card main">IDH1/2</div>
                    <div class="orbit-card small a">RNA-seq</div>
                    <div class="orbit-card small b">SVM</div>
                  </div>
                </section>
                """
            )

            with gr.Row(elem_id="workspace-row"):
                with gr.Column(scale=5, elem_classes=["product-card", "upload-card"]):
                    gr.HTML(
                        """
                        <div class="section-heading">
                          <div class="section-icon">UP</div>
                          <div>
                            <h2>Upload expression data</h2>
                            <p>Use a TSV/TXT gene-by-sample matrix or CSV sample-by-gene table.</p>
                          </div>
                        </div>
                        """
                    )
                    input_file = gr.File(
                        label="Expression file",
                        file_types=[".csv", ".tsv", ".txt"],
                        type="filepath",
                        elem_id="upload-box",
                    )
                    gr.HTML(
                        """
                        <div class="input-hints">
                          <div><strong>cBioPortal TSV/TXT</strong><span>requires Hugo_Symbol and Entrez_Gene_Id</span></div>
                          <div><strong>Sample CSV</strong><span>requires SAMPLE_ID plus gene columns</span></div>
                        </div>
                        """
                    )
                    with gr.Accordion("Advanced settings", open=False, elem_classes=["settings-panel"]):
                        threshold = gr.Slider(
                            minimum=0.05,
                            maximum=0.95,
                            value=DEFAULT_DECISION_THRESHOLD,
                            step=0.01,
                            label="IDH-mutant decision threshold",
                            info="Default is 0.50. Raising it makes IDH-mutant calls more conservative.",
                        )
                    predict_button = gr.Button("Predict IDH Status", variant="primary", size="lg", elem_id="predict-button")

                with gr.Column(scale=4, elem_classes=["product-card", "summary-card"]):
                    gr.HTML(
                        """
                        <div class="section-heading">
                          <div class="section-icon">AI</div>
                          <div>
                            <h2>Prediction summary</h2>
                            <p>Live cohort overview after inference completes.</p>
                          </div>
                        </div>
                        """
                    )
                    status = gr.HTML(ready_html(), elem_id="status-panel")

            single_case = gr.HTML("", elem_id="single-case-panel")

            with gr.Row(elem_id="charts-row"):
                with gr.Column(elem_classes=["product-card", "charts-card"]):
                    gr.HTML(
                        """
                        <div class="section-heading">
                          <div class="section-icon">DB</div>
                          <div>
                            <h2>Batch dashboard</h2>
                            <p>Visual summary of class balance and probability distribution.</p>
                          </div>
                        </div>
                        """
                    )
                    charts = gr.HTML(batch_charts_html(pd.DataFrame()), elem_id="charts-panel")

            with gr.Row(elem_id="results-row"):
                with gr.Column(scale=7, elem_classes=["product-card", "results-card"]):
                    gr.HTML(
                        """
                        <div class="section-heading">
                          <div class="section-icon">RS</div>
                          <div>
                            <h2>Results</h2>
                            <p>Predicted class and IDH-mutant probability for each uploaded sample.</p>
                          </div>
                        </div>
                        """
                    )
                    with gr.Row(elem_id="filter-row"):
                        label_filter = gr.Dropdown(
                            ["All predictions", "IDH-mutant", "IDH-wildtype"],
                            value="All predictions",
                            label="Prediction filter",
                        )
                        confidence_filter = gr.Dropdown(
                            ["All confidence levels", "High confidence", "Moderate confidence", "Borderline / uncertain"],
                            value="All confidence levels",
                            label="Confidence filter",
                        )
                        sample_search = gr.Textbox(label="Sample search", placeholder="Search SAMPLE_ID")
                    results = gr.HTML(results_table_html(pd.DataFrame()), elem_id="results-panel")

                with gr.Column(scale=3, elem_classes=["product-card", "download-card"]):
                    gr.HTML(
                        """
                        <div class="section-heading compact">
                          <div class="section-icon">CSV</div>
                          <div>
                            <h2>Export</h2>
                            <p>Download the complete prediction file.</p>
                          </div>
                        </div>
                        """
                    )
                    download = gr.File(label="Predictions CSV", interactive=False, elem_id="download-file")
                    report_download = gr.File(label="HTML Summary Report", interactive=False, elem_id="report-file")
                    export_note = gr.Markdown("No prediction file generated.", elem_id="export-note")
                    gr.HTML(
                        """
                        <div class="column-guide">
                          <h3>Columns</h3>
                          <p><b>SAMPLE_ID</b> uploaded sample identifier</p>
                          <p><b>predicted</b> IDH-mutant or IDH-wildtype</p>
                          <p><b>IDH_mutation_probability</b> model score for IDH-mutant class</p>
                        </div>
                        """
                    )

            with gr.Row(elem_id="info-row"):
                with gr.Column(scale=1, elem_classes=["product-card", "helper-card"]):
                    with gr.Accordion("Input format guide", open=True):
                        gr.HTML(
                            """
                            <div class="guide-copy">
                              <p><b>Accepted file types:</b> .csv, .tsv, .txt</p>
                              <p><b>cBioPortal style:</b> rows are genes, columns are samples, and the first columns include Hugo_Symbol and Entrez_Gene_Id.</p>
                              <p><b>Sample CSV style:</b> rows are samples, one column must be SAMPLE_ID, and gene-expression columns should be numeric.</p>
                              <p><b>Extra columns:</b> extra columns are ignored if they are not model genes. Required model genes must be present.</p>
                              <p><b>Validation:</b> duplicate sample IDs, blank IDs, malformed files, and non-numeric required gene columns are reported before prediction.</p>
                            </div>
                            """
                        )
                    with gr.Accordion("How it works", open=False):
                        gr.HTML(
                            """
                            <div class="guide-copy">
                              <p><b>Input:</b> upload gene expression data for one or more glioma samples.</p>
                              <p><b>Processing:</b> the existing saved pipeline aligns model genes, applies fitted preprocessing, and scores each sample.</p>
                              <p><b>Output:</b> predicted IDH status, probability, confidence category, interpretation text, borderline flag, charts, and downloadable files.</p>
                            </div>
                            """
                        )

                with gr.Column(scale=1, elem_classes=["product-card", "model-card"]):
                    with gr.Accordion("About the model", open=True):
                        gr.HTML(model_info_html())
                    with gr.Accordion("Global interpretability", open=False):
                        gr.HTML(top_genes_html(15), elem_id="top-genes-panel")
                        if TOP_GENES_PATH.exists():
                            gr.File(value=str(TOP_GENES_PATH), label="Download ranked top genes", interactive=False)

            if EXAMPLE_PATH.exists():
                with gr.Row(elem_id="demo-row"):
                    gr.Examples(
                        examples=[[str(EXAMPLE_PATH)]],
                        inputs=[input_file],
                        label="Try the included demo file",
                        elem_id="examples-panel",
                    )
                    gr.File(value=str(EXAMPLE_PATH), label="Download example input", interactive=False, elem_id="example-download")

            gr.HTML(
                """
                <footer class="app-footer">
                  <span>Research/demo tool only. Not intended for clinical diagnosis or treatment decisions.</span>
                  <span>Built on the preserved saved model pipeline.</span>
                </footer>
                """
            )

            theme_toggle.change(
                fn=None,
                inputs=theme_toggle,
                outputs=[],
                js=THEME_TOGGLE_JS,
                show_progress="hidden",
            )

            predictions_state = gr.State(pd.DataFrame())
            predict_event = predict_button.click(
                fn=loading_html,
                inputs=None,
                outputs=status,
                queue=False,
                show_progress="hidden",
            )
            predict_event.then(
                fn=run_prediction_for_app,
                inputs=[input_file, threshold],
                outputs=[status, single_case, charts, results, download, report_download, predictions_state, export_note],
                show_progress="hidden",
            )
            for component in (label_filter, confidence_filter, sample_search):
                component.change(
                    fn=filter_predictions,
                    inputs=[predictions_state, label_filter, confidence_filter, sample_search],
                    outputs=results,
                    show_progress="hidden",
                )

    return demo


if __name__ == "__main__":
    build_app().launch()
