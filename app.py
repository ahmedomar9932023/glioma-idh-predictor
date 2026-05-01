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
from web.components import (
    loading_html,
    model_info_html,
    ready_html,
    results_table_html,
    top_genes_html,
)
from web.metadata import TOP_GENES_PATH
from web.predictor import run_prediction_for_app
from web.settings import DEFAULT_DECISION_THRESHOLD


CSS_PATH = PROJECT_ROOT / "web" / "styles.css"
APP_JS = """
() => {
  document.documentElement.setAttribute("data-theme", "light");
  document.body.setAttribute("data-theme", "light");
  document.addEventListener("click", (event) => {
    if (event.target.closest("[data-open-model-drawer]")) {
      document.body.classList.add("model-drawer-open");
    }
    if (event.target.closest("[data-close-model-drawer]") || event.target.closest(".model-drawer-backdrop")) {
      document.body.classList.remove("model-drawer-open");
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      document.body.classList.remove("model-drawer-open");
    }
  });
}
"""

def load_css() -> str:
    return CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Glioma IDH Molecular Classification",
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
                        <div class="brand-name">GliomaIDH</div>
                        <div class="brand-subtitle">Expression-based IDH status classification</div>
                      </div>
                      <span class="research-badge">Research Interface</span>
                    </button>
                    """
                )

            gr.HTML(
                """
                <section class="hero-product hero-science">
                  <div class="hero-copy hero-copy-science">
                    <div class="eyebrow">Understanding IDH Mutation in Glioma</div>
                    <h1>Molecular Biology of IDH-Mutant Glioma</h1>
                    <p class="hero-subtitle">
                      A research-focused overview of how <span class="text-accent text-idh">IDH1</span>/<span class="text-accent text-idh">IDH2</span> mutations reshape glioma biology
                      through <span class="text-accent text-metabolite">2-hydroxyglutarate</span>, <span class="text-accent text-epigenetic">epigenetic reprogramming</span>, and <span class="text-accent text-expression">gene-expression changes</span>.
                    </p>
                    <div class="hero-body-grid">
                      <div class="hero-body-copy">
                        <article class="hero-info-block">
                          <div class="hero-info-icon info-icon-blue">GL</div>
                          <div class="hero-info-text">
                            <h3>WHO molecular classification</h3>
                            <p>
                              Gliomas are primary brain tumors that arise from glial-lineage cells and are now classified
                              not only by microscopic appearance, but also by molecular features. Adult-type diffuse gliomas
                              are mainly separated into <b>astrocytoma, IDH-mutant</b>; <b>oligodendroglioma, IDH-mutant and 1p/19q-codeleted</b>;
                              and <b>glioblastoma, IDH-wildtype</b>.
                            </p>
                          </div>
                        </article>
                        <article class="hero-info-block">
                          <div class="hero-info-icon info-icon-cyan">2H</div>
                          <div class="hero-info-text">
                            <h3>Mutant metabolism</h3>
                            <p>
                              One of the most important molecular events in glioma is mutation of <span class="text-accent text-idh"><b>IDH1</b></span> or <span class="text-accent text-idh"><b>IDH2</b></span>.
                              Mutant IDH enzymes gain a new abnormal function: they convert <b>&alpha;-ketoglutarate</b> into
                              <span class="text-accent text-metabolite"><b>2-hydroxyglutarate (2-HG)</b></span>, an oncometabolite that perturbs downstream regulation.
                            </p>
                          </div>
                        </article>
                        <article class="hero-info-block">
                          <div class="hero-info-icon info-icon-purple">ME</div>
                          <div class="hero-info-text">
                            <h3>Epigenetic reprogramming</h3>
                            <p>
                              Because <span class="text-accent text-metabolite">2-HG</span> interferes with &alpha;-ketoglutarate-dependent enzymes, it can disrupt DNA and
                              histone demethylation, leading to widespread <span class="text-accent text-epigenetic">epigenetic reprogramming</span> and shifts in chromatin state.
                            </p>
                          </div>
                        </article>
                        <article class="hero-info-block">
                          <div class="hero-info-icon info-icon-green">GE</div>
                          <div class="hero-info-text">
                            <h3>Transcriptional identity</h3>
                            <p>
                              These molecular changes alter <span class="text-accent text-expression">gene-expression</span> programs inside tumor cells. The resulting transcriptomic
                              patterns help explain differences in tumor behavior, prognosis, and potential treatment response across glioma subtypes.
                            </p>
                          </div>
                        </article>
                        <article class="hero-info-block">
                          <div class="hero-info-icon info-icon-gold">AI</div>
                          <div class="hero-info-text">
                            <h3>Research workflow</h3>
                            <p>
                              This application uses <span class="text-accent text-expression">gene-expression</span> profiles together with a trained <span class="text-accent text-ml">machine-learning</span> model
                              to predict IDH mutation status. It is designed as an interpretable research workflow rather than a replacement for clinical diagnosis.
                            </p>
                          </div>
                        </article>
                      </div>
                      <aside class="hero-side-panel" aria-label="Molecular pathway summary">
                        <div class="hero-side-panel-inner">
                          <div class="hero-side-kicker">Molecular pathway summary</div>
                          <h3>IDH-driven molecular cascade</h3>
                          <p>
                            A compact view of how mutant metabolism, chromatin remodeling, and transcriptomic shifts connect
                            molecular glioma biology to expression-based classification.
                          </p>
                          <div class="hero-cascade">
                            <div class="hero-cascade-step">
                              <span class="cascade-dot cascade-idh"></span>
                              <div>
                                <strong>IDH mutation</strong>
                                <span>Mutant <span class="text-accent text-idh">IDH1/IDH2</span> alters enzyme function in glioma cells.</span>
                              </div>
                              <div class="cascade-icon">ID</div>
                            </div>
                            <div class="hero-cascade-arrow">&rarr;</div>
                            <div class="hero-cascade-step">
                              <span class="cascade-dot cascade-metabolite"></span>
                              <div>
                                <strong>2-HG accumulation</strong>
                                <span><span class="text-accent text-metabolite">2-HG</span> acts as an oncometabolite and rewires metabolism.</span>
                              </div>
                              <div class="cascade-icon">2H</div>
                            </div>
                            <div class="hero-cascade-arrow">&rarr;</div>
                            <div class="hero-cascade-step">
                              <span class="cascade-dot cascade-epigenetic"></span>
                              <div>
                                <strong>Epigenetic shift</strong>
                                <span><span class="text-accent text-epigenetic">Epigenetic reprogramming</span> changes chromatin state.</span>
                              </div>
                              <div class="cascade-icon">ME</div>
                            </div>
                            <div class="hero-cascade-arrow">&rarr;</div>
                            <div class="hero-cascade-step">
                              <span class="cascade-dot cascade-expression"></span>
                              <div>
                                <strong>Altered gene expression</strong>
                                <span><span class="text-accent text-expression">Gene-expression</span> programs shift across tumor cells.</span>
                              </div>
                              <div class="cascade-icon">GE</div>
                            </div>
                            <div class="hero-cascade-arrow">&rarr;</div>
                            <div class="hero-cascade-step">
                              <span class="cascade-dot cascade-ml"></span>
                              <div>
                                <strong>Expression-based classification</strong>
                                <span><span class="text-accent text-ml">Computational modeling</span> maps transcriptomic signatures to IDH status.</span>
                              </div>
                              <div class="cascade-icon">AI</div>
                            </div>
                          </div>
                        </div>
                      </aside>
                    </div>
                    <div class="pathway-mini-grid hero-edu-grid">
                      <article class="mini-edu-card hero-edu-card hero-edu-card-1">
                        <span class="hero-edu-kicker">1</span>
                        <h3>IDH Mutation</h3>
                        <p>IDH1/IDH2 mutations create an abnormal enzyme activity that produces 2-hydroxyglutarate.</p>
                      </article>
                      <article class="mini-edu-card hero-edu-card hero-edu-card-2">
                        <span class="hero-edu-kicker">2</span>
                        <h3>2-HG Accumulation</h3>
                        <p>2-hydroxyglutarate acts as an oncometabolite and interferes with epigenetic enzymes.</p>
                      </article>
                      <article class="mini-edu-card hero-edu-card hero-edu-card-3">
                        <span class="hero-edu-kicker">3</span>
                        <h3>Epigenetic Reprogramming</h3>
                        <p>DNA and histone methylation patterns shift, changing chromatin structure and gene regulation.</p>
                      </article>
                      <article class="mini-edu-card hero-edu-card hero-edu-card-4">
                        <span class="hero-edu-kicker">4</span>
                        <h3>Gene Expression Signature</h3>
                        <p>Altered epigenetic programs reshape which genes are turned on or off in glioma cells.</p>
                      </article>
                      <article class="mini-edu-card hero-edu-card hero-edu-card-5">
                        <span class="hero-edu-kicker">5</span>
                        <h3>Expression-Based Classification</h3>
                        <p>Gene-expression data can support computational classification of IDH status and highlight relevant molecular features.</p>
                      </article>
                    </div>
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
                            <h2>Upload expression matrix</h2>
                            <p>Provide a compatible TSV/TXT gene-by-sample matrix or CSV sample-by-gene table.</p>
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
                          <div><strong>Sample CSV</strong><span>requires SAMPLE_ID and gene-expression columns</span></div>
                        </div>
                        <p class="upload-note">Accepted inputs: cBioPortal-style gene-by-sample TSV/TXT files or sample-by-gene CSV files with a SAMPLE_ID column.</p>
                        """
                    )
                    gr.HTML("<div class='inline-panel-heading'><h3>Decision threshold</h3><p>Optional adjustment for IDH-mutant classification sensitivity.</p></div>")
                    with gr.Group(elem_classes=["settings-panel"]):
                        threshold = gr.Slider(
                            minimum=0.05,
                            maximum=0.95,
                            value=DEFAULT_DECISION_THRESHOLD,
                            step=0.01,
                            label="IDH-mutant classification threshold",
                            info="Default is 0.50. Higher values make IDH-mutant calls more conservative.",
                        )
                    predict_button = gr.Button("Run IDH Classification", variant="primary", size="lg", elem_id="predict-button")

                with gr.Column(scale=4, elem_classes=["product-card", "summary-card"]):
                    gr.HTML(
                        """
                        <div class="section-heading">
                          <div class="section-icon">PS</div>
                          <div>
                            <h2>Prediction summary</h2>
                            <p>Analytical summary for the current sample set.</p>
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
                            <h2>Cohort analytics</h2>
                            <p>Cohort-level interpretation and recurring molecular patterns derived from the current results.</p>
                          </div>
                        </div>
                        """
                    )
                    charts = gr.HTML(batch_charts_html(pd.DataFrame()), elem_id="charts-panel")

            with gr.Row(elem_id="results-row"):
                with gr.Column(elem_classes=["product-card", "results-card", "results-card-full"]):
                    gr.HTML(
                        """
                        <div class="section-heading">
                          <div class="section-icon">RS</div>
                          <div>
                            <h2>Results</h2>
                            <p>Sample-level IDH classification, confidence score, and interpretive summary.</p>
                          </div>
                        </div>
                        """
                    )
                    results = gr.HTML(results_table_html(pd.DataFrame()), elem_id="results-panel")

            with gr.Row(elem_id="export-row"):
                with gr.Column(elem_classes=["product-card", "download-card", "download-card-wide"]):
                    gr.HTML(
                        """
                        <div class="export-toolbar">
                          <div class="section-heading compact">
                            <div class="section-icon">CSV</div>
                            <div>
                              <h2>Exports and model details</h2>
                              <p>Download result files, review output columns, or inspect model metadata.</p>
                            </div>
                          </div>
                          <button class="about-model-trigger" type="button" data-open-model-drawer>Model details</button>
                        </div>
                        """
                    )
                    with gr.Row(elem_id="export-actions-row"):
                        download = gr.File(label="Download results CSV", interactive=False, elem_id="download-file")
                        report_download = gr.File(label="Download patient report (HTML)", interactive=False, elem_id="report-file")
                    export_note = gr.Markdown("No output files have been generated yet.", elem_id="export-note")
                    gr.HTML(
                        """
                        <div class="column-guide compact-guide">
                          <h3>Output columns</h3>
                          <p><b>SAMPLE_ID</b> sample identifier provided in the uploaded file</p>
                          <p><b>predicted</b> expression-based IDH class assignment</p>
                          <p><b>IDH_mutation_probability</b> estimated probability for the IDH-mutant class</p>
                        </div>
                        """
                    )

            with gr.Row(elem_id="genes-row"):
                with gr.Column(elem_classes=["product-card", "genes-card", "genes-card-wide"]):
                    gr.HTML(
                        """
                        <div class="section-heading">
                          <div class="section-icon">GN</div>
                          <div>
                            <h2>Top global driver genes</h2>
                            <p>Global feature-ranking view from the preserved classification pipeline.</p>
                          </div>
                        </div>
                        """
                    )
                    gr.HTML(top_genes_html(18), elem_id="top-genes-panel")
                    if TOP_GENES_PATH.exists():
                        gr.File(value=str(TOP_GENES_PATH), label="Download ranked top genes", interactive=False)

            gr.HTML(
                f"""
                <div class="model-drawer-backdrop" data-close-model-drawer></div>
                <aside class="model-drawer" aria-label="About the model drawer">
                  <div class="model-drawer-header">
                    <div>
                      <span class="drawer-kicker">Model metadata</span>
                      <h2>Model details</h2>
                      <p>Training summary, held-out evaluation, and feature coverage.</p>
                    </div>
                    <button type="button" class="model-drawer-close" data-close-model-drawer aria-label="Close model drawer">X</button>
                  </div>
                  <div class="model-drawer-body">
                    {model_info_html()}
                  </div>
                </aside>
                """
            )

            predictions_state = gr.State(pd.DataFrame())
            predict_event = predict_button.click(
                fn=loading_html,
                inputs=None,
                outputs=status,
                queue=False,
                show_progress="hidden",
            )
            result_event = predict_event.then(
                fn=run_prediction_for_app,
                inputs=[input_file, threshold],
                outputs=[status, single_case, charts, results, download, report_download, predictions_state, export_note],
                queue=False,
                show_progress="hidden",
            )

    return demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    build_app().launch()
