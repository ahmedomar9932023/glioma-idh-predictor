from __future__ import annotations

import base64
from html import escape
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from .metadata import class_driver_genes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_MP4_PATH = PROJECT_ROOT / "web" / "assets" / "idh_pathway_animation.mp4"
VIDEO_WEBM_PATH = PROJECT_ROOT / "web" / "assets" / "idh_pathway_animation.webm"
FALLBACK_IMAGE_PATH = PROJECT_ROOT / "web" / "assets" / "idh_pathway_base.png"


def _prediction_context(predictions: pd.DataFrame | None) -> dict[str, Any]:
    empty = {
        "mode": "background",
        "title": "Educational pathway overview",
        "message": "This pathway video illustrates IDH mutation, 2-HG production, DNA methylation, altered RNA expression, and glioma outcome.",
        "genes": [],
    }
    if predictions is None or predictions.empty:
        return empty

    if len(predictions) == 1:
        row = predictions.iloc[0]
        label = str(row["predicted_label"])
        sample_id = escape(str(row["SAMPLE_ID"]))
        probability = float(row["IDH_mutation_probability"])
        confidence = escape(str(row["confidence_level"]).lower())
        if label == "IDH-mutant":
            return {
                "mode": "mutant",
                "title": f"{sample_id}: mutant-like pathway emphasis",
                "message": (
                    f"This sample is consistent with an IDH-mutant-like expression profile "
                    f"({probability * 100:.1f}% mutant probability, {confidence} confidence)."
                ),
                "genes": class_driver_genes(label, limit=5),
            }
        return {
            "mode": "wildtype",
            "title": f"{sample_id}: wildtype-like pathway context",
            "message": "This pathway is shown for educational context.",
            "genes": class_driver_genes(label, limit=5),
        }

    mutant_count = int((predictions["predicted_label"] == "IDH-mutant").sum())
    wildtype_count = int((predictions["predicted_label"] == "IDH-wildtype").sum())
    mean_probability = float(predictions["IDH_mutation_probability"].mean())
    if mutant_count >= wildtype_count:
        label = "IDH-mutant"
        return {
            "mode": "mutant",
            "title": "Cohort-level mutant-like pathway emphasis",
            "message": (
                f"This uploaded cohort trends IDH-mutant-like ({mutant_count}/{len(predictions)} samples; "
                f"mean mutant probability {mean_probability * 100:.1f}%)."
            ),
            "genes": class_driver_genes(label, limit=5),
        }
    label = "IDH-wildtype"
    return {
        "mode": "wildtype",
        "title": "Cohort-level wildtype-like pathway context",
        "message": "This pathway is shown for educational context.",
        "genes": class_driver_genes(label, limit=5),
    }


def _media_data_uri(path: Path, mime_type: str) -> str | None:
    if not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _video_source() -> tuple[str | None, str | None]:
    mp4 = _media_data_uri(VIDEO_MP4_PATH, "video/mp4")
    if mp4:
        return mp4, "video/mp4"
    webm = _media_data_uri(VIDEO_WEBM_PATH, "video/webm")
    if webm:
        return webm, "video/webm"
    return None, None


def _fallback_image() -> str | None:
    return _media_data_uri(FALLBACK_IMAGE_PATH, "image/png")


def pathway_illustration_html(predictions: pd.DataFrame | None = None) -> str:
    context = _prediction_context(predictions)
    video_src, video_type = _video_source()
    image_src = _fallback_image()
    uid = f"idh-video-{uuid4().hex}"
    mode_chip = {
        "mutant": "IDH-mutant-like emphasis",
        "wildtype": "IDH-wildtype educational context",
    }.get(context["mode"], "Educational pathway")
    driver_genes = context["genes"] or []
    driver_markup = (
        "".join(f"<span>{escape(gene)}</span>" for gene in driver_genes[:5])
        if driver_genes
        else '<span class="fallback-chip">Global driver genes unavailable</span>'
    )

    media_markup: str
    if video_src and video_type:
        media_markup = f"""
        <div class="video-shell">
          <video class="pathway-video" playsinline muted autoplay loop preload="auto">
            <source src="{video_src}" type="{video_type}" />
          </video>
        </div>
        """
    else:
        fallback_visual = (
            f'<img class="fallback-image" src="{image_src}" alt="Static IDH pathway illustration fallback" />'
            if image_src
            else ""
        )
        media_markup = f"""
        <div class="missing-shell">
          <div class="missing-message">
            <strong>Pathway animation video not found.</strong>
            <p>Add <code>web/assets/idh_pathway_animation.mp4</code> or <code>web/assets/idh_pathway_animation.webm</code>.</p>
          </div>
          {fallback_visual}
        </div>
        """

    return f"""
<section id="{uid}" class="idh-video-panel {context['mode']}">
  <style>
    #{uid} {{
      --ink: #12263f;
      --muted: #5e7388;
      --line: rgba(101, 126, 154, 0.2);
      display: grid;
      gap: 14px;
      color: var(--ink);
    }}
    #{uid} * {{ box-sizing: border-box; }}
    #{uid} .header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 14px;
      flex-wrap: wrap;
    }}
    #{uid} .header h3 {{
      margin: 6px 0 6px;
      font-size: 24px;
      line-height: 1.15;
    }}
    #{uid} .header p {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.55;
      max-width: 860px;
    }}
    #{uid} .chip {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 0 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.96);
      color: var(--ink);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    #{uid} .controls {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    #{uid} .controls button {{
      border: 1px solid var(--line);
      background: linear-gradient(180deg, #ffffff, #f4f8fc);
      color: var(--ink);
      border-radius: 12px;
      padding: 9px 12px;
      font: inherit;
      font-size: 12px;
      font-weight: 760;
      cursor: pointer;
    }}
    #{uid} .stage {{
      border: 1px solid var(--line);
      border-radius: 28px;
      overflow: hidden;
      background: linear-gradient(180deg, #ffffff, #f8fbff);
      box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
    }}
    #{uid} .video-shell,
    #{uid} .missing-shell {{
      width: 100%;
      aspect-ratio: 16 / 9;
      display: grid;
      place-items: center;
      background: radial-gradient(circle at top, rgba(236,244,255,0.9), rgba(247,250,255,0.96));
    }}
    #{uid} .pathway-video,
    #{uid} .fallback-image {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
      background: #f8fbff;
    }}
    #{uid} .missing-shell {{
      gap: 12px;
      padding: 18px;
      align-content: center;
    }}
    #{uid} .missing-message {{
      padding: 18px 20px;
      border: 1px solid var(--line);
      border-radius: 20px;
      background: rgba(255,255,255,0.94);
      color: var(--ink);
      max-width: 720px;
      text-align: center;
    }}
    #{uid} .missing-message p {{
      margin: 8px 0 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    #{uid} .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 10px;
      align-items: center;
    }}
    #{uid} .legend span,
    #{uid} .gene-row span,
    #{uid} .fallback-chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 28px;
      padding: 0 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.96);
      color: #425971;
      font-size: 11px;
      font-weight: 760;
    }}
    #{uid} .legend i {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }}
    #{uid} .footer-copy {{
      color: #425971;
      font-size: 13px;
      line-height: 1.5;
    }}
    #{uid} .footer-copy strong {{ color: var(--ink); }}
    #{uid} .gene-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    #{uid} .educational-note {{
      color: var(--muted);
      font-size: 12px;
    }}
  </style>

  <div class="header">
    <div>
      <span class="chip">Video pathway</span>
      <h3>Pre-rendered medical pathway animation.</h3>
      <p>A dedicated video asset handles the visual quality, while the app keeps the surrounding pathway context light, stable, and readable.</p>
    </div>
    <div style="display:grid;gap:10px;justify-items:end;">
      <span class="chip">{escape(mode_chip)}</span>
      <div class="controls">
        <button type="button" data-action="play">Play</button>
        <button type="button" data-action="pause">Pause</button>
        <button type="button" data-action="restart">Restart</button>
        <button type="button" data-action="loop">Loop: On</button>
      </div>
    </div>
  </div>

  <div class="stage">
    {media_markup}
  </div>

  <div style="display:grid;gap:10px;">
    <div class="legend">
      <span><i style="background:#14a86f"></i>IDH / mitochondria</span>
      <span><i style="background:#267fff"></i>2-HG</span>
      <span><i style="background:#7b4ce8"></i>DNA methylation</span>
      <span><i style="background:#ff9526"></i>RNA expression</span>
      <span><i style="background:#e7597d"></i>Glioma outcome</span>
    </div>
    <div class="footer-copy"><strong>{escape(context['title'])}</strong> {escape(context['message'])}</div>
    <div class="gene-row">{driver_markup}</div>
    <div class="educational-note">This visualization is educational and is not a clinical diagnosis.</div>
  </div>

  <script>
    (() => {{
      const root = document.getElementById({uid!r});
      if (!root || root.dataset.bound === "1") return;
      root.dataset.bound = "1";
      const video = root.querySelector(".pathway-video");
      const loopButton = root.querySelector('[data-action="loop"]');
      const updateLoopButton = () => {{
        if (loopButton && video) loopButton.textContent = video.loop ? "Loop: On" : "Loop: Off";
      }};
      root.querySelectorAll("[data-action]").forEach((button) => {{
        button.addEventListener("click", () => {{
          if (!video) return;
          const action = button.getAttribute("data-action");
          if (action === "play") video.play();
          if (action === "pause") video.pause();
          if (action === "restart") {{
            video.currentTime = 0;
            video.play();
          }}
          if (action === "loop") {{
            video.loop = !video.loop;
            updateLoopButton();
          }}
        }});
      }});
      updateLoopButton();
    }})();
  </script>
</section>
"""


def pathway_context_html(predictions: pd.DataFrame | None = None) -> str:
    context = _prediction_context(predictions)
    return f"""
<section class="pathway-context-card">
  <div class="mini-context">
    <strong>{escape(context['title'])}</strong>
    <p>{escape(context['message'])}</p>
  </div>
</section>
"""
