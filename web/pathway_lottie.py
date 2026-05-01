from __future__ import annotations

import base64
from html import escape
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from .metadata import class_driver_genes
from .pathway import (
    pathway_context_html as _fallback_context_html,
    pathway_illustration_html as _fallback_illustration_html,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_ILLUSTRATION_PATH = PROJECT_ROOT / "web" / "assets" / "idh_pathway_base.png"
DEFAULT_GENES = ["IDH1", "IDH2", "MGMT", "PDGFRA", "ATRX"]


def _prediction_context(predictions: pd.DataFrame | None) -> dict[str, Any]:
    empty = {
        "mode": "background",
        "title": "Educational IDH pathway overview",
        "message": (
            "Static illustration plus simple animated overlays showing mutant metabolism, "
            "DNA methylation, RNA expression, and glioma outcome."
        ),
        "genes": [],
    }
    if predictions is None or predictions.empty:
        return empty

    if len(predictions) == 1:
        row = predictions.iloc[0]
        label = str(row["predicted_label"])
        probability = float(row["IDH_mutation_probability"])
        confidence = str(row["confidence_level"])
        sample_id = escape(str(row["SAMPLE_ID"]))
        if label == "IDH-mutant":
            mode = "mutant"
            title = f"{sample_id}: mutant-like pathway emphasis"
            message = (
                f"This sample shows an IDH-mutant-like expression pattern with "
                f"{probability * 100:.1f}% mutant probability and {escape(confidence.lower())} confidence."
            )
        else:
            mode = "wildtype"
            title = f"{sample_id}: wildtype-like pathway context"
            message = (
                "This sample is more consistent with an IDH-wildtype-like expression profile, so the "
                "mutant pathway is shown as educational context."
            )
        return {
            "mode": mode,
            "title": title,
            "message": message,
            "genes": class_driver_genes(label, limit=5),
        }

    mutant_count = int((predictions["predicted_label"] == "IDH-mutant").sum())
    wildtype_count = int((predictions["predicted_label"] == "IDH-wildtype").sum())
    mean_probability = float(predictions["IDH_mutation_probability"].mean())
    if mutant_count >= wildtype_count:
        label = "IDH-mutant"
        mode = "mutant"
        title = "Cohort-level mutant-like pathway emphasis"
        message = (
            f"{mutant_count} of {len(predictions)} uploaded samples were predicted IDH-mutant. "
            f"Mean mutant probability is {mean_probability * 100:.1f}%."
        )
    else:
        label = "IDH-wildtype"
        mode = "wildtype"
        title = "Cohort-level wildtype-like pathway context"
        message = (
            f"{wildtype_count} of {len(predictions)} uploaded samples were predicted IDH-wildtype. "
            "The mutant pathway remains visible as biologic teaching context only."
        )
    return {
        "mode": mode,
        "title": title,
        "message": message,
        "genes": class_driver_genes(label, limit=5),
    }


def _base_image_data_uri() -> str | None:
    if not BASE_ILLUSTRATION_PATH.exists():
        return None
    encoded = base64.b64encode(BASE_ILLUSTRATION_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def pathway_illustration_html(predictions: pd.DataFrame | None = None) -> str:
    context = _prediction_context(predictions)
    image_src = _base_image_data_uri()
    if image_src is None:
        fallback = _fallback_illustration_html(predictions)
        return f"""
<section class="idh-pathway-missing">
  <div style="padding:18px 20px;border:1px solid rgba(108,136,165,0.2);border-radius:24px;background:linear-gradient(180deg,rgba(255,255,255,0.98),rgba(248,251,255,0.98));color:#142a42;">
    <strong>Base illustration missing.</strong>
    <p style="margin:8px 0 0;color:#4e647a;line-height:1.5;">Expected image file: {escape(str(BASE_ILLUSTRATION_PATH))}. Showing clean fallback pathway until the image is restored.</p>
  </div>
  {fallback}
</section>
"""

    uid = f"idh-overlay-{uuid4().hex}"
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
    overlay_opacity = "0.62" if context["mode"] == "wildtype" else "1"

    return f"""
<section id="{uid}" class="idh-illustration-overlay {context['mode']}">
  <style>
    #{uid} {{
      --ink: #11263e;
      --muted: #596d83;
      --line: rgba(101, 126, 154, 0.22);
      --green: #14a86f;
      --blue: #267fff;
      --purple: #7b4ce8;
      --orange: #ff9526;
      --red: #e7597d;
      --play: running;
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
      max-width: 840px;
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
      position: relative;
    }}
    #{uid} .base-wrap {{
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      overflow: hidden;
    }}
    #{uid} .base-image {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
      transform-origin: 46% 51%;
      animation: {uid}-cellBreath 8.4s ease-in-out infinite;
      animation-play-state: var(--play);
    }}
    #{uid} .overlay-svg {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      display: block;
    }}
    #{uid} .overlay-group {{
      opacity: {overlay_opacity};
    }}
    #{uid}.wildtype .overlay-group {{
      opacity: 0.68;
    }}
    #{uid} .cyto-speck {{
      fill: rgba(60, 153, 255, 0.18);
      opacity: 0.75;
    }}
    #{uid} .flow-track {{
      fill: none;
      stroke-linecap: round;
      stroke-linejoin: round;
      stroke-width: 6;
      opacity: 0.96;
    }}
    #{uid} .flow-blue {{ stroke: var(--blue); }}
    #{uid} .flow-orange {{ stroke: var(--orange); }}
    #{uid} .flow-red {{ stroke: var(--red); }}
    #{uid} .particle {{
      fill: var(--blue);
      opacity: 0;
      filter: drop-shadow(0 0 6px rgba(38,127,255,0.45));
    }}
    #{uid} .me-dot {{
      fill: #d26aff;
      opacity: 0.22;
      filter: drop-shadow(0 0 5px rgba(210,106,255,0.35));
    }}
    #{uid} .me-text {{
      fill: #fff;
      font-size: 9px;
      font-weight: 800;
    }}
    #{uid} .nucleus-breath {{
      transform-origin: 50.2% 42.2%;
      animation: {uid}-nucleusBreath 6.4s ease-in-out infinite;
      animation-play-state: var(--play);
    }}
    #{uid} .chromatin-drift {{
      animation: {uid}-chromatinDrift 5.2s ease-in-out infinite;
      animation-play-state: var(--play);
    }}
    #{uid} .rna-glow {{
      fill: none;
      stroke: rgba(255, 149, 38, 0.42);
      stroke-width: 12;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    #{uid} .rna-strand {{
      fill: none;
      stroke: rgba(255, 149, 38, 0.88);
      stroke-width: 4;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    #{uid} .rna-bead {{
      fill: rgba(255, 149, 38, 0.96);
      filter: drop-shadow(0 0 5px rgba(255,149,38,0.42));
    }}
    #{uid} .tumor-pulse {{
      fill: rgba(231, 89, 125, 0.18);
      transform-origin: 87.8% 58.7%;
      animation: {uid}-tumorPulse 2.4s ease-in-out infinite;
      animation-play-state: var(--play);
    }}
    #{uid} .tumor-halo {{
      fill: rgba(231, 89, 125, 0.12);
      transform-origin: 87.8% 58.7%;
      animation: {uid}-tumorHalo 3.6s ease-in-out infinite;
      animation-play-state: var(--play);
    }}
    #{uid} .label-box {{
      fill: rgba(255,255,255,0.97);
      stroke: rgba(100, 126, 154, 0.22);
      stroke-width: 1.1;
    }}
    #{uid} .label-line {{
      stroke: rgba(101, 125, 150, 0.56);
      stroke-width: 1.5;
      stroke-linecap: round;
    }}
    #{uid} .label-text {{
      fill: var(--ink);
      font-size: 13px;
      font-weight: 800;
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
    #{uid} .footer-copy strong {{
      color: var(--ink);
    }}
    #{uid} .gene-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    #{uid} .educational-note {{
      color: var(--muted);
      font-size: 12px;
    }}
    #{uid} .p1 {{ animation: {uid}-p1 3.6s ease-in-out infinite; animation-play-state: var(--play); }}
    #{uid} .p2 {{ animation: {uid}-p2 3.6s ease-in-out infinite; animation-play-state: var(--play); }}
    #{uid} .p3 {{ animation: {uid}-p3 3.7s ease-in-out infinite; animation-play-state: var(--play); }}
    #{uid} .p4 {{ animation: {uid}-p4 3.7s ease-in-out infinite; animation-play-state: var(--play); }}
    #{uid} .p5 {{ animation: {uid}-p5 3.8s ease-in-out infinite; animation-play-state: var(--play); }}
    #{uid} .p6 {{ animation: {uid}-p6 3.8s ease-in-out infinite; animation-play-state: var(--play); }}
    #{uid} .p7 {{ animation: {uid}-p7 3.9s ease-in-out infinite; animation-play-state: var(--play); }}
    #{uid} .p8 {{ animation: {uid}-p8 3.9s ease-in-out infinite; animation-play-state: var(--play); }}
    #{uid} .p9 {{ animation: {uid}-p9 4s ease-in-out infinite; animation-play-state: var(--play); }}
    #{uid} .p10 {{ animation: {uid}-p10 4s ease-in-out infinite; animation-play-state: var(--play); }}
    #{uid} .me1 {{ animation: {uid}-me1 4.1s ease-in-out infinite; animation-play-state: var(--play); }}
    #{uid} .me2 {{ animation: {uid}-me2 4.6s ease-in-out infinite; animation-play-state: var(--play); }}
    #{uid} .me3 {{ animation: {uid}-me3 4.3s ease-in-out infinite; animation-play-state: var(--play); }}
    #{uid} .me4 {{ animation: {uid}-me4 4.8s ease-in-out infinite; animation-play-state: var(--play); }}
    #{uid} .me5 {{ animation: {uid}-me5 4.4s ease-in-out infinite; animation-play-state: var(--play); }}
    #{uid} .rna-anim {{ animation: {uid}-rnaPulse 2.2s ease-in-out infinite; animation-play-state: var(--play); }}
    #{uid} .vesicle-1 {{ animation: {uid}-vesicle1 6.8s ease-in-out infinite; animation-play-state: var(--play); }}
    #{uid} .vesicle-2 {{ animation: {uid}-vesicle2 7.6s ease-in-out infinite; animation-play-state: var(--play); }}
    #{uid} .vesicle-3 {{ animation: {uid}-vesicle3 7.1s ease-in-out infinite; animation-play-state: var(--play); }}
    @keyframes {uid}-cellBreath {{ 0%,100% {{ transform: scale(1); }} 50% {{ transform: scale(1.018); }} }}
    @keyframes {uid}-nucleusBreath {{ 0%,100% {{ transform: scale(1); }} 50% {{ transform: scale(1.013); }} }}
    @keyframes {uid}-chromatinDrift {{ 0%,100% {{ transform: translate(0px,0px); }} 50% {{ transform: translate(2px,-2px); }} }}
    @keyframes {uid}-p1 {{ 0% {{ opacity: 0; transform: translate(0px,0px); }} 12% {{ opacity: 0.95; }} 56% {{ opacity: 1; transform: translate(228px,-8px); }} 100% {{ opacity: 0; transform: translate(436px,-6px); }} }}
    @keyframes {uid}-p2 {{ 0% {{ opacity: 0; transform: translate(0px,0px); }} 18% {{ opacity: 0.95; }} 58% {{ opacity: 1; transform: translate(226px,10px); }} 100% {{ opacity: 0; transform: translate(430px,14px); }} }}
    @keyframes {uid}-p3 {{ 0% {{ opacity: 0; transform: translate(0px,0px); }} 22% {{ opacity: 0.95; }} 60% {{ opacity: 1; transform: translate(236px,18px); }} 100% {{ opacity: 0; transform: translate(438px,22px); }} }}
    @keyframes {uid}-p4 {{ 0% {{ opacity: 0; transform: translate(0px,0px); }} 28% {{ opacity: 0.95; }} 62% {{ opacity: 1; transform: translate(242px,28px); }} 100% {{ opacity: 0; transform: translate(444px,34px); }} }}
    @keyframes {uid}-p5 {{ 0% {{ opacity: 0; transform: translate(0px,0px); }} 32% {{ opacity: 0.95; }} 64% {{ opacity: 1; transform: translate(248px,38px); }} 100% {{ opacity: 0; transform: translate(452px,44px); }} }}
    @keyframes {uid}-p6 {{ 0% {{ opacity: 0; transform: translate(0px,0px); }} 10% {{ opacity: 0.9; }} 54% {{ opacity: 1; transform: translate(216px,4px); }} 100% {{ opacity: 0; transform: translate(422px,10px); }} }}
    @keyframes {uid}-p7 {{ 0% {{ opacity: 0; transform: translate(0px,0px); }} 16% {{ opacity: 0.9; }} 56% {{ opacity: 1; transform: translate(222px,16px); }} 100% {{ opacity: 0; transform: translate(430px,24px); }} }}
    @keyframes {uid}-p8 {{ 0% {{ opacity: 0; transform: translate(0px,0px); }} 24% {{ opacity: 0.9; }} 60% {{ opacity: 1; transform: translate(232px,28px); }} 100% {{ opacity: 0; transform: translate(438px,38px); }} }}
    @keyframes {uid}-p9 {{ 0% {{ opacity: 0; transform: translate(0px,0px); }} 30% {{ opacity: 0.9; }} 62% {{ opacity: 1; transform: translate(240px,40px); }} 100% {{ opacity: 0; transform: translate(446px,50px); }} }}
    @keyframes {uid}-p10 {{ 0% {{ opacity: 0; transform: translate(0px,0px); }} 36% {{ opacity: 0.9; }} 64% {{ opacity: 1; transform: translate(248px,52px); }} 100% {{ opacity: 0; transform: translate(454px,62px); }} }}
    @keyframes {uid}-me1 {{ 0%,100% {{ opacity: 0.18; transform: scale(0.96); }} 35% {{ opacity: 0.98; transform: scale(1.08); }} 60% {{ opacity: 0.52; transform: scale(1); }} }}
    @keyframes {uid}-me2 {{ 0%,100% {{ opacity: 0.22; transform: scale(0.96); }} 40% {{ opacity: 1; transform: scale(1.1); }} 64% {{ opacity: 0.46; transform: scale(1); }} }}
    @keyframes {uid}-me3 {{ 0%,100% {{ opacity: 0.16; transform: scale(0.96); }} 32% {{ opacity: 0.95; transform: scale(1.08); }} 58% {{ opacity: 0.48; transform: scale(1); }} }}
    @keyframes {uid}-me4 {{ 0%,100% {{ opacity: 0.2; transform: scale(0.96); }} 44% {{ opacity: 1; transform: scale(1.1); }} 68% {{ opacity: 0.5; transform: scale(1); }} }}
    @keyframes {uid}-me5 {{ 0%,100% {{ opacity: 0.18; transform: scale(0.96); }} 38% {{ opacity: 0.96; transform: scale(1.08); }} 62% {{ opacity: 0.44; transform: scale(1); }} }}
    @keyframes {uid}-rnaPulse {{ 0%,100% {{ opacity: 0.62; transform: translateY(0px); }} 50% {{ opacity: 1; transform: translateY(-3px); }} }}
    @keyframes {uid}-tumorPulse {{ 0%,100% {{ transform: scale(1); opacity: 0.17; }} 50% {{ transform: scale(1.1); opacity: 0.3; }} }}
    @keyframes {uid}-tumorHalo {{ 0%,100% {{ transform: scale(1); opacity: 0.08; }} 50% {{ transform: scale(1.14); opacity: 0.18; }} }}
    @keyframes {uid}-vesicle1 {{ 0%,100% {{ transform: translate(0px,0px); opacity: 0.35; }} 50% {{ transform: translate(8px,-10px); opacity: 0.5; }} }}
    @keyframes {uid}-vesicle2 {{ 0%,100% {{ transform: translate(0px,0px); opacity: 0.28; }} 50% {{ transform: translate(-6px,9px); opacity: 0.44; }} }}
    @keyframes {uid}-vesicle3 {{ 0%,100% {{ transform: translate(0px,0px); opacity: 0.3; }} 50% {{ transform: translate(5px,-7px); opacity: 0.46; }} }}
  </style>

  <div class="header">
    <div>
      <span class="chip">Illustrated pathway</span>
      <h3>Static medical illustration with focused animated overlays.</h3>
      <p>Blue 2-HG particles, methylation dots, RNA activation, and tumor pulse sit above a richer base illustration instead of redrawing the whole cell in code.</p>
    </div>
    <div style="display:grid;gap:10px;justify-items:end;">
      <span class="chip">{escape(mode_chip)}</span>
      <div class="controls">
        <button type="button" data-action="play">Play</button>
        <button type="button" data-action="pause">Pause</button>
        <button type="button" data-action="restart">Restart</button>
      </div>
    </div>
  </div>

  <div class="stage">
    <div class="base-wrap">
      <img class="base-image" src="{image_src}" alt="Base IDH pathway medical illustration" />
      <svg class="overlay-svg" viewBox="0 0 1600 900" role="img" aria-label="Animated pathway overlay">
        <g class="overlay-group">
          <g opacity="0.74">
            <circle class="cyto-speck vesicle-1" cx="248" cy="594" r="11" />
            <circle class="cyto-speck vesicle-2" cx="458" cy="248" r="8" />
            <circle class="cyto-speck vesicle-3" cx="1074" cy="604" r="10" />
            <circle class="cyto-speck vesicle-1" cx="1208" cy="308" r="7" />
            <circle class="cyto-speck vesicle-2" cx="914" cy="204" r="9" />
          </g>

          <path class="flow-track flow-blue" d="M520,334 C642,338 712,342 774,344 C830,346 884,344 938,334" />
          <path class="flow-track flow-orange" d="M944,470 C1026,472 1092,474 1152,466 C1220,456 1272,438 1322,420" />
          <path class="flow-track flow-red" d="M1334,420 C1396,430 1442,452 1480,492" />

          <g class="nucleus-breath">
            <g class="chromatin-drift">
              <path class="rna-strand rna-anim" d="M982,516 C1040,492 1102,500 1160,520" opacity="0.62" />
            </g>
          </g>

          <g class="p1"><circle class="particle" cx="522" cy="332" r="10" /></g>
          <g class="p2"><circle class="particle" cx="526" cy="344" r="8" /></g>
          <g class="p3"><circle class="particle" cx="530" cy="336" r="9" /></g>
          <g class="p4"><circle class="particle" cx="534" cy="352" r="8" /></g>
          <g class="p5"><circle class="particle" cx="538" cy="346" r="9" /></g>
          <g class="p6"><circle class="particle" cx="542" cy="358" r="8" /></g>
          <g class="p7"><circle class="particle" cx="546" cy="368" r="8" /></g>
          <g class="p8"><circle class="particle" cx="550" cy="380" r="8" /></g>
          <g class="p9"><circle class="particle" cx="554" cy="392" r="8" /></g>
          <g class="p10"><circle class="particle" cx="558" cy="404" r="8" /></g>

          <g class="me1"><circle class="me-dot" cx="742" cy="318" r="16" /><text class="me-text" x="732" y="323">Me</text></g>
          <g class="me2"><circle class="me-dot" cx="808" cy="294" r="16" /><text class="me-text" x="798" y="299">Me</text></g>
          <g class="me3"><circle class="me-dot" cx="888" cy="316" r="16" /><text class="me-text" x="878" y="321">Me</text></g>
          <g class="me4"><circle class="me-dot" cx="934" cy="408" r="16" /><text class="me-text" x="924" y="413">Me</text></g>
          <g class="me5"><circle class="me-dot" cx="816" cy="492" r="16" /><text class="me-text" x="806" y="497">Me</text></g>

          <path class="rna-glow rna-anim" d="M978,494 C1064,452 1138,458 1210,488 C1264,510 1306,512 1346,500" />
          <path class="rna-strand rna-anim" d="M982,502 C1062,470 1132,476 1196,506 C1248,530 1300,530 1346,516" />
          <circle class="rna-bead rna-anim" cx="1096" cy="482" r="7" />
          <circle class="rna-bead rna-anim" cx="1188" cy="506" r="6" />
          <circle class="rna-bead rna-anim" cx="1286" cy="520" r="7" />

          <ellipse class="tumor-halo" cx="1404" cy="528" rx="120" ry="92" />
          <ellipse class="tumor-pulse" cx="1404" cy="528" rx="98" ry="76" />

          <g transform="translate(266 174)">
            <line class="label-line" x1="80" y1="40" x2="96" y2="76" />
            <rect class="label-box" width="136" height="34" rx="14" />
            <text class="label-text" x="16" y="22">IDH mutation</text>
          </g>
          <g transform="translate(554 452)">
            <line class="label-line" x1="72" y1="0" x2="74" y2="-52" />
            <rect class="label-box" width="88" height="34" rx="14" />
            <text class="label-text" x="20" y="22">2-HG</text>
          </g>
          <g transform="translate(706 162)">
            <line class="label-line" x1="116" y1="34" x2="118" y2="82" />
            <rect class="label-box" width="168" height="34" rx="14" />
            <text class="label-text" x="16" y="22">DNA methylation</text>
          </g>
          <g transform="translate(1078 246)">
            <line class="label-line" x1="116" y1="34" x2="116" y2="86" />
            <rect class="label-box" width="176" height="34" rx="14" />
            <text class="label-text" x="16" y="22">Altered RNA expression</text>
          </g>
          <g transform="translate(1326 230)">
            <line class="label-line" x1="92" y1="34" x2="76" y2="164" />
            <rect class="label-box" width="140" height="34" rx="14" />
            <text class="label-text" x="16" y="22">Glioma outcome</text>
          </g>
        </g>
      </svg>
    </div>
  </div>

  <div style="display:grid;gap:10px;">
    <div class="legend">
      <span><i style="background:#14a86f"></i>IDH / mitochondria</span>
      <span><i style="background:#267fff"></i>2-HG</span>
      <span><i style="background:#7b4ce8"></i>Nucleus / DNA</span>
      <span><i style="background:#ff9526"></i>RNA / expression</span>
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
      const setPlay = (state) => root.style.setProperty("--play", state);
      root.querySelectorAll("[data-action]").forEach((button) => {{
        button.addEventListener("click", () => {{
          const action = button.getAttribute("data-action");
          if (action === "play") setPlay("running");
          if (action === "pause") setPlay("paused");
          if (action === "restart") {{
            setPlay("paused");
            const stage = root.querySelector(".stage");
            if (stage) {{
              stage.style.display = "none";
              void stage.offsetHeight;
              stage.style.display = "";
            }}
            requestAnimationFrame(() => setPlay("running"));
          }}
        }});
      }});
    }})();
  </script>
</section>
"""


def pathway_context_html(predictions: pd.DataFrame | None = None) -> str:
    return _fallback_context_html(predictions)
