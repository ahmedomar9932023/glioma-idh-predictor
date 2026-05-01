from __future__ import annotations

from html import escape
from typing import Any
from uuid import uuid4

import pandas as pd

from .metadata import class_driver_genes


DEFAULT_GENES = ["IDH1", "IDH2", "MGMT", "PDGFRA", "ATRX"]


def _prediction_context(predictions: pd.DataFrame | None) -> dict[str, Any]:
    empty = {
        "mode": "background",
        "title": "Animated IDH pathway overview",
        "message": (
            "This looping educational diagram shows how mutant IDH rewires metabolism, "
            "DNA methylation, RNA programs, and glioma behavior."
        ),
        "genes": [],
        "mean_probability": None,
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
                f"This sample shows an IDH-mutant-like expression pattern. The model assigned "
                f"{probability * 100:.1f}% mutant probability with {escape(confidence.lower())} confidence."
            )
        else:
            mode = "wildtype"
            title = f"{sample_id}: wildtype-like pathway context"
            message = (
                "This sample is more consistent with an IDH-wildtype-like expression profile, so the mutant "
                "pathway is shown as educational background rather than confirmed activity."
            )
        return {
            "mode": mode,
            "title": title,
            "message": message,
            "genes": class_driver_genes(label, limit=5),
            "mean_probability": probability,
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
        "mean_probability": mean_probability,
    }


def pathway_illustration_html(predictions: pd.DataFrame | None = None) -> str:
    context = _prediction_context(predictions)
    mode = context["mode"]
    uid = f"idh-pathway-{uuid4().hex}"
    genes = (context.get("genes") or DEFAULT_GENES)[:4]
    driver_markup = "".join(f"<span>{escape(gene)}</span>" for gene in genes)
    mode_chip = {
        "mutant": "Mutant-like emphasis",
        "wildtype": "Wildtype-like context",
    }.get(mode, "Educational overview")
    note = {
        "mutant": "Pathway shown at full educational emphasis.",
        "wildtype": "Pathway softened and shown as educational context.",
    }.get(mode, "Looping pathway overview.")
    flow_opacity = "1" if mode == "mutant" else "0.54" if mode == "wildtype" else "0.88"

    return f"""
<section id="{uid}" class="idh-pathway-rich {mode}">
  <style>
    #{uid} {{
      --ink: #11263e;
      --muted: #596d83;
      --line: rgba(101, 126, 154, 0.22);
      --green: #14a86f;
      --green-2: #43d89a;
      --green-3: #b7ffe1;
      --blue: #267fff;
      --blue-2: #8ec7ff;
      --blue-3: #d9eeff;
      --purple: #7b4ce8;
      --purple-2: #cab3ff;
      --purple-3: #efe6ff;
      --orange: #ff9526;
      --orange-2: #ffce66;
      --red: #e7597d;
      --red-2: #ffb1c3;
      --shadow: 0 12px 24px rgba(15, 23, 42, 0.05);
      --speed: 7s;
      --play: running;
      --flow-opacity: {flow_opacity};
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
    #{uid} .scene {{
      border: 1px solid var(--line);
      border-radius: 28px;
      overflow: hidden;
      background: linear-gradient(180deg, #ffffff, #f8fbff);
      box-shadow: var(--shadow);
    }}
    #{uid} svg {{ display: block; width: 100%; height: auto; }}
    #{uid} .zone-fill {{ fill: rgba(255,255,255,0.94); stroke: rgba(114, 140, 170, 0.18); stroke-width: 1.2; }}
    #{uid} .zone-title {{ fill: #6a7d92; font-size: 11px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }}
    #{uid} .mito-glow {{ fill: rgba(67, 216, 154, 0.08); }}
    #{uid} .mito-shape {{ fill: var(--green); stroke: rgba(10, 96, 80, 0.42); stroke-width: 2; }}
    #{uid} .mito-cristae {{ fill: none; stroke: rgba(205,255,240,0.95); stroke-width: 4; stroke-linecap: round; }}
    #{uid} .enzyme-core {{ fill: #91ff72; }}
    #{uid} .enzyme-star {{ fill: #7ff463; }}
    #{uid} .small-molecule {{ fill: rgba(180, 190, 203, 0.9); stroke: rgba(128, 142, 158, 0.24); stroke-width: 1.1; }}
    #{uid} .small-molecule-text {{ fill: #66798d; font-size: 10px; font-weight: 800; }}
    #{uid} .nucleus-glow {{ fill: rgba(124, 76, 232, 0.06); }}
    #{uid} .nucleus-shell {{ fill: var(--purple-2); stroke: rgba(103, 66, 197, 0.42); stroke-width: 2; }}
    #{uid} .nucleus-ring {{ fill: none; stroke: rgba(231, 223, 255, 0.94); stroke-width: 8; }}
    #{uid} .dna-rail-a,
    #{uid} .dna-rail-b {{ fill: none; stroke: #6c38d7; stroke-width: 5.5; stroke-linecap: round; }}
    #{uid} .dna-rung {{ stroke: rgba(219, 194, 255, 0.96); stroke-width: 2.4; stroke-linecap: round; }}
    #{uid} .nucleosome {{ fill: rgba(123, 76, 232, 0.52); }}
    #{uid} .me-dot {{ fill: #d26aff; }}
    #{uid} .me-text {{ fill: #fff; font-size: 9px; font-weight: 800; }}
    #{uid} .inhibit-ring {{ fill: none; stroke: rgba(217, 60, 94, 0.85); stroke-width: 2.1; }}
    #{uid} .inhibit-bar {{ stroke: rgba(217, 60, 94, 0.85); stroke-width: 3; stroke-linecap: round; }}
    #{uid} .rna-core {{ fill: none; stroke: #ff9526; stroke-width: 6; stroke-linecap: round; stroke-linejoin: round; }}
    #{uid} .polymerase {{ fill: #ffb34e; }}
    #{uid} .rna-bead {{ fill: #ffc45f; }}
    #{uid} .tumor-shadow {{ fill: rgba(231, 89, 125, 0.14); }}
    #{uid} .outcome-cell {{ fill: var(--red-2); stroke: rgba(195, 63, 103, 0.36); stroke-width: 2; }}
    #{uid} .outcome-core {{ fill: rgba(197, 34, 82, 0.56); }}
    #{uid} .outcome-branch {{ fill: none; stroke: rgba(231, 89, 125, 0.42); stroke-width: 2; stroke-linecap: round; }}
    #{uid} .flow-line-blue,
    #{uid} .flow-line-orange,
    #{uid} .flow-line-red {{
      fill: none;
      stroke-linecap: round;
      stroke-linejoin: round;
      opacity: var(--flow-opacity);
    }}
    #{uid} .flow-line-blue {{ stroke: var(--blue); stroke-width: 7; }}
    #{uid} .flow-line-orange {{ stroke: var(--orange); stroke-width: 7; }}
    #{uid} .flow-line-red {{ stroke: var(--red); stroke-width: 7; }}
    #{uid} .flow-head-blue {{ fill: var(--blue); opacity: var(--flow-opacity); }}
    #{uid} .flow-head-orange {{ fill: var(--orange); opacity: var(--flow-opacity); }}
    #{uid} .flow-head-red {{ fill: var(--red); opacity: var(--flow-opacity); }}
    #{uid} .particle {{ fill: var(--blue); opacity: 0; }}
    #{uid} .label-box {{ fill: rgba(255,255,255,0.98); stroke: rgba(100, 126, 154, 0.22); stroke-width: 1.3; }}
    #{uid} .label-line {{ stroke: rgba(101, 125, 150, 0.56); stroke-width: 1.4; stroke-linecap: round; }}
    #{uid} .label-title {{ fill: var(--ink); font-size: 13px; font-weight: 800; }}
    #{uid} .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 10px;
      align-items: center;
    }}
    #{uid} .legend span,
    #{uid} .gene-row span {{
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
    #{uid} .legend i {{ width: 10px; height: 10px; border-radius: 999px; display: inline-block; }}
    #{uid} .timeline {{ display: grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: 10px; padding: 18px; }}
    #{uid} .step {{
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.96);
      color: #425971;
      font-size: 12px;
      font-weight: 760;
    }}
    #{uid} .step strong {{ display: block; color: var(--ink); margin-bottom: 2px; font-size: 12px; }}
    #{uid} .footer-copy {{ color: #425971; font-size: 13px; line-height: 1.5; }}
    #{uid} .footer-copy strong {{ color: var(--ink); }}
    #{uid} .gene-row {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    #{uid} .educational-note {{ color: var(--muted); font-size: 12px; }}
    #{uid}.wildtype .fg-primary {{ opacity: 0.7; }}
    #{uid}.wildtype .flow-line-blue,
    #{uid}.wildtype .flow-line-orange,
    #{uid}.wildtype .flow-line-red,
    #{uid}.wildtype .flow-head-blue,
    #{uid}.wildtype .flow-head-orange,
    #{uid}.wildtype .flow-head-red {{ opacity: 0.56; }}

    #{uid} .mito-pulse {{ animation: {uid}-mitoPulse var(--speed) linear infinite; animation-play-state: var(--play); transform-origin: 226px 250px; }}
    #{uid} .enzyme-pulse {{ animation: {uid}-enzymePulse calc(var(--speed) * 0.6) linear infinite; animation-play-state: var(--play); transform-origin: 300px 188px; }}
    #{uid} .dna-alive {{ animation: {uid}-dnaAlive calc(var(--speed) * 0.7) ease-in-out infinite; animation-play-state: var(--play); transform-origin: 622px 246px; }}
    #{uid} .rna-alive {{ animation: {uid}-rnaAlive var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .tumor-alive {{ animation: {uid}-tumorAlive calc(var(--speed) * 0.8) ease-in-out infinite; animation-play-state: var(--play); transform-origin: 1116px 248px; }}
    #{uid} .me1 {{ animation: {uid}-me1 var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .me2 {{ animation: {uid}-me2 var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .me3 {{ animation: {uid}-me3 var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .me4 {{ animation: {uid}-me4 var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .me5 {{ animation: {uid}-me5 var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .p1 {{ animation: {uid}-p1 var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .p2 {{ animation: {uid}-p2 var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .p3 {{ animation: {uid}-p3 var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .p4 {{ animation: {uid}-p4 var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .p5 {{ animation: {uid}-p5 var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .p6 {{ animation: {uid}-p6 var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .p7 {{ animation: {uid}-p7 var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .p8 {{ animation: {uid}-p8 var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .p9 {{ animation: {uid}-p9 var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .p10 {{ animation: {uid}-p10 var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .p11 {{ animation: {uid}-p11 var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .p12 {{ animation: {uid}-p12 var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .p13 {{ animation: {uid}-p13 var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .p14 {{ animation: {uid}-p14 var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .p15 {{ animation: {uid}-p15 var(--speed) linear infinite; animation-play-state: var(--play); }}
    #{uid} .p16 {{ animation: {uid}-p16 var(--speed) linear infinite; animation-play-state: var(--play); }}

    @keyframes {uid}-mitoPulse {{
      0%,100% {{ transform: scale(1); }}
      22% {{ transform: scale(1.03); }}
      48% {{ transform: scale(1.05); }}
      76% {{ transform: scale(1.03); }}
    }}
    @keyframes {uid}-enzymePulse {{
      0%,100% {{ transform: scale(1); opacity: 0.82; }}
      30% {{ transform: scale(1.12); opacity: 1; }}
      58% {{ transform: scale(1.18); opacity: 0.96; }}
      84% {{ transform: scale(1.06); opacity: 0.9; }}
    }}
    @keyframes {uid}-dnaAlive {{
      0%,100% {{ transform: translateY(0); }}
      30% {{ transform: translateY(-2px); }}
      60% {{ transform: translateY(2px); }}
      84% {{ transform: translateY(-1px); }}
    }}
    @keyframes {uid}-rnaAlive {{
      0%,100% {{ opacity: 0.92; }}
      50% {{ opacity: 1; }}
    }}
    @keyframes {uid}-tumorAlive {{
      0%,100% {{ transform: scale(1); }}
      40% {{ transform: scale(1.05); }}
      70% {{ transform: scale(1.07); }}
    }}
    @keyframes {uid}-me1 {{ 0%,18% {{ opacity: 0.16; }} 26%,100% {{ opacity: 1; }} }}
    @keyframes {uid}-me2 {{ 0%,24% {{ opacity: 0.16; }} 32%,100% {{ opacity: 1; }} }}
    @keyframes {uid}-me3 {{ 0%,30% {{ opacity: 0.16; }} 38%,100% {{ opacity: 1; }} }}
    @keyframes {uid}-me4 {{ 0%,36% {{ opacity: 0.16; }} 44%,100% {{ opacity: 1; }} }}
    @keyframes {uid}-me5 {{ 0%,42% {{ opacity: 0.16; }} 50%,100% {{ opacity: 1; }} }}
    @keyframes {uid}-p1 {{
      0%,8% {{ opacity: 0; transform: translate(0,0); }}
      14% {{ opacity: 1; transform: translate(0,0); }}
      34% {{ opacity: 1; transform: translate(142px,-6px); }}
      58% {{ opacity: 0.98; transform: translate(296px,-8px); }}
      74%,100% {{ opacity: 0; transform: translate(438px,-6px); }}
    }}
    @keyframes {uid}-p2 {{
      0%,10% {{ opacity: 0; transform: translate(0,0); }}
      16% {{ opacity: 1; transform: translate(0,0); }}
      36% {{ opacity: 1; transform: translate(144px,4px); }}
      60% {{ opacity: 0.98; transform: translate(298px,2px); }}
      76%,100% {{ opacity: 0; transform: translate(440px,0); }}
    }}
    @keyframes {uid}-p3 {{
      0%,12% {{ opacity: 0; transform: translate(0,0); }}
      18% {{ opacity: 1; transform: translate(0,0); }}
      38% {{ opacity: 1; transform: translate(146px,14px); }}
      62% {{ opacity: 0.98; transform: translate(300px,14px); }}
      78%,100% {{ opacity: 0; transform: translate(442px,12px); }}
    }}
    @keyframes {uid}-p4 {{
      0%,14% {{ opacity: 0; transform: translate(0,0); }}
      20% {{ opacity: 1; transform: translate(0,0); }}
      40% {{ opacity: 1; transform: translate(148px,24px); }}
      64% {{ opacity: 0.98; transform: translate(302px,26px); }}
      80%,100% {{ opacity: 0; transform: translate(444px,24px); }}
    }}
    @keyframes {uid}-p5 {{
      0%,16% {{ opacity: 0; transform: translate(0,0); }}
      22% {{ opacity: 1; transform: translate(0,0); }}
      42% {{ opacity: 1; transform: translate(150px,34px); }}
      66% {{ opacity: 0.98; transform: translate(304px,38px); }}
      82%,100% {{ opacity: 0; transform: translate(446px,36px); }}
    }}
    @keyframes {uid}-p6 {{
      0%,18% {{ opacity: 0; transform: translate(0,0); }}
      24% {{ opacity: 1; transform: translate(0,0); }}
      44% {{ opacity: 1; transform: translate(152px,46px); }}
      68% {{ opacity: 0.98; transform: translate(306px,48px); }}
      84%,100% {{ opacity: 0; transform: translate(448px,46px); }}
    }}
    @keyframes {uid}-p7 {{
      0%,20% {{ opacity: 0; transform: translate(0,0); }}
      26% {{ opacity: 1; transform: translate(0,0); }}
      46% {{ opacity: 1; transform: translate(154px,58px); }}
      70% {{ opacity: 0.98; transform: translate(308px,60px); }}
      86%,100% {{ opacity: 0; transform: translate(450px,58px); }}
    }}
    @keyframes {uid}-p8 {{
      0%,22% {{ opacity: 0; transform: translate(0,0); }}
      28% {{ opacity: 1; transform: translate(0,0); }}
      48% {{ opacity: 1; transform: translate(156px,-16px); }}
      72% {{ opacity: 0.98; transform: translate(310px,-18px); }}
      88%,100% {{ opacity: 0; transform: translate(452px,-16px); }}
    }}
    @keyframes {uid}-p9 {{
      0%,24% {{ opacity: 0; transform: translate(0,0); }}
      30% {{ opacity: 1; transform: translate(0,0); }}
      50% {{ opacity: 1; transform: translate(158px,-28px); }}
      74% {{ opacity: 0.98; transform: translate(312px,-30px); }}
      90%,100% {{ opacity: 0; transform: translate(454px,-28px); }}
    }}
    @keyframes {uid}-p10 {{
      0%,26% {{ opacity: 0; transform: translate(0,0); }}
      32% {{ opacity: 1; transform: translate(0,0); }}
      52% {{ opacity: 1; transform: translate(160px,-40px); }}
      76% {{ opacity: 0.98; transform: translate(314px,-40px); }}
      92%,100% {{ opacity: 0; transform: translate(456px,-38px); }}
    }}
    @keyframes {uid}-p11 {{
      0%,6% {{ opacity: 0; transform: translate(0,0); }}
      12% {{ opacity: 1; transform: translate(0,0); }}
      32% {{ opacity: 1; transform: translate(142px,10px); }}
      56% {{ opacity: 0.98; transform: translate(296px,10px); }}
      72%,100% {{ opacity: 0; transform: translate(438px,10px); }}
    }}
    @keyframes {uid}-p12 {{
      0%,8% {{ opacity: 0; transform: translate(0,0); }}
      14% {{ opacity: 1; transform: translate(0,0); }}
      34% {{ opacity: 1; transform: translate(144px,22px); }}
      58% {{ opacity: 0.98; transform: translate(298px,24px); }}
      74%,100% {{ opacity: 0; transform: translate(440px,22px); }}
    }}
    @keyframes {uid}-p13 {{
      0%,10% {{ opacity: 0; transform: translate(0,0); }}
      16% {{ opacity: 1; transform: translate(0,0); }}
      36% {{ opacity: 1; transform: translate(146px,34px); }}
      60% {{ opacity: 0.98; transform: translate(300px,36px); }}
      76%,100% {{ opacity: 0; transform: translate(442px,34px); }}
    }}
    @keyframes {uid}-p14 {{
      0%,12% {{ opacity: 0; transform: translate(0,0); }}
      18% {{ opacity: 1; transform: translate(0,0); }}
      38% {{ opacity: 1; transform: translate(148px,46px); }}
      62% {{ opacity: 0.98; transform: translate(302px,48px); }}
      78%,100% {{ opacity: 0; transform: translate(444px,46px); }}
    }}
    @keyframes {uid}-p15 {{
      0%,14% {{ opacity: 0; transform: translate(0,0); }}
      20% {{ opacity: 1; transform: translate(0,0); }}
      40% {{ opacity: 1; transform: translate(150px,58px); }}
      64% {{ opacity: 0.98; transform: translate(304px,60px); }}
      80%,100% {{ opacity: 0; transform: translate(446px,58px); }}
    }}
    @keyframes {uid}-p16 {{
      0%,16% {{ opacity: 0; transform: translate(0,0); }}
      22% {{ opacity: 1; transform: translate(0,0); }}
      42% {{ opacity: 1; transform: translate(152px,70px); }}
      66% {{ opacity: 0.98; transform: translate(306px,72px); }}
      82%,100% {{ opacity: 0; transform: translate(448px,70px); }}
    }}

    @media (max-width: 1100px) {{
      #{uid} .timeline {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
    }}
    @media (max-width: 760px) {{
      #{uid} .header {{ flex-direction: column; }}
      #{uid} .timeline {{ grid-template-columns: 1fr; }}
    }}
  </style>

  <div class="header">
    <div>
      <span class="chip">Scientific pathway animation</span>
      <h3>IDH mutation to glioma outcome with synchronized molecular flow.</h3>
      <p>Left: mutant metabolism. Center: nuclear methylation. Right: altered RNA expression and tumor behavior.</p>
    </div>
    <div style="display:grid;gap:10px;justify-items:end;">
      <span class="chip">{escape(mode_chip)}</span>
      <div class="controls">
        <button type="button" data-action="play">Play</button>
        <button type="button" data-action="pause">Pause</button>
        <button type="button" data-action="restart">Restart</button>
        <button type="button" data-action="slow">0.5x</button>
        <button type="button" data-action="normal">1x</button>
      </div>
    </div>
  </div>

  <div class="scene">
    <svg viewBox="0 0 1280 600" role="img" aria-label="Detailed animated IDH mutation pathway from mitochondria to glioma outcome">
      <defs></defs>

      <rect class="zone-fill" x="60" y="84" width="340" height="392" rx="30" />
      <rect class="zone-fill" x="470" y="84" width="340" height="392" rx="30" />
      <rect class="zone-fill" x="880" y="84" width="340" height="392" rx="30" />
      <text class="zone-title" x="82" y="116">Left zone</text>
      <text class="zone-title" x="492" y="116">Center zone</text>
      <text class="zone-title" x="904" y="116">Right zone</text>

      <g class="fg-primary">
        <ellipse class="mito-glow" cx="230" cy="282" rx="142" ry="108" />
        <g class="mito-pulse">
          <path class="mito-shape" d="M116,286 C122,214 170,176 244,178 C316,180 366,220 374,286 C382,346 338,392 268,402 C194,412 130,378 116,286 Z" />
          <path class="mito-cristae" d="M146,272 C170,232 198,230 222,270 C246,310 278,308 304,268 C330,230 354,228 372,264" />
          <path class="mito-cristae" d="M150,306 C174,274 200,272 222,304 C246,336 276,336 302,304 C328,272 352,272 372,302" />
          <path class="mito-cristae" d="M146,238 C170,208 196,206 220,236 C244,266 274,266 300,234 C326,206 350,206 372,238" />
        </g>
        <g class="enzyme-pulse">
          <circle class="enzyme-core" cx="298" cy="196" r="18" />
          <path class="enzyme-star" d="M298,170 l6,14 14,6 -14,6 -6,14 -6,-14 -14,-6 14,-6z" />
        </g>
        <g>
          <circle class="small-molecule" cx="126" cy="188" r="16" />
          <circle class="small-molecule" cx="150" cy="188" r="16" />
          <circle class="small-molecule" cx="138" cy="168" r="16" />
          <text class="small-molecule-text" x="116" y="192">a-KG</text>
        </g>

        <ellipse class="nucleus-glow" cx="640" cy="282" rx="154" ry="120" />
        <g class="dna-alive">
          <ellipse class="nucleus-shell" cx="640" cy="282" rx="136" ry="112" />
          <ellipse class="nucleus-ring" cx="640" cy="282" rx="122" ry="100" />
          <path class="dna-rail-a" d="M566,236 C592,214 618,214 640,236 C664,258 690,258 716,234" />
          <path class="dna-rail-b" d="M564,272 C590,296 616,298 640,274 C664,250 690,252 716,276" />
          <path class="dna-rail-a" d="M570,306 C594,284 620,284 640,306 C664,328 690,328 714,306" />
          <path class="dna-rung" d="M584,224 L584,280" />
          <path class="dna-rung" d="M608,220 L608,292" />
          <path class="dna-rung" d="M640,236 L640,316" />
          <path class="dna-rung" d="M670,222 L670,294" />
          <path class="dna-rung" d="M696,226 L696,302" />
          <circle class="nucleosome" cx="574" cy="252" r="9" />
          <circle class="nucleosome" cx="616" cy="244" r="9" />
          <circle class="nucleosome" cx="658" cy="254" r="9" />
          <circle class="nucleosome" cx="694" cy="246" r="9" />
          <g class="me1"><circle class="me-dot" cx="576" cy="214" r="13" /><text class="me-text" x="567" y="218">Me</text></g>
          <g class="me2"><circle class="me-dot" cx="620" cy="202" r="13" /><text class="me-text" x="611" y="206">Me</text></g>
          <g class="me3"><circle class="me-dot" cx="662" cy="214" r="13" /><text class="me-text" x="653" y="218">Me</text></g>
          <g class="me4"><circle class="me-dot" cx="700" cy="258" r="13" /><text class="me-text" x="691" y="262">Me</text></g>
          <g class="me5"><circle class="me-dot" cx="618" cy="324" r="13" /><text class="me-text" x="609" y="328">Me</text></g>
          <g>
            <circle class="inhibit-ring" cx="748" cy="230" r="14" />
            <path class="inhibit-bar" d="M739,221 l18,18" />
            <circle class="inhibit-ring" cx="768" cy="270" r="14" />
            <path class="inhibit-bar" d="M759,261 l18,18" />
          </g>
        </g>

        <g class="rna-alive">
          <path class="rna-core" d="M842,270 C880,248 912,248 942,272 C972,296 1002,304 1040,290" />
          <path class="rna-core" d="M848,320 C878,302 906,302 934,320 C962,338 992,344 1024,338" />
          <circle class="polymerase" cx="850" cy="268" r="10" />
          <circle class="rna-bead" cx="878" cy="254" r="6" />
          <circle class="rna-bead" cx="914" cy="256" r="6" />
          <circle class="rna-bead" cx="954" cy="282" r="6" />
          <circle class="rna-bead" cx="994" cy="298" r="6" />
          <circle class="rna-bead" cx="884" cy="316" r="6" />
          <circle class="rna-bead" cx="938" cy="328" r="6" />
          <circle class="rna-bead" cx="992" cy="338" r="6" />
        </g>

        <g class="tumor-alive">
          <ellipse class="tumor-shadow" cx="1106" cy="354" rx="72" ry="26" />
          <circle class="outcome-cell" cx="1100" cy="246" r="34" />
          <circle class="outcome-cell" cx="1138" cy="238" r="30" />
          <circle class="outcome-cell" cx="1162" cy="272" r="28" />
          <circle class="outcome-cell" cx="1128" cy="282" r="30" />
          <circle class="outcome-core" cx="1100" cy="246" r="9" />
          <circle class="outcome-core" cx="1138" cy="238" r="9" />
          <circle class="outcome-core" cx="1128" cy="282" r="9" />
          <path class="outcome-branch" d="M1178,296 C1192,306 1200,322 1204,340" />
        </g>

        <path class="flow-line-blue" d="M374,278 C452,278 512,278 566,278 C598,278 620,280 640,282" />
        <polygon class="flow-head-blue" points="640,282 620,270 620,294" />
        <path class="flow-line-orange" d="M776,282 C828,282 870,282 908,282 C938,282 962,284 984,288" />
        <polygon class="flow-head-orange" points="984,288 964,276 964,300" />
        <path class="flow-line-red" d="M1040,282 C1064,282 1080,282 1090,282" />
        <polygon class="flow-head-red" points="1090,282 1070,270 1070,294" />

        <g class="p1"><circle class="particle" cx="388" cy="264" r="8" /></g>
        <g class="p2"><circle class="particle" cx="392" cy="276" r="7" /></g>
        <g class="p3"><circle class="particle" cx="396" cy="264" r="8" /></g>
        <g class="p4"><circle class="particle" cx="400" cy="278" r="7" /></g>
        <g class="p5"><circle class="particle" cx="404" cy="266" r="8" /></g>
        <g class="p6"><circle class="particle" cx="408" cy="280" r="7" /></g>
        <g class="p7"><circle class="particle" cx="388" cy="298" r="7" /></g>
        <g class="p8"><circle class="particle" cx="392" cy="308" r="7" /></g>
        <g class="p9"><circle class="particle" cx="396" cy="318" r="7" /></g>
        <g class="p10"><circle class="particle" cx="400" cy="328" r="7" /></g>
        <g class="p11"><circle class="particle" cx="404" cy="338" r="7" /></g>
        <g class="p12"><circle class="particle" cx="408" cy="348" r="7" /></g>
        <g class="p13"><circle class="particle" cx="412" cy="268" r="8" /></g>
        <g class="p14"><circle class="particle" cx="416" cy="282" r="7" /></g>
        <g class="p15"><circle class="particle" cx="420" cy="296" r="7" /></g>
        <g class="p16"><circle class="particle" cx="424" cy="310" r="7" /></g>

        <g transform="translate(112 146)">
          <line class="label-line" x1="152" y1="38" x2="186" y2="92" />
          <rect class="label-box" width="136" height="34" rx="14" />
          <text class="label-title" x="16" y="22">IDH mutation</text>
        </g>
        <g transform="translate(408 356)">
          <line class="label-line" x1="68" y1="0" x2="58" y2="-64" />
          <rect class="label-box" width="88" height="34" rx="14" />
          <text class="label-title" x="20" y="22">2-HG</text>
        </g>
        <g transform="translate(560 138)">
          <line class="label-line" x1="100" y1="34" x2="98" y2="72" />
          <rect class="label-box" width="160" height="34" rx="14" />
          <text class="label-title" x="16" y="22">DNA methylation</text>
        </g>
        <g transform="translate(866 164)">
          <line class="label-line" x1="98" y1="34" x2="96" y2="86" />
          <rect class="label-box" width="170" height="34" rx="14" />
          <text class="label-title" x="16" y="22">Altered RNA expression</text>
        </g>
        <g transform="translate(1038 154)">
          <line class="label-line" x1="82" y1="34" x2="80" y2="58" />
          <rect class="label-box" width="136" height="34" rx="14" />
          <text class="label-title" x="16" y="22">Glioma outcome</text>
        </g>
      </g>
    </svg>

    <div class="timeline">
      <div class="step"><strong>1. IDH mutation</strong>Mitochondrial metabolism</div>
      <div class="step"><strong>2. 2-HG</strong>Continuous blue particle stream</div>
      <div class="step"><strong>3. DNA methylation</strong>Epigenetic remodeling</div>
      <div class="step"><strong>4. RNA expression</strong>Active transcription output</div>
      <div class="step"><strong>5. Glioma outcome</strong>Tumor-state consequence</div>
    </div>
  </div>

  <div style="display:grid;gap:10px;">
    <div class="legend">
      <span><i style="background:#14a86f"></i>IDH / mitochondria</span>
      <span><i style="background:#267fff"></i>2-HG</span>
      <span><i style="background:#7b4ce8"></i>Nucleus / DNA</span>
      <span><i style="background:#ff9526"></i>RNA / expression</span>
      <span><i style="background:#e7597d"></i>Tumor outcome</span>
    </div>
    <div class="footer-copy"><strong>{escape(context['title'])}</strong> {escape(context['message'])}</div>
    <div class="gene-row">{driver_markup}</div>
    <div class="footer-copy"><strong>Prediction-aware note:</strong> {escape(note)}</div>
    <div class="educational-note">This visualization is educational and is not a clinical diagnosis.</div>
  </div>

  <script>
    (() => {{
      const root = document.getElementById({uid!r});
      if (!root || root.dataset.bound === '1') return;
      root.dataset.bound = '1';
      const setPlay = (state) => root.style.setProperty('--play', state);
      const setSpeed = (seconds) => root.style.setProperty('--speed', seconds);
      root.querySelectorAll('[data-action]').forEach((button) => {{
        button.addEventListener('click', () => {{
          const action = button.getAttribute('data-action');
          if (action === 'play') setPlay('running');
          if (action === 'pause') setPlay('paused');
          if (action === 'slow') setSpeed('24s');
          if (action === 'normal') setSpeed('14s');
          if (action === 'restart') {{
            setPlay('paused');
            const scene = root.querySelector('.scene');
            if (scene) {{
              scene.style.display = 'none';
              void scene.offsetHeight;
              scene.style.display = '';
            }}
            requestAnimationFrame(() => setPlay('running'));
          }}
        }});
      }});
    }})();
  </script>
</section>
"""


def pathway_context_html(predictions: pd.DataFrame | None = None) -> str:
    context = _prediction_context(predictions)
    genes = context["genes"] or DEFAULT_GENES[:5]
    gene_markup = "".join(f"<span>{escape(gene)}</span>" for gene in genes[:5])
    return f"""
<section class="pathway-context-compact-clean {context['mode']}">
  <style>
    .pathway-context-compact-clean {{
      display: grid;
      gap: 12px;
      padding: 18px 20px;
      border: 1px solid rgba(108, 136, 165, 0.2);
      border-radius: 24px;
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,251,255,0.98));
      box-shadow: 0 16px 34px rgba(15,23,42,0.05);
      color: #142a42;
    }}
    .pathway-context-compact-clean .topline {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .pathway-context-compact-clean .context-chip {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 0 12px;
      border-radius: 999px;
      border: 1px solid rgba(108, 136, 165, 0.22);
      background: rgba(255,255,255,0.96);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      color: #142a42;
    }}
    .pathway-context-compact-clean p {{ margin: 0; color: #4e647a; font-size: 13px; line-height: 1.55; }}
    .pathway-context-compact-clean strong {{ color: #142a42; }}
    .pathway-context-compact-clean .legend-row,
    .pathway-context-compact-clean .gene-row {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .pathway-context-compact-clean .legend-row span,
    .pathway-context-compact-clean .gene-row span {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 30px;
      padding: 0 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.96);
      border: 1px solid rgba(108, 136, 165, 0.18);
      font-size: 12px;
      color: #3e556e;
      font-weight: 760;
    }}
    .pathway-context-compact-clean .legend-row i {{ width: 11px; height: 11px; border-radius: 999px; display: inline-block; }}
  </style>
  <div class="topline">
    <span class="context-chip">Pathway legend</span>
    <span class="context-chip">{escape(context['title'])}</span>
  </div>
  <div class="legend-row">
    <span><i style="background:#14a86f"></i>IDH / mitochondria</span>
    <span><i style="background:#267fff"></i>2-HG</span>
    <span><i style="background:#7b4ce8"></i>Nucleus / DNA</span>
    <span><i style="background:#ff9526"></i>RNA / expression</span>
    <span><i style="background:#e7597d"></i>Outcome</span>
  </div>
  <p><strong>Interpretation:</strong> {escape(context['message'])}</p>
  <div>
    <strong>Global expression drivers</strong>
    <div class="gene-row">{gene_markup}</div>
  </div>
  <p>This visualization is educational and should not be used as a clinical diagnosis.</p>
</section>
"""
