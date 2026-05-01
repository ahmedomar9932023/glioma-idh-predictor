from __future__ import annotations

import json
from html import escape
from uuid import uuid4

import pandas as pd

from .pathway import _prediction_context, pathway_illustration_html


def pathway_three_runtime_js() -> str:
    return r"""
(() => {
  if (window.__gliomaThreeRuntimeInstalled) return;
  window.__gliomaThreeRuntimeInstalled = true;

  const CDN_SRC = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js";

  const loadThree = () => {
    if (window.THREE) return Promise.resolve(window.THREE);
    if (window.__gliomaThreeLoadPromise) return window.__gliomaThreeLoadPromise;
    window.__gliomaThreeLoadPromise = new Promise((resolve, reject) => {
      const existing = document.querySelector(`script[src="${CDN_SRC}"]`);
      if (existing) {
        existing.addEventListener("load", () => resolve(window.THREE));
        existing.addEventListener("error", () => reject(new Error("Three.js CDN failed")));
        return;
      }
      const script = document.createElement("script");
      script.src = CDN_SRC;
      script.async = true;
      script.onload = () => resolve(window.THREE);
      script.onerror = () => reject(new Error("Three.js CDN failed"));
      document.head.appendChild(script);
    });
    return window.__gliomaThreeLoadPromise;
  };

  const smoothstep = (a, b, x) => {
    const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
    return t * t * (3 - 2 * t);
  };

  const installScene = async (root) => {
    if (!root || root.dataset.threeReady === "1" || root.dataset.threeInit === "1") return;
    root.dataset.threeInit = "1";
    const stage = root.querySelector(".webgl-stage");
    const overlay = root.querySelector(".overlay");
    const tooltip = root.querySelector(".tooltip");
    const fallback = root.querySelector(".fallback");
    const data = JSON.parse(root.dataset.context || "{}");
    const mode = data.mode || "background";
    const genes = data.genes || [];
    const stepCards = Array.from(root.querySelectorAll(".step-chip"));
    const playButton = root.querySelector('[data-action="play"]');
    const pauseButton = root.querySelector('[data-action="pause"]');
    const restartButton = root.querySelector('[data-action="restart"]');
    const speedButtons = Array.from(root.querySelectorAll("[data-speed]"));

    const showFallback = (reason) => {
      if (stage) stage.style.display = "none";
      if (fallback) {
        fallback.style.display = "block";
        const note = fallback.querySelector(".error-note");
        if (note && reason) note.textContent = `Three.js renderer unavailable: ${reason}. Showing SVG fallback instead.`;
      }
      root.dataset.threeInit = "0";
    };

    try {
      await loadThree();
      if (!window.THREE) throw new Error("Three.js did not load");
      if (!stage || !overlay || !tooltip) throw new Error("Renderer container missing");
      const THREE = window.THREE;

      const rect = stage.getBoundingClientRect();
      const width = Math.max(320, Math.round(rect.width || stage.clientWidth || 960));
      const height = Math.max(560, Math.round(rect.height || stage.clientHeight || 760));

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(width, height, false);
      if ("outputColorSpace" in renderer && THREE.SRGBColorSpace) {
        renderer.outputColorSpace = THREE.SRGBColorSpace;
      }
      renderer.domElement.style.width = "100%";
      renderer.domElement.style.height = `${height}px`;
      renderer.domElement.style.position = "relative";
      renderer.domElement.style.zIndex = "1";
      stage.prepend(renderer.domElement);

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(34, width / height, 0.1, 200);
      camera.position.set(0, 2.2, 46);
      camera.lookAt(0, 0, 0);

      scene.add(new THREE.HemisphereLight(0xf7fcff, 0xdbe8f6, 1.28));
      const key = new THREE.DirectionalLight(0xffffff, 1.18);
      key.position.set(12, 16, 18);
      scene.add(key);
      const rim = new THREE.PointLight(0x9ed8ff, 0.52, 100);
      rim.position.set(-18, -8, 14);
      scene.add(rim);

      const state = {
        playing: true,
        speed: 1,
        manualStep: null,
        start: performance.now(),
        pauseAt: null,
        mouse: new THREE.Vector2(-10, -10),
      };
      const raycaster = new THREE.Raycaster();
      const interactive = [];
      const labels = [];
      const callouts = [];
      const rootGroup = new THREE.Group();
      scene.add(rootGroup);

      const addInteractive = (mesh, title, body) => {
        mesh.userData.tooltip = `<strong>${title}</strong><br>${body}`;
        interactive.push(mesh);
        return mesh;
      };
      const createLabel = (text, position, tooltipHtml) => {
        const el = document.createElement("div");
        el.className = "label";
        el.textContent = text;
        if (tooltipHtml) {
          el.addEventListener("mouseenter", () => {
            tooltip.innerHTML = tooltipHtml;
            tooltip.style.opacity = "1";
          });
          el.addEventListener("mouseleave", () => {
            tooltip.style.opacity = "0";
          });
        }
        overlay.appendChild(el);
        labels.push({ el, position: position.clone(), kind: "label" });
      };
      const createDriverChip = (text, position) => {
        const el = document.createElement("div");
        el.className = "driver-chip";
        el.textContent = text;
        overlay.appendChild(el);
        labels.push({ el, position: position.clone(), kind: "driver" });
      };
      const createCallout = (html, anchor, stepIndex) => {
        const el = document.createElement("div");
        el.className = "callout";
        el.innerHTML = html;
        overlay.appendChild(el);
        callouts.push({ el, anchor: anchor.clone(), stepIndex });
      };
      const project = (vec3) => {
        const projected = vec3.clone().project(camera);
        return {
          x: (projected.x * 0.5 + 0.5) * renderer.domElement.clientWidth,
          y: (-projected.y * 0.5 + 0.5) * renderer.domElement.clientHeight,
        };
      };
      const getStepWeights = (t) => {
        const times = [0.00, 0.125, 0.25, 0.375, 0.56];
        return times.map((start) => {
          const fadeIn = smoothstep(start, start + 0.10, t);
          const fadeOut = 1 - smoothstep(start + 0.42, start + 0.56, t);
          return Math.max(0.12, fadeIn * fadeOut);
        });
      };

      const cellGroup = new THREE.Group();
      rootGroup.add(cellGroup);

      const cellGeom = new THREE.SphereGeometry(14, 56, 56);
      cellGeom.scale(1.75, 1.08, 1.14);
      const cell = new THREE.Mesh(
        cellGeom,
        new THREE.MeshPhysicalMaterial({
          color: 0xbbeff8,
          transparent: true,
          opacity: 0.16,
          roughness: 0.18,
          transmission: 0.18,
          thickness: 0.5,
          clearcoat: 0.6,
          clearcoatRoughness: 0.4,
          side: THREE.DoubleSide,
        })
      );
      cell.rotation.set(-0.08, -0.18, 0.02);
      cellGroup.add(cell);

      const cytoplasm = new THREE.Mesh(
        new THREE.SphereGeometry(13.5, 48, 48),
        new THREE.MeshPhongMaterial({
          color: 0xdaf7fb,
          transparent: true,
          opacity: 0.10,
          shininess: 16,
          side: THREE.DoubleSide,
        })
      );
      cytoplasm.scale.set(1.63, 1.0, 1.05);
      cytoplasm.rotation.copy(cell.rotation);
      cellGroup.add(cytoplasm);

      for (let i = 0; i < 28; i++) {
        const haze = new THREE.Mesh(
          new THREE.SphereGeometry(0.24 + (i % 3) * 0.05, 12, 12),
          new THREE.MeshBasicMaterial({
            color: 0xd7f4f8,
            transparent: true,
            opacity: 0.10 + (i % 4) * 0.02,
          })
        );
        haze.position.set(-18 + (i % 7) * 5.6 + ((i * 13) % 10) * 0.06, -8 + (i % 5) * 3.6, -7 + (i % 6) * 2.6);
        cellGroup.add(haze);
      }

      const makeBackgroundMito = (x, y, z, sx, sy, rotZ) => {
        const g = new THREE.Group();
        const body = new THREE.Mesh(
          new THREE.SphereGeometry(1.9, 20, 20),
          new THREE.MeshPhongMaterial({
            color: 0x2aa7a0,
            transparent: true,
            opacity: 0.18,
            shininess: 14,
          })
        );
        body.scale.set(sx, sy, 0.82);
        g.add(body);
        for (let i = 0; i < 3; i++) {
          const cr = new THREE.Mesh(
            new THREE.TorusGeometry(0.6 + i * 0.1, 0.05, 6, 22, Math.PI * 1.2),
            new THREE.MeshBasicMaterial({
              color: 0x9cefd7,
              transparent: true,
              opacity: 0.18,
            })
          );
          cr.rotation.set(1.4, 0.2, 0.4 + i * 0.18);
          cr.position.set(-0.3 + i * 0.42, 0.0, 0.08);
          g.add(cr);
        }
        g.position.set(x, y, z);
        g.rotation.z = rotZ;
        cellGroup.add(g);
      };
      makeBackgroundMito(-15, -7, -4.2, 1.3, 0.72, 0.42);
      makeBackgroundMito(16, 8, -3.5, 1.12, 0.66, -0.66);
      makeBackgroundMito(20, -3, -5.4, 1.0, 0.56, 0.22);

      const makeRibbon = (points, color, opacity, radius, segments) => {
        const curve = new THREE.CatmullRomCurve3(points);
        return new THREE.Mesh(
          new THREE.TubeGeometry(curve, segments, radius, 10, false),
          new THREE.MeshPhongMaterial({
            color,
            transparent: true,
            opacity,
            shininess: 12,
          })
        );
      };

      const erGroup = new THREE.Group();
      erGroup.add(makeRibbon([
        new THREE.Vector3(-5, -10, -1.6), new THREE.Vector3(-1, -8.8, -1.8), new THREE.Vector3(3, -9.6, -1.2), new THREE.Vector3(7, -8.0, -1.6),
      ], 0xd99df2, 0.22, 0.18, 64));
      erGroup.add(makeRibbon([
        new THREE.Vector3(-6, -8.4, -1.2), new THREE.Vector3(-2, -7.2, -1.4), new THREE.Vector3(2, -7.6, -1.0), new THREE.Vector3(6, -6.4, -1.2),
      ], 0xe7c0f6, 0.20, 0.16, 64));
      erGroup.add(makeRibbon([
        new THREE.Vector3(2, 6.6, -2.2), new THREE.Vector3(5, 7.4, -2.0), new THREE.Vector3(9, 6.4, -1.8), new THREE.Vector3(12, 7.2, -1.9),
      ], 0xd7dff8, 0.10, 0.12, 52));
      cellGroup.add(erGroup);

      const golgiGroup = new THREE.Group();
      golgiGroup.add(makeRibbon([
        new THREE.Vector3(-20, -4.4, -1.8), new THREE.Vector3(-17.8, -3.1, -1.6), new THREE.Vector3(-15.0, -4.6, -1.4), new THREE.Vector3(-12.8, -3.4, -1.5),
      ], 0x7fb5ff, 0.20, 0.16, 48));
      golgiGroup.add(makeRibbon([
        new THREE.Vector3(-20.8, -5.8, -1.4), new THREE.Vector3(-17.6, -4.6, -1.2), new THREE.Vector3(-14.2, -5.8, -1.0), new THREE.Vector3(-11.2, -4.4, -1.1),
      ], 0x93c5ff, 0.18, 0.14, 48));
      cellGroup.add(golgiGroup);

      const ribosomeMat = new THREE.MeshBasicMaterial({ color: 0xd85e8c, transparent: true, opacity: 0.28 });
      for (let i = 0; i < 30; i++) {
        const rib = new THREE.Mesh(new THREE.SphereGeometry(0.12, 10, 10), ribosomeMat);
        rib.position.set(-8 + (i % 10) * 1.45, -10 + ((i * 3) % 7) * 0.8, -0.8 - (i % 3) * 0.2);
        cellGroup.add(rib);
      }

      const vesicleMat = new THREE.MeshPhongMaterial({ color: 0xcfeef8, transparent: true, opacity: 0.22, shininess: 16 });
      for (let i = 0; i < 9; i++) {
        const v = new THREE.Mesh(new THREE.SphereGeometry(0.42 + (i % 3) * 0.06, 18, 18), vesicleMat);
        v.position.set(-18 + (i % 5) * 8.2, -2 + ((i * 7) % 6) * 2.2, -4.8 + (i % 4));
        cellGroup.add(v);
      }

      const mainMito = new THREE.Group();
      const mitoBody = new THREE.Mesh(
        new THREE.SphereGeometry(2.6, 28, 24),
        new THREE.MeshPhysicalMaterial({
          color: 0x13a7a1,
          roughness: 0.24,
          metalness: 0.02,
          transparent: true,
          opacity: 0.98,
          clearcoat: 0.3,
        })
      );
      mitoBody.scale.set(2.2, 1.14, 1.0);
      mainMito.add(mitoBody);
      for (let i = 0; i < 4; i++) {
        const curve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(-3.2 + i * 1.4, -0.9, 0.25),
          new THREE.Vector3(-2.8 + i * 1.4, 0.3, 0.15),
          new THREE.Vector3(-2.3 + i * 1.4, -0.2, 0.05),
          new THREE.Vector3(-1.9 + i * 1.4, 0.8, 0.18),
        ]);
        const crista = new THREE.Mesh(
          new THREE.TubeGeometry(curve, 32, 0.09, 10, false),
          new THREE.MeshBasicMaterial({ color: 0xa7ffe3, transparent: true, opacity: 0.78 })
        );
        crista.rotation.z = 0.22;
        mainMito.add(crista);
      }
      mainMito.position.set(16, 8, 6);
      mainMito.rotation.set(0.14, -0.18, -0.36);
      rootGroup.add(addInteractive(mainMito, "IDH mutation", "The cascade begins in a mitochondrion-like compartment where mutant IDH activity is emphasized."));

      const idhMarker = new THREE.Mesh(
        new THREE.SphereGeometry(0.78, 18, 18),
        new THREE.MeshPhongMaterial({ color: 0x7fe43d, emissive: 0x2f5c18, emissiveIntensity: 0.8 })
      );
      idhMarker.position.set(18.8, 11.4, 9.2);
      rootGroup.add(idhMarker);

      const akgBadge = new THREE.Mesh(
        new THREE.SphereGeometry(0.48, 16, 16),
        new THREE.MeshPhongMaterial({ color: 0xe7edf6, transparent: true, opacity: 0.52 })
      );
      akgBadge.position.set(10.6, 5.2, 5.6);
      rootGroup.add(addInteractive(akgBadge, "alpha-KG", "alpha-ketoglutarate is shown as a faded reference metabolite for contrast with 2-HG."));

      const nucleusGroup = new THREE.Group();
      const nucleus = new THREE.Mesh(
        new THREE.SphereGeometry(5.9, 36, 36),
        new THREE.MeshPhysicalMaterial({
          color: 0x8f5cff,
          roughness: 0.26,
          transparent: true,
          opacity: 0.94,
          clearcoat: 0.5,
        })
      );
      nucleusGroup.add(nucleus);
      const envelope = new THREE.Mesh(
        new THREE.SphereGeometry(6.55, 34, 34),
        new THREE.MeshPhongMaterial({
          color: 0xc8b2ff,
          transparent: true,
          opacity: 0.34,
          shininess: 18,
        })
      );
      nucleusGroup.add(envelope);
      nucleusGroup.position.set(-1.8, 2.4, 2.5);
      rootGroup.add(addInteractive(nucleusGroup, "Nucleus", "The nucleus contains chromatin and the epigenetic remodeling effects that follow 2-HG accumulation."));

      const dnaGroup = new THREE.Group();
      const addHelix = (offsetZ, color, radius = 0.11) => {
        const pts = [];
        for (let i = 0; i < 120; i++) {
          const t = i / 119;
          const angle = t * Math.PI * 6.0;
          pts.push(new THREE.Vector3(Math.sin(angle) * 2.0, (t - 0.5) * 6.2, Math.cos(angle) * 0.9 + offsetZ));
        }
        const curve = new THREE.CatmullRomCurve3(pts);
        return new THREE.Mesh(
          new THREE.TubeGeometry(curve, 120, radius, 12, false),
          new THREE.MeshPhongMaterial({ color, transparent: true, opacity: 0.92, shininess: 18 })
        );
      };
      dnaGroup.add(addHelix(-0.45, 0x5c23bf, 0.12));
      dnaGroup.add(addHelix(0.45, 0x9f73f4, 0.12));
      dnaGroup.position.set(-1.2, 2.1, 3.2);
      dnaGroup.rotation.set(0.22, -0.12, 0.38);
      rootGroup.add(addInteractive(dnaGroup, "DNA / epigenetic remodeling", "Chromatin glows and methylation markers accumulate as the cascade reaches the nucleus."));

      const meDots = [];
      for (let i = 0; i < 5; i++) {
        const dot = new THREE.Mesh(
          new THREE.SphereGeometry(0.34, 14, 14),
          new THREE.MeshPhongMaterial({ color: 0xda46ff, transparent: true, opacity: 0.08, emissive: 0x7a20a9, emissiveIntensity: 0.2 })
        );
        dot.position.set(-2.8 + i * 1.3, 0.8 + (i % 2) * 1.4, 5.8 - (i % 3) * 0.7);
        rootGroup.add(dot);
        meDots.push(dot);
      }

      const blockIcons = [];
      for (let i = 0; i < 3; i++) {
        const circle = new THREE.Mesh(
          new THREE.RingGeometry(0.24, 0.42, 24),
          new THREE.MeshBasicMaterial({ color: 0xd95371, transparent: true, opacity: 0.08, side: THREE.DoubleSide })
        );
        const slash = new THREE.Mesh(
          new THREE.BoxGeometry(0.08, 0.58, 0.08),
          new THREE.MeshBasicMaterial({ color: 0xd95371, transparent: true, opacity: 0.08 })
        );
        const group = new THREE.Group();
        group.add(circle);
        group.add(slash);
        slash.rotation.z = 0.72;
        group.position.set(4.6 + i * 0.9, 4.0 - i * 1.6, 5.4);
        rootGroup.add(group);
        blockIcons.push(group);
      }

      const flowCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(15.4, 8.0, 7.2),
        new THREE.Vector3(10.4, 6.4, 6.4),
        new THREE.Vector3(6.6, 5.2, 5.4),
        new THREE.Vector3(2.0, 4.1, 4.6),
        new THREE.Vector3(-1.8, 3.0, 4.0),
      ]);
      const flowPath = new THREE.Mesh(
        new THREE.TubeGeometry(flowCurve, 120, 0.12, 10, false),
        new THREE.MeshBasicMaterial({
          color: 0x58a8ff,
          transparent: true,
          opacity: mode === "wildtype" ? 0.22 : 0.54,
        })
      );
      rootGroup.add(flowPath);

      const particles = [];
      for (let i = 0; i < 11; i++) {
        const p = new THREE.Mesh(
          new THREE.SphereGeometry(0.34 + (i % 3) * 0.04, 14, 14),
          new THREE.MeshPhongMaterial({
            color: 0x2d82ff,
            emissive: 0x2d82ff,
            emissiveIntensity: 0.30,
            transparent: true,
            opacity: 0.92,
          })
        );
        rootGroup.add(p);
        particles.push(p);
      }
      addInteractive(particles[0], "2-HG", "Blue 2-HG particles travel from the metabolic compartment toward the nucleus in overlapping waves.");

      const rnaCurveA = new THREE.CatmullRomCurve3([
        new THREE.Vector3(-4.0, -1.6, 3.2),
        new THREE.Vector3(-7.0, -3.2, 4.0),
        new THREE.Vector3(-10.4, -4.8, 4.4),
        new THREE.Vector3(-13.2, -6.2, 4.8),
      ]);
      const rnaCurveB = new THREE.CatmullRomCurve3([
        new THREE.Vector3(-3.2, -0.6, 2.6),
        new THREE.Vector3(-7.0, -1.8, 3.5),
        new THREE.Vector3(-10.0, -2.2, 4.0),
        new THREE.Vector3(-13.8, -1.8, 4.4),
      ]);
      const rnaA = new THREE.Mesh(
        new THREE.TubeGeometry(rnaCurveA, 72, 0.10, 10, false),
        new THREE.MeshPhongMaterial({ color: 0xff9022, emissive: 0xff9022, emissiveIntensity: 0.08, transparent: true, opacity: 0.88 })
      );
      const rnaB = new THREE.Mesh(
        new THREE.TubeGeometry(rnaCurveB, 72, 0.10, 10, false),
        new THREE.MeshPhongMaterial({ color: 0xffc964, emissive: 0xffc964, emissiveIntensity: 0.06, transparent: true, opacity: 0.82 })
      );
      rootGroup.add(rnaA, rnaB);
      addInteractive(rnaA, "Altered gene expression", "RNA strands brighten and extend after epigenetic remodeling shifts the expression program.");
      const rnaBeads = [];
      for (let i = 0; i < 5; i++) {
        const bead = new THREE.Mesh(
          new THREE.SphereGeometry(0.20, 12, 12),
          new THREE.MeshPhongMaterial({ color: 0xff9022, emissive: 0xff9022, emissiveIntensity: 0.12 })
        );
        rootGroup.add(bead);
        rnaBeads.push(bead);
      }

      const outcomeGroup = new THREE.Group();
      const outcomeOffsets = [
        [-1.1, 0.4, 0.2, 1.1], [0.6, 1.1, 0.0, 1.2], [1.4, -0.2, 0.4, 0.9], [-0.2, -1.1, 0.1, 1.0], [1.8, -1.2, 0.2, 0.72],
      ];
      outcomeOffsets.forEach(([x, y, z, s]) => {
        const blob = new THREE.Mesh(
          new THREE.SphereGeometry(1.0, 20, 20),
          new THREE.MeshPhongMaterial({
            color: 0xf55e85,
            emissive: 0xcc365b,
            emissiveIntensity: 0.08,
            transparent: true,
            opacity: 0.96,
          })
        );
        blob.scale.setScalar(s);
        blob.position.set(x, y, z);
        outcomeGroup.add(blob);
      });
      outcomeGroup.position.set(-16.8, -9.4, 7.0);
      rootGroup.add(addInteractive(outcomeGroup, "Glioma behavior / outcome", "A soft cell cluster suggests downstream glioma behavior as the expression program shifts."));

      const outcomeParticles = [];
      for (let i = 0; i < 6; i++) {
        const bead = new THREE.Mesh(
          new THREE.SphereGeometry(0.16, 10, 10),
          new THREE.MeshBasicMaterial({ color: 0xf55e85, transparent: true, opacity: 0.0 })
        );
        rootGroup.add(bead);
        outcomeParticles.push(bead);
      }

      const debugSphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.42, 18, 18),
        new THREE.MeshBasicMaterial({ color: 0xffdd00 })
      );
      debugSphere.position.set(0, 0, 0);
      rootGroup.add(debugSphere);

      createLabel("IDH1/2 mutation", new THREE.Vector3(19.4, 13.0, 9.2), "<strong>IDH mutation</strong><br>Mutant IDH activity is emphasized in the mitochondrial compartment.");
      createLabel("2-HG accumulation", new THREE.Vector3(7.0, 3.8, 5.0), "<strong>2-HG</strong><br>Particles diffuse inward in overlapping waves instead of one discrete burst.");
      createLabel("DNA / epigenetic remodeling", new THREE.Vector3(6.8, 10.2, 6.0), "<strong>Epigenetics</strong><br>2-HG affects alpha-KG-dependent enzymes and promotes methylation change.");
      createLabel("Altered gene expression", new THREE.Vector3(-11.8, -5.2, 5.0), "<strong>Expression</strong><br>Orange RNA strands brighten as chromatin state shifts.");
      createLabel("Glioma behavior / outcome", new THREE.Vector3(-18.4, -4.8, 7.4), "<strong>Outcome</strong><br>Rounded tumor-like cells pulse to suggest downstream behavior.");
      genes.forEach((gene, index) => createDriverChip(gene, new THREE.Vector3(-8.2 - index * 1.2, -2.0 - index * 1.4, 5.4)));

      createCallout("<strong>Step 1</strong><br>IDH1/2 mutation changes enzyme activity.", new THREE.Vector3(17.6, 15.4, 8.8), 0);
      createCallout("<strong>Step 2</strong><br>Mutant IDH produces 2-HG, an oncometabolite.", new THREE.Vector3(7.8, 0.2, 5.6), 1);
      createCallout("<strong>Step 3</strong><br>2-HG affects alpha-KG-dependent enzymes and promotes methylation changes.", new THREE.Vector3(8.0, 13.6, 6.6), 2);
      createCallout("<strong>Step 4</strong><br>Epigenetic remodeling shifts gene-expression programs.", new THREE.Vector3(-11.6, -9.4, 5.2), 3);
      createCallout("<strong>Step 5</strong><br>Altered expression patterns influence glioma behavior and prognosis.", new THREE.Vector3(-18.2, -0.8, 7.6), 4);

      const updateControls = () => {
        if (playButton) playButton.classList.toggle("active", state.playing);
        if (pauseButton) pauseButton.classList.toggle("active", !state.playing);
        speedButtons.forEach((button) => {
          button.classList.toggle("active", Number(button.dataset.speed) === state.speed);
        });
      };
      updateControls();

      playButton && playButton.addEventListener("click", () => {
        if (!state.playing) {
          const now = performance.now();
          if (state.pauseAt != null) state.start += now - state.pauseAt;
        }
        state.playing = true;
        state.pauseAt = null;
        state.manualStep = null;
        updateControls();
      });
      pauseButton && pauseButton.addEventListener("click", () => {
        if (state.playing) state.pauseAt = performance.now();
        state.playing = false;
        updateControls();
      });
      restartButton && restartButton.addEventListener("click", () => {
        state.start = performance.now();
        state.manualStep = null;
        state.playing = true;
        state.pauseAt = null;
        updateControls();
      });
      speedButtons.forEach((button) => button.addEventListener("click", () => {
        state.speed = Number(button.dataset.speed) || 1;
        updateControls();
      }));
      stepCards.forEach((card) => card.addEventListener("click", () => {
        state.manualStep = Number(card.dataset.step);
        state.playing = false;
        state.pauseAt = performance.now();
        updateControls();
      }));

      renderer.domElement.addEventListener("mousemove", (event) => {
        const rectLocal = renderer.domElement.getBoundingClientRect();
        state.mouse.x = ((event.clientX - rectLocal.left) / rectLocal.width) * 2 - 1;
        state.mouse.y = -((event.clientY - rectLocal.top) / rectLocal.height) * 2 + 1;
        tooltip.style.left = `${event.clientX - rectLocal.left}px`;
        tooltip.style.top = `${event.clientY - rectLocal.top}px`;
      });
      renderer.domElement.addEventListener("mouseleave", () => {
        state.mouse.set(-10, -10);
        tooltip.style.opacity = "0";
      });

      const resize = () => {
        const stageRect = stage.getBoundingClientRect();
        const w = Math.max(320, Math.round(stageRect.width || stage.clientWidth || 960));
        const h = Math.max(560, Math.min(760, Math.round((stageRect.width || 960) * 0.62)));
        renderer.setSize(w, h, false);
        renderer.domElement.style.height = `${h}px`;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      };
      window.addEventListener("resize", resize);
      resize();

      const duration = 16;
      const animate = () => {
        if (!root.isConnected) return;
        requestAnimationFrame(animate);
        const now = performance.now();
        const elapsed = ((state.playing ? now : state.pauseAt || now) - state.start) / 1000 * state.speed;
        const cycle = ((elapsed % duration) + duration) % duration / duration;
        const weights = getStepWeights(cycle);
        const activeStep = state.manualStep != null ? state.manualStep : weights.indexOf(Math.max(...weights));
        stepCards.forEach((card, index) => card.classList.toggle("active", index === activeStep));

        const emphasis = mode === "wildtype" ? 0.58 : mode === "mutant" ? 1.0 : 0.82;
        const mitoPulse = 1 + 0.05 * Math.sin(elapsed * 1.4);
        mainMito.scale.setScalar(mitoPulse);
        idhMarker.scale.setScalar(0.92 + 0.14 * Math.sin(elapsed * 1.9 + 0.6));
        idhMarker.material.emissiveIntensity = 0.35 + weights[0] * 0.85;
        akgBadge.material.opacity = 0.12 + (1 - weights[1]) * 0.34;

        const flowStrength = (0.18 + weights[1] * 0.82) * emphasis;
        flowPath.material.opacity = flowStrength * 0.66;
        particles.forEach((particle, index) => {
          const local = (cycle * 1.45 + index * 0.09) % 1;
          const point = flowCurve.getPoint(local);
          particle.position.copy(point);
          particle.scale.setScalar(0.84 + 0.28 * Math.sin((elapsed + index) * 2.2));
          const alpha = (local > 0.05 && local < 0.96 ? 0.22 + flowStrength * 0.78 : 0.0) * (1 - Math.abs(local - 0.5) * 0.75);
          particle.material.opacity = Math.max(0, alpha);
          particle.material.emissiveIntensity = 0.16 + flowStrength * 0.42;
        });

        nucleus.material.emissive = new THREE.Color(0x4a22a5);
        nucleus.material.emissiveIntensity = 0.04 + weights[2] * 0.18;
        envelope.material.opacity = 0.26 + weights[2] * 0.16;
        dnaGroup.children.forEach((strand, index) => {
          strand.material.opacity = 0.72 + weights[2] * 0.24;
          strand.material.emissive = new THREE.Color(index ? 0x6d36dc : 0x4b1ca1);
          strand.material.emissiveIntensity = 0.04 + weights[2] * 0.18;
        });
        meDots.forEach((dot, index) => {
          const w = Math.max(0, smoothstep(0.28 + index * 0.03, 0.52 + index * 0.03, cycle));
          dot.material.opacity = 0.08 + w * 0.84 * emphasis;
          dot.material.emissiveIntensity = 0.10 + w * 0.22;
        });
        blockIcons.forEach((group, index) => {
          const w = Math.max(0, smoothstep(0.34 + index * 0.03, 0.58 + index * 0.03, cycle));
          group.children.forEach((part) => {
            part.material.opacity = 0.08 + w * 0.72 * emphasis;
          });
        });

        const rnaStrength = (0.12 + weights[3] * 0.88) * emphasis;
        [rnaA, rnaB].forEach((strand, index) => {
          strand.material.opacity = (0.30 + rnaStrength * 0.62) * (index === 0 ? 1 : 0.92);
          strand.material.emissiveIntensity = 0.04 + rnaStrength * 0.16;
        });
        rnaBeads.forEach((bead, index) => {
          const t = (cycle * 1.2 + index * 0.16) % 1;
          const curve = index % 2 === 0 ? rnaCurveA : rnaCurveB;
          bead.position.copy(curve.getPoint(t));
          bead.material.opacity = t > 0.2 ? 0.24 + rnaStrength * 0.76 : 0.0;
        });

        const outcomeStrength = (0.14 + weights[4] * 0.86) * emphasis;
        outcomeGroup.scale.setScalar(0.88 + outcomeStrength * 0.22);
        outcomeGroup.children.forEach((blob, index) => {
          blob.material.emissiveIntensity = 0.04 + outcomeStrength * 0.10;
          blob.material.opacity = 0.78 + outcomeStrength * 0.18;
          blob.position.z = [-1,0,1,2,3][index] * 0.1 + Math.sin(elapsed * 1.8 + index) * 0.08;
        });
        outcomeParticles.forEach((bead, index) => {
          bead.position.set(-15.0 + index * 0.9, -9.0 + Math.sin(elapsed * 1.4 + index) * 1.1, 8.0 + Math.cos(elapsed * 1.1 + index) * 0.6);
          bead.material.opacity = 0.04 + outcomeStrength * 0.46;
        });

        callouts.forEach((entry) => {
          const pos = project(entry.anchor);
          entry.el.style.left = `${pos.x}px`;
          entry.el.style.top = `${pos.y}px`;
          const alpha = state.manualStep == null ? (entry.stepIndex === activeStep ? 1 : 0.22 + weights[entry.stepIndex] * 0.36) : (entry.stepIndex === state.manualStep ? 1 : 0.16);
          entry.el.style.opacity = String(alpha);
        });
        labels.forEach((entry) => {
          const pos = project(entry.position);
          entry.el.style.left = `${pos.x}px`;
          entry.el.style.top = `${pos.y}px`;
        });
        overlay.querySelectorAll(".label").forEach((label, index) => {
          label.classList.toggle("current", index === activeStep);
        });

        raycaster.setFromCamera(state.mouse, camera);
        const hit = raycaster.intersectObjects(interactive, true)[0];
        if (hit && hit.object.userData.tooltip) {
          tooltip.innerHTML = hit.object.userData.tooltip;
          tooltip.style.opacity = "1";
        } else if (!overlay.querySelector(".label:hover")) {
          tooltip.style.opacity = "0";
        }

        camera.position.x = Math.sin(elapsed * 0.12) * 0.6;
        camera.lookAt(0, 0, 0);
        renderer.render(scene, camera);
      };

      root.dataset.threeReady = "1";
      animate();
    } catch (error) {
      console.error("Glioma Three.js pathway error:", error);
      showFallback(error && error.message ? error.message : "renderer initialization failed");
    }
  };

  const initAll = () => {
    document.querySelectorAll(".three-pathway-shell").forEach((root) => installScene(root));
  };

  const observer = new MutationObserver(() => initAll());
  observer.observe(document.body, { childList: true, subtree: true });
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAll, { once: true });
  } else {
    initAll();
  }
})();
"""


def pathway_three_html(predictions: pd.DataFrame | None = None) -> str:
    context = _prediction_context(predictions)
    mode = context["mode"]
    scene_id = f"idh-cell-{uuid4().hex}"
    fallback_id = f"{scene_id}-fallback"
    note = (
        "Mutant-like result: the pathway is fully emphasized."
        if mode == "mutant"
        else "Wildtype-like result: the pathway is softened and shown as educational context."
        if mode == "wildtype"
        else "Interactive WebGL medical explainer: IDH mutation to glioma behavior."
    )
    genes = (context.get("genes") or ["IDH1", "IDH2", "MGMT"])[:3]
    payload = {
        "mode": mode,
        "title": context["title"],
        "message": context["message"],
        "note": note,
        "genes": genes,
    }
    fallback_markup = pathway_illustration_html(predictions)
    return f"""
    <section class="three-pathway-shell {escape(mode)}" id="{scene_id}" data-context='{escape(json.dumps(payload))}'>
      <style>
        .three-pathway-shell {{
          --navy: #142a42;
          --muted: #65788d;
          --line: rgba(124, 163, 196, 0.24);
          --panel: rgba(255,255,255,0.86);
          --shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
          color: var(--navy);
          display: grid;
          gap: 14px;
        }}
        .three-pathway-shell .pathway-topbar {{
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 14px;
        }}
        .three-pathway-shell .pathway-topbar h3 {{
          margin: 8px 0 4px;
          font-size: 24px;
          color: var(--navy);
        }}
        .three-pathway-shell .pathway-topbar p {{
          margin: 0;
          color: var(--muted);
          font-size: 14px;
        }}
        .three-pathway-shell .kicker,
        .three-pathway-shell .mode-chip {{
          display: inline-flex;
          align-items: center;
          min-height: 30px;
          padding: 0 12px;
          border-radius: 999px;
          border: 1px solid var(--line);
          background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(244,249,255,0.94));
          color: var(--navy);
          font-size: 12px;
          font-weight: 760;
        }}
        .three-pathway-shell .controls {{
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
        }}
        .three-pathway-shell .controls button {{
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-height: 38px;
          padding: 0 12px;
          border: 1px solid var(--line);
          border-radius: 12px;
          background: linear-gradient(145deg, rgba(255,255,255,0.98), rgba(244,249,255,0.94));
          color: var(--navy);
          font: inherit;
          font-size: 12px;
          font-weight: 760;
          cursor: pointer;
          box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
          transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
        }}
        .three-pathway-shell .controls button:hover {{
          transform: translateY(-1px);
          box-shadow: 0 12px 22px rgba(15, 23, 42, 0.08);
          border-color: rgba(37, 99, 235, 0.22);
        }}
        .three-pathway-shell .controls button.active {{
          background: linear-gradient(135deg, rgba(226,239,255,0.98), rgba(243,248,255,0.98));
          border-color: rgba(37, 99, 235, 0.24);
        }}
        .three-pathway-shell .webgl-stage {{
          position: relative;
          min-height: 760px;
          overflow: hidden;
          border: 1px solid var(--line);
          border-radius: 30px;
          background:
            radial-gradient(circle at 50% 42%, rgba(255,255,255,0.10), transparent 22%),
            radial-gradient(circle at 50% 46%, transparent 38%, rgba(118, 172, 201, 0.08) 76%, rgba(65, 99, 126, 0.12) 100%),
            linear-gradient(180deg, rgba(255,255,255,0.99), rgba(245,250,255,0.99));
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.90), var(--shadow);
        }}
        .three-pathway-shell canvas {{
          display: block;
          width: 100%;
          height: 760px;
        }}
        .three-pathway-shell .overlay,
        .three-pathway-shell .timeline,
        .three-pathway-shell .legend {{
          position: absolute;
          inset: 0;
          pointer-events: none;
        }}
        .three-pathway-shell .label {{
          position: absolute;
          display: inline-flex;
          align-items: center;
          min-height: 26px;
          padding: 0 9px;
          border-radius: 999px;
          background: rgba(255,255,255,0.94);
          border: 1px solid rgba(124,163,196,0.20);
          color: var(--navy);
          font-size: 10px;
          font-weight: 760;
          box-shadow: 0 8px 16px rgba(15,23,42,0.05);
          transform: translate(-50%, -50%);
          white-space: nowrap;
          pointer-events: auto;
        }}
        .three-pathway-shell .label.current {{
          background: rgba(246, 250, 255, 0.98);
          border-color: rgba(37,99,235,0.20);
          box-shadow: 0 12px 20px rgba(15,23,42,0.08);
        }}
        .three-pathway-shell .driver-chip {{
          position: absolute;
          transform: translate(-50%, -50%);
          min-height: 24px;
          padding: 0 8px;
          border-radius: 999px;
          background: rgba(255,255,255,0.90);
          border: 1px solid rgba(124,163,196,0.18);
          color: var(--navy);
          font-size: 10px;
          font-weight: 760;
          pointer-events: none;
        }}
        .three-pathway-shell .callout {{
          position: absolute;
          max-width: 220px;
          padding: 12px 14px;
          border-radius: 18px;
          background: rgba(255,255,255,0.96);
          border: 1px solid rgba(124,163,196,0.20);
          color: var(--navy);
          font-size: 12px;
          line-height: 1.5;
          box-shadow: 0 14px 24px rgba(15,23,42,0.06);
          transform: translate(-50%, -50%);
          pointer-events: none;
        }}
        .three-pathway-shell .tooltip {{
          position: absolute;
          max-width: 220px;
          padding: 10px 12px;
          border-radius: 14px;
          border: 1px solid rgba(124,163,196,0.20);
          background: rgba(255,255,255,0.97);
          color: var(--navy);
          font-size: 12px;
          line-height: 1.45;
          box-shadow: 0 12px 24px rgba(15,23,42,0.08);
          opacity: 0;
          transform: translate(14px, 14px);
          transition: opacity 140ms ease;
          pointer-events: none;
          z-index: 8;
        }}
        .three-pathway-shell .timeline {{
          inset: auto 4% 3.4% 4%;
          display: grid;
          grid-template-columns: repeat(5, minmax(0, 1fr));
          gap: 10px;
          z-index: 5;
          pointer-events: auto;
        }}
        .three-pathway-shell .step-chip {{
          display: grid;
          gap: 4px;
          padding: 10px 10px 9px;
          border: 1px solid rgba(124,163,196,0.18);
          border-radius: 16px;
          background: rgba(255,255,255,0.84);
          cursor: pointer;
          transition: box-shadow 140ms ease, border-color 140ms ease, transform 140ms ease;
        }}
        .three-pathway-shell .step-chip:hover {{
          transform: translateY(-1px);
          box-shadow: 0 12px 22px rgba(15,23,42,0.06);
        }}
        .three-pathway-shell .step-chip.active {{
          background: rgba(239,247,255,0.98);
          border-color: rgba(37,99,235,0.18);
          box-shadow: 0 12px 24px rgba(15,23,42,0.05);
        }}
        .three-pathway-shell .step-chip strong {{
          color: var(--navy);
          font-size: 12px;
        }}
        .three-pathway-shell .step-chip span {{
          color: var(--muted);
          font-size: 11px;
          line-height: 1.35;
        }}
        .three-pathway-shell .step-no {{
          display: inline-grid;
          place-items: center;
          width: 22px;
          height: 22px;
          border-radius: 50%;
          color: #fff;
          font-size: 10px;
          font-weight: 850;
        }}
        .three-pathway-shell .sn1 {{ background: #0f8b8e; }}
        .three-pathway-shell .sn2 {{ background: #2d82ff; }}
        .three-pathway-shell .sn3 {{ background: #7441e6; }}
        .three-pathway-shell .sn4 {{ background: #ff9122; }}
        .three-pathway-shell .sn5 {{ background: #f55e85; }}
        .three-pathway-shell .legend {{
          inset: 18px 20px auto auto;
          width: 180px;
          display: grid;
          gap: 10px;
          padding: 16px;
          border-radius: 18px;
          background: rgba(255,255,255,0.82);
          border: 1px solid rgba(124,163,196,0.18);
          box-shadow: 0 12px 24px rgba(15,23,42,0.05);
          z-index: 5;
          pointer-events: none;
        }}
        .three-pathway-shell .legend strong {{
          font-size: 12px;
          color: var(--navy);
        }}
        .three-pathway-shell .legend span {{
          display: flex;
          align-items: center;
          gap: 8px;
          color: var(--muted);
          font-size: 12px;
        }}
        .three-pathway-shell .legend i {{
          width: 12px;
          height: 12px;
          border-radius: 50%;
          display: inline-block;
        }}
        .three-pathway-shell .fallback {{
          display: none;
        }}
        .three-pathway-shell .error-note {{
          margin: 8px 0 0;
          color: var(--muted);
          font-size: 12px;
        }}
        @media (max-width: 980px) {{
          .three-pathway-shell .timeline {{
            grid-template-columns: 1fr 1fr;
          }}
          .three-pathway-shell .legend {{
            width: 160px;
          }}
          .three-pathway-shell canvas,
          .three-pathway-shell .webgl-stage {{
            min-height: 680px;
          }}
        }}
        @media (max-width: 720px) {{
          .three-pathway-shell .pathway-topbar {{
            flex-direction: column;
          }}
          .three-pathway-shell .controls {{
            flex-direction: column;
          }}
          .three-pathway-shell .timeline {{
            grid-template-columns: 1fr;
          }}
          .three-pathway-shell canvas,
          .three-pathway-shell .webgl-stage {{
            min-height: 560px;
          }}
          .three-pathway-shell .legend {{
            inset: auto 14px 118px 14px;
            width: auto;
            grid-template-columns: 1fr 1fr;
          }}
        }}
      </style>

      <div class="pathway-topbar">
        <div>
          <span class="kicker">Interactive WebGL explainer</span>
          <h3>IDH mutation pathway inside a glioma cell</h3>
          <p>{escape(note)}</p>
        </div>
        <span class="mode-chip">{escape(context["title"])}</span>
      </div>

      <div class="controls">
        <button type="button" data-action="play" class="active">Play</button>
        <button type="button" data-action="pause">Pause</button>
        <button type="button" data-action="restart">Restart</button>
        <button type="button" data-speed="1" class="active">1x</button>
        <button type="button" data-speed="0.5">0.5x</button>
      </div>

      <div class="webgl-stage">
        <div class="overlay"></div>
        <div class="tooltip"></div>
        <div class="legend">
          <strong>Legend</strong>
          <span><i style="background:#0f8b8e"></i>IDH / mitochondria</span>
          <span><i style="background:#2d82ff"></i>2-HG</span>
          <span><i style="background:#7441e6"></i>Nucleus / DNA</span>
          <span><i style="background:#ff9122"></i>RNA / expression</span>
          <span><i style="background:#f55e85"></i>Outcome</span>
          <span><i style="background:#97a8bc"></i>Background organelles</span>
        </div>
        <div class="timeline">
          <article class="step-chip" data-step="0"><span class="step-no sn1">1</span><strong>IDH mutation</strong><span>Mitochondrial enzyme activity changes.</span></article>
          <article class="step-chip" data-step="1"><span class="step-no sn2">2</span><strong>2-HG</strong><span>Oncometabolite accumulates and moves inward.</span></article>
          <article class="step-chip" data-step="2"><span class="step-no sn3">3</span><strong>Epigenetics</strong><span>Methylation-like remodeling appears.</span></article>
          <article class="step-chip" data-step="3"><span class="step-no sn4">4</span><strong>Expression</strong><span>RNA output increases after chromatin changes.</span></article>
          <article class="step-chip" data-step="4"><span class="step-no sn5">5</span><strong>Outcome</strong><span>Glioma behavior emerges at the end of the loop.</span></article>
        </div>
      </div>
      <div class="fallback" id="{fallback_id}">
        {fallback_markup}
        <p class="error-note">Three.js could not be loaded, so the SVG fallback is shown.</p>
      </div>
    </section>
    """
