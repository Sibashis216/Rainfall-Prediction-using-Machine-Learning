(function () {
  "use strict";

  const GAUGE_MAX = 80;           // mm — full-scale reading on the dial
  const CX = 120, CY = 130, R = 100;
  const ARC_LEN = Math.PI * R;    // semicircle length, matches stroke-dasharray

  const els = {
    sliders: Array.from(document.querySelectorAll(".feature-slider")),
    predictBtn: document.getElementById("predictBtn"),
    errorMsg: document.getElementById("errorMsg"),
    mmValue: document.getElementById("mmValue"),
    gaugeCategory: document.getElementById("gaugeCategory"),
    gaugeRange: document.getElementById("gaugeRange"),
    gaugeFill: document.getElementById("gaugeFill"),
    gaugeNeedle: document.getElementById("gaugeNeedle"),
    gaugeTicks: document.getElementById("gaugeTicks"),
    statusDot: document.querySelector("#statusLight .dot"),
    statusText: document.getElementById("statusText"),
    metricTrainR2: document.getElementById("metricTrainR2"),
    metricTestR2: document.getElementById("metricTestR2"),
    metricMae: document.getElementById("metricMae"),
    metricSizes: document.getElementById("metricSizes"),
    importanceBars: document.getElementById("importanceBars"),
  };

  const CATEGORY_COLOR = {
    "No Rain": "#6FCF8E",
    "Light Rain": "#5CB8D6",
    "Moderate Rain": "#4FA9D6",
    "Heavy Rain": "#C9A15C",
    "Extreme Rain": "#E2703A",
  };

  const NICE_NAMES = {
    "tempavg": "Temperature",
    "DPavg": "Dew Point",
    "humidity avg": "Humidity",
    "SLPavg": "Sea-Level Pressure",
    "visibilityavg": "Visibility",
    "windavg": "Wind",
  };

  /* ── Gauge: draw static tick marks ─────────────────── */
  function drawTicks() {
    const steps = 4;
    let svgNS = "http://www.w3.org/2000/svg";
    for (let i = 0; i <= steps; i++) {
      const frac = i / steps;
      const angle = -180 + frac * 180; // -180deg (left) -> 0deg (right), measured from +x axis
      const rad = (angle * Math.PI) / 180;
      const rInner = R - 20;
      const x = CX + rInner * Math.cos(rad);
      const y = CY + rInner * Math.sin(rad);
      const label = Math.round(frac * GAUGE_MAX);

      const text = document.createElementNS(svgNS, "text");
      text.setAttribute("x", x);
      text.setAttribute("y", y + 3);
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("class", "tick-label");
      text.textContent = label;
      els.gaugeTicks.appendChild(text);
    }
  }

  function setGauge(mm, lo, hi, category) {
    const clamped = Math.min(mm, GAUGE_MAX);
    const frac = clamped / GAUGE_MAX;

    // Needle: -90deg (value 0, pointing left-ish along vertical baseline) to +90deg
    const needleDeg = -90 + frac * 180;
    els.gaugeNeedle.style.transform = `rotate(${needleDeg}deg)`;

    // Arc fill
    const offset = ARC_LEN - frac * ARC_LEN;
    els.gaugeFill.style.strokeDashoffset = offset;

    const color = CATEGORY_COLOR[category] || "#5CB8D6";
    els.gaugeFill.style.stroke = color;
    els.gaugeCategory.style.color = color;
    els.gaugeCategory.textContent = category;

    els.mmValue.textContent = mm.toFixed(1);
    els.gaugeRange.textContent = `Range: ${lo.toFixed(1)} – ${hi.toFixed(1)} mm (10th–90th pct.)`;
  }

  /* ── Sliders ────────────────────────────────────────── */
  function formatSliderValue(feature, raw) {
    return Number(raw).toFixed(1);
  }

  function bindSliders() {
    els.sliders.forEach((slider, i) => {
      const valueEl = document.getElementById(`val_${i + 1}`);
      const unit = slider.dataset.unit || "";
      const update = () => {
        valueEl.textContent = `${formatSliderValue(slider.dataset.feature, slider.value)}${unit}`;
      };
      update();
      slider.addEventListener("input", update);
    });
  }

  function collectFeatureValues() {
    const payload = {};
    els.sliders.forEach((slider) => {
      payload[slider.dataset.feature] = Number(slider.value);
    });
    return payload;
  }

  /* ── API calls ──────────────────────────────────────── */
  async function loadMeta() {
    try {
      const res = await fetch("/api/meta");
      if (!res.ok) throw new Error("meta endpoint unavailable");
      const data = await res.json();

      const m = data.metrics;
      els.metricTrainR2.textContent = m.train_r2.toFixed(2);
      els.metricTestR2.textContent = m.test_r2.toFixed(2);
      els.metricMae.textContent = `${m.test_mae.toFixed(2)} mm`;
      els.metricSizes.textContent = `${m.n_train} / ${m.n_test}`;

      renderImportances(data.importances);

      els.statusDot.classList.add("live");
      els.statusText.textContent = "Model ready";
      els.predictBtn.disabled = false;
    } catch (err) {
      els.statusText.textContent = "Model not trained — run train_model.py";
      els.predictBtn.disabled = true;
    }
  }

  function renderImportances(importances) {
    const entries = Object.entries(importances).sort((a, b) => b[1] - a[1]);
    const max = Math.max(...entries.map(([, v]) => v)) || 1;
    els.importanceBars.innerHTML = "";
    entries.forEach(([feature, value]) => {
      const row = document.createElement("div");
      row.className = "imp-row";
      row.innerHTML = `
        <span class="imp-label">${NICE_NAMES[feature] || feature}</span>
        <span class="imp-track"><span class="imp-fill"></span></span>
        <span class="imp-pct">${(value * 100).toFixed(1)}%</span>
      `;
      els.importanceBars.appendChild(row);
      requestAnimationFrame(() => {
        row.querySelector(".imp-fill").style.width = `${(value / max) * 100}%`;
      });
    });
  }

  async function runPrediction() {
    els.errorMsg.textContent = "";
    els.predictBtn.disabled = true;
    els.predictBtn.querySelector("span").textContent = "Reading instruments…";

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectFeatureValues()),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed");

      setGauge(data.rainfall_mm, data.range_low, data.range_high, data.category);
    } catch (err) {
      els.errorMsg.textContent = err.message;
    } finally {
      els.predictBtn.disabled = false;
      els.predictBtn.querySelector("span").textContent = "Run Prediction";
    }
  }

  /* ── Init ───────────────────────────────────────────── */
  drawTicks();
  bindSliders();
  loadMeta();
  els.predictBtn.addEventListener("click", runPrediction);
})();
