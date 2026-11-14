// Small SVG chart kit shared by the CAP and ProAct pages.
//
// Every chart renders at its container's pixel width (so text stays at true
// size), re-renders on resize, carries a hover/focus tooltip, and has a
// table-view twin wired by mountChartCard(). No dependencies.

(function () {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const css = (name) => getComputedStyle(document.body || document.documentElement).getPropertyValue(name).trim();

  function el(name, attrs = {}, parent) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [key, value] of Object.entries(attrs)) {
      if (value !== undefined && value !== null) node.setAttribute(key, String(value));
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  function text(parent, content, attrs = {}) {
    const node = el("text", attrs, parent);
    node.textContent = content;
    return node;
  }

  function linear([d0, d1], [r0, r1]) {
    const scale = (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
    scale.domain = [d0, d1];
    return scale;
  }

  function log10Scale([d0, d1], [r0, r1]) {
    const l0 = Math.log10(d0);
    const l1 = Math.log10(d1);
    const scale = (v) => r0 + ((Math.log10(v) - l0) / (l1 - l0 || 1)) * (r1 - r0);
    scale.domain = [d0, d1];
    return scale;
  }

  function niceStep(span, count) {
    const raw = span / Math.max(1, count);
    const mag = 10 ** Math.floor(Math.log10(raw));
    const norm = raw / mag;
    return (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  }

  function ticks([d0, d1], count = 5) {
    const step = niceStep(d1 - d0, count);
    const out = [];
    for (let v = Math.ceil(d0 / step) * step; v <= d1 + step * 1e-9; v += step) {
      out.push(Number(v.toFixed(10)));
    }
    return out;
  }

  function fmt(value, digits = 1, unit = "") {
    if (value === null || value === undefined || Number.isNaN(value)) return "–";
    return `${Number(value).toFixed(digits)}${unit}`;
  }

  function signed(value, digits = 1, unit = "") {
    if (value === null || value === undefined || Number.isNaN(value)) return "–";
    const sign = value > 0 ? "+" : value < 0 ? "−" : "±";
    return `${sign}${Math.abs(value).toFixed(digits)}${unit}`;
  }

  // ---------- tooltip (singleton) ----------

  let tip;
  function tooltipNode() {
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "viz-tooltip";
      tip.setAttribute("role", "status");
      document.body.appendChild(tip);
    }
    return tip;
  }

  // content: {title, rows: [{color, value, name}], foot}
  function showTooltip(anchor, content, event) {
    const node = tooltipNode();
    node.replaceChildren();
    if (content.title) {
      const title = document.createElement("div");
      title.className = "tt-title";
      title.textContent = content.title;
      node.appendChild(title);
    }
    for (const row of content.rows || []) {
      const line = document.createElement("div");
      line.className = "tt-row";
      const key = document.createElement("span");
      key.className = "tt-key";
      key.style.background = row.color || "transparent";
      const value = document.createElement("span");
      value.className = "tt-value";
      value.textContent = row.value;
      const name = document.createElement("span");
      name.className = "tt-name";
      name.textContent = row.name || "";
      line.append(key, value, name);
      node.appendChild(line);
    }
    if (content.foot) {
      const foot = document.createElement("div");
      foot.className = "tt-foot";
      foot.textContent = content.foot;
      node.appendChild(foot);
    }

    const box = anchor.getBoundingClientRect();
    const x = event && event.clientX !== undefined && event.clientX !== 0 ? event.clientX : box.left + box.width / 2;
    const y = event && event.clientY !== undefined && event.clientY !== 0 ? event.clientY : box.top;
    node.classList.add("show");
    const w = node.offsetWidth;
    const h = node.offsetHeight;
    let left = x + 14;
    if (left + w > window.innerWidth - 8) left = x - w - 14;
    let top = y - h - 12;
    if (top < 8) top = y + 18;
    node.style.left = `${Math.max(8, left)}px`;
    node.style.top = `${top}px`;
  }

  function hideTooltip() {
    if (tip) tip.classList.remove("show");
  }

  // Attach hover + keyboard-focus tooltip behaviour to an SVG node.
  function bindTip(node, content, { onClick, onEnter, onLeave } = {}) {
    node.setAttribute("tabindex", "0");
    node.setAttribute("data-focusable", "");
    const getContent = typeof content === "function" ? content : () => content;
    node.addEventListener("pointermove", (event) => showTooltip(node, getContent(), event));
    node.addEventListener("pointerenter", () => onEnter && onEnter());
    node.addEventListener("pointerleave", () => {
      hideTooltip();
      if (onLeave) onLeave();
    });
    node.addEventListener("focus", () => {
      showTooltip(node, getContent());
      if (onEnter) onEnter();
    });
    node.addEventListener("blur", () => {
      hideTooltip();
      if (onLeave) onLeave();
    });
    if (onClick) {
      node.style.cursor = "pointer";
      node.addEventListener("click", onClick);
      node.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onClick(event);
        }
      });
    }
  }

  // ---------- responsive mount ----------

  // Calls draw(width) now and whenever the host's width changes.
  function responsive(host, draw) {
    let last = 0;
    const run = () => {
      const width = Math.round(host.clientWidth);
      if (!width || width === last) return;
      last = width;
      hideTooltip(); // the hovered mark is about to be replaced and will never fire pointerleave
      host.replaceChildren();
      draw(width);
    };
    run();
    new ResizeObserver(run).observe(host);
    return () => {
      last = 0;
      run();
    };
  }

  function svgRoot(host, width, height, label) {
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img", "aria-label": label });
    host.appendChild(svg);
    return svg;
  }

  // ---------- color ----------

  function hexToRgb(hex) {
    const v = hex.replace("#", "");
    return [0, 2, 4].map((i) => parseInt(v.slice(i, i + 2), 16));
  }

  function mix(a, b, t) {
    const ca = hexToRgb(a);
    const cb = hexToRgb(b);
    const c = ca.map((x, i) => Math.round(x + (cb[i] - x) * t));
    return `#${c.map((x) => x.toString(16).padStart(2, "0")).join("")}`;
  }

  // Diverging: red arm (negative) <- gray midpoint -> blue arm (positive).
  const DIVERGING = {
    neg: ["#f0efec", "#f3c3bd", "#e8867c", "#c8433b", "#9b2621"],
    pos: ["#f0efec", "#c4dbf6", "#7fb0ec", "#2a78d6", "#184f95"],
  };

  function rampAt(stops, t) {
    const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(x));
    return mix(stops[i], stops[i + 1], x - i);
  }

  // CAP page: orange arm = CAP better (the paper's "ours" hue), blue arm = baseline better.
  const DIVERGING_CAP = {
    neg: ["#f0efec", "#c4dbf6", "#7fb0ec", "#2a78d6", "#184f95"],
    pos: ["#f0efec", "#f9d3bf", "#f3a179", "#eb6834", "#b4461a"],
  };

  function diverging(value, maxAbs, ramp = DIVERGING) {
    const t = Math.min(1, Math.abs(value) / (maxAbs || 1));
    return rampAt(value >= 0 ? ramp.pos : ramp.neg, t);
  }

  function inkFor(fill) {
    const [r, g, b] = hexToRgb(fill).map((v) => {
      const s = v / 255;
      return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
    });
    const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    return lum > 0.36 ? "#20211f" : "#ffffff";
  }

  // ---------- chart card: legend + table toggle ----------

  // card: element containing .chart, optional .legend and .data-table-wrap.
  // table: {columns: [label...], rows: [[cell...]...], bestCol?: fn(row)->index}
  function renderTable(card, table) {
    let wrap = card.querySelector(".data-table-wrap");
    if (!wrap) {
      wrap = document.createElement("div");
      wrap.className = "data-table-wrap";
      wrap.hidden = true;
      card.appendChild(wrap);
    }
    const tableEl = document.createElement("table");
    tableEl.className = "data-table";
    const thead = tableEl.createTHead().insertRow();
    for (const col of table.columns) {
      const th = document.createElement("th");
      th.textContent = col;
      thead.appendChild(th);
    }
    const tbody = tableEl.createTBody();
    for (const row of table.rows) {
      const tr = tbody.insertRow();
      row.forEach((cell, i) => {
        const td = tr.insertCell();
        td.textContent = cell;
        if (table.bold && table.bold(row, i)) td.className = "best";
      });
    }
    wrap.replaceChildren(tableEl);

    let toggle = card.querySelector(".table-toggle");
    if (!toggle) {
      const head = card.querySelector(".chart-card-head");
      if (!head) return;
      let tools = head.querySelector(".chart-tools");
      if (!tools) {
        tools = document.createElement("div");
        tools.className = "chart-tools";
        head.appendChild(tools);
      }
      toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "table-toggle";
      toggle.textContent = "Table";
      toggle.setAttribute("aria-pressed", "false");
      toggle.addEventListener("click", () => {
        const on = toggle.getAttribute("aria-pressed") !== "true";
        toggle.setAttribute("aria-pressed", String(on));
        card.querySelector(".data-table-wrap").hidden = !on;
        card.querySelectorAll(".chart, .small-multiples").forEach((chart) => { chart.hidden = on; });
      });
      tools.appendChild(toggle);
    }
  }

  function renderLegend(container, items) {
    if (!container) return;
    container.replaceChildren();
    for (const item of items) {
      const span = document.createElement("span");
      span.className = "legend-item";
      const sw = document.createElement("span");
      sw.className = `legend-swatch ${item.shape || ""}`;
      if (item.gradient) {
        sw.className = "legend-ramp";
        sw.style.background = item.gradient;
      } else {
        sw.style.background = item.color;
      }
      const label = document.createElement("span");
      label.textContent = item.label;
      if (item.before) {
        const before = document.createElement("span");
        before.textContent = item.before;
        span.append(before, sw, label);
      } else {
        span.append(sw, label);
      }
      container.appendChild(span);
    }
  }

  // ---------- dumbbell: before -> after per row ----------
  //
  // opts: {rows: [{label, group, a, b, aErr, bErr, meta}], aName, bName,
  //        aColor, bColor, domain, unit, digits, onClick(row), labelWidth,
  //        rowHeight, title(row)}
  function dumbbell(host, opts) {
    return responsive(host, (width) => {
      const rows = opts.rows;
      const unit = opts.unit ?? "";
      const digits = opts.digits ?? 1;
      const aColor = opts.aColor || css("--base");
      const bColor = opts.bColor || css("--ours");
      const rowH = opts.rowHeight || 26;
      const groupGap = 26;
      const labelW = Math.min(opts.labelWidth || 150, width * 0.34);
      const deltaW = 58;
      const margin = { top: 24, right: deltaW + 8, bottom: 30, left: labelW };
      const groups = [];
      for (const row of rows) {
        let g = groups.find((x) => x.name === row.group);
        if (!g) groups.push((g = { name: row.group, rows: [] }));
        g.rows.push(row);
      }
      const hasGroups = groups.length > 1 || (groups[0] && groups[0].name);
      const height = margin.top + margin.bottom + rows.length * rowH + (hasGroups ? groups.length * groupGap : 0);
      const svg = svgRoot(host, width, height, opts.label || "Dumbbell chart");
      const domain = opts.domain || [0, 100];
      const x = linear(domain, [margin.left, width - margin.right]);
      const plotBottom = height - margin.bottom;

      const grid = el("g", {}, svg);
      for (const t of ticks(domain, width < 520 ? 4 : 6)) {
        el("line", { class: "gridline", x1: x(t), x2: x(t), y1: margin.top - 6, y2: plotBottom }, grid);
        text(grid, `${t}${unit}`, { class: "tick", x: x(t), y: plotBottom + 16, "text-anchor": "middle" });
      }
      text(svg, opts.deltaName || "Δ", { class: "axis-label", x: width - 4, y: margin.top - 10, "text-anchor": "end" });

      let y = margin.top;
      for (const group of groups) {
        if (hasGroups) {
          y += groupGap;
          text(svg, group.name, { class: "group-label", x: 0, y: y - 9 });
          el("line", { class: "baseline-rule", x1: 0, x2: width, y1: y - 4, y2: y - 4 }, svg);
        }
        for (const row of group.rows) {
          const cy = y + rowH / 2;
          const g = el("g", {}, svg);
          const bg = el("rect", { class: "row-bg", x: 0, y, width, height: rowH, rx: 4 }, g);
          text(g, row.label, { class: "row-label", x: 6, y: cy + 4 });
          const delta = row.b - row.a;
          const lo = Math.min(row.a, row.b);
          const hi = Math.max(row.a, row.b);
          el("line", {
            x1: x(lo), x2: x(hi), y1: cy, y2: cy,
            stroke: delta >= 0 ? mix(bColor, "#ffffff", 0.55) : mix(css("--bad"), "#ffffff", 0.45),
            "stroke-width": 4, "stroke-linecap": "round",
          }, g);
          el("circle", { cx: x(row.a), cy, r: 5, fill: aColor, stroke: "#fff", "stroke-width": 2 }, g);
          el("circle", { cx: x(row.b), cy, r: 6, fill: bColor, stroke: "#fff", "stroke-width": 2 }, g);
          text(g, signed(delta, digits), {
            class: `value-label${Math.abs(delta) < 0.05 ? " muted" : ""}`,
            x: width - 4, y: cy + 4, "text-anchor": "end",
          });
          const hit = el("rect", { class: "hit", x: 0, y, width, height: rowH }, g);
          bindTip(hit, () => ({
            title: opts.title ? opts.title(row) : `${row.group ? row.group + " · " : ""}${row.label}`,
            rows: [
              { color: bColor, value: fmt(row.b, digits, unit) + (row.bErr != null ? ` ± ${fmt(row.bErr, digits)}` : ""), name: opts.bName },
              { color: aColor, value: fmt(row.a, digits, unit) + (row.aErr != null ? ` ± ${fmt(row.aErr, digits)}` : ""), name: opts.aName },
            ],
            foot: `${signed(delta, digits, unit === "%" ? " pp" : unit)}${opts.clickHint && opts.onClick ? " · " + opts.clickHint : ""}`,
          }), {
            onClick: opts.onClick ? () => opts.onClick(row) : null,
            onEnter: () => bg.classList.add("hover"),
            onLeave: () => bg.classList.remove("hover"),
          });
          y += rowH;
        }
      }
    });
  }

  // ---------- heatmap of signed deltas ----------
  //
  // opts: {rows: [{key, label}], cols: [{key, label, group}], value(r, c) -> number|null,
  //        detail(r, c) -> tooltip content, onClick(r, c), maxAbs, digits, unit,
  //        selected: {row, col}}
  function heatmap(host, opts) {
    return responsive(host, (width) => {
      const labelW = Math.min(opts.labelWidth || 92, width * 0.24);
      const cols = opts.cols;
      const rows = opts.rows;
      const hasColGroups = cols.some((c) => c.group);
      const gap = 2;
      const cellW = Math.max(26, (width - labelW - 4) / cols.length);
      // wrap long column labels onto several lines instead of abbreviating them
      const wrap = (label) => {
        const maxChars = Math.max(4, Math.floor((cellW - 6) / 5.8));
        const lines = [];
        for (const word of String(label).split(" ")) {
          const last = lines[lines.length - 1];
          if (last && (last + " " + word).length <= maxChars) lines[lines.length - 1] = `${last} ${word}`;
          else lines.push(word);
        }
        return lines;
      };
      const colLines = cols.map((c) => wrap(c.label));
      const maxLines = Math.max(1, ...colLines.map((l) => l.length));
      const top = (hasColGroups ? 30 : 8) + maxLines * 13 + 6;
      const cellH = Math.min(40, Math.max(28, cellW * 0.62));
      const height = top + rows.length * cellH + 6;
      const svg = svgRoot(host, width, height, opts.label || "Heatmap");
      const digits = opts.digits ?? 1;

      if (hasColGroups) {
        let start = 0;
        for (let i = 1; i <= cols.length; i++) {
          if (i === cols.length || cols[i].group !== cols[start].group) {
            const x0 = labelW + start * cellW;
            const x1 = labelW + i * cellW - gap;
            text(svg, cols[start].group, { class: "group-label", x: (x0 + x1) / 2, y: 14, "text-anchor": "middle" });
            el("line", { class: "baseline-rule", x1: x0 + 2, x2: x1 - 2, y1: 20, y2: 20 }, svg);
            start = i;
          }
        }
      }
      cols.forEach((col, ci) => {
        const lines = colLines[ci];
        lines.forEach((line, li) => {
          text(svg, line, { class: "tick", x: labelW + ci * cellW + (cellW - gap) / 2, y: top - 8 - (lines.length - 1 - li) * 13, "text-anchor": "middle" });
        });
      });

      rows.forEach((row, ri) => {
        const y = top + ri * cellH;
        text(svg, row.label, { class: "row-label", x: labelW - 8, y: y + cellH / 2 + 4, "text-anchor": "end" });
        cols.forEach((col, ci) => {
          const v = opts.value(row, col);
          const x = labelW + ci * cellW;
          const g = el("g", {}, svg);
          const fill = v === null || v === undefined ? "#f7f7f4" : diverging(v, opts.maxAbs, opts.ramp);
          const rect = el("rect", { x, y, width: cellW - gap, height: cellH - gap, rx: 3, fill }, g);
          const selected = opts.selected && opts.selected.row === row.key && opts.selected.col === col.key;
          if (selected) {
            el("rect", { x: x + 1, y: y + 1, width: cellW - gap - 2, height: cellH - gap - 2, rx: 3, fill: "none", stroke: "#20211f", "stroke-width": 2 }, g);
          }
          if (v !== null && v !== undefined && cellW >= 34) {
            const d = cellW < 46 ? 0 : digits;
            const shown = opts.cellValue ? opts.cellValue(row, col) : v;
            const label = Number(shown.toFixed(d)) === 0 ? "0" : signed(shown, d);
            text(g, label, {
              x: x + (cellW - gap) / 2, y: y + (cellH - gap) / 2 + 4, "text-anchor": "middle",
              fill: inkFor(fill), "font-size": cellW < 46 ? 10 : 11, "font-weight": 650,
              "font-variant-numeric": "tabular-nums", "pointer-events": "none",
            });
          }
          bindTip(rect, () => opts.detail(row, col), {
            onClick: opts.onClick ? () => opts.onClick(row, col) : null,
            onEnter: () => {
              rect.setAttribute("opacity", "0.82");
              if (opts.onHover) opts.onHover(row, col);
            },
            onLeave: () => {
              rect.removeAttribute("opacity");
              if (opts.onHoverEnd) opts.onHoverEnd(row, col);
            },
          });
        });
      });
    });
  }

  // ---------- grouped bars (vertical) ----------
  //
  // opts: {categories: [label], series: [{key, label, color}], value(cat, s) -> {v, err}|number,
  //        domain, unit, digits, yLabel, label, highlight: key}
  function groupedBars(host, opts) {
    return responsive(host, (width) => {
      const height = opts.height || 260;
      const margin = { top: 18, right: 8, bottom: opts.rotateLabels ? 56 : 34, left: 44 };
      const svg = svgRoot(host, width, height, opts.label || "Bar chart");
      const unit = opts.unit ?? "";
      const digits = opts.digits ?? 1;
      const domain = opts.domain || [0, 100];
      const y = linear(domain, [height - margin.bottom, margin.top]);
      const plotW = width - margin.left - margin.right;
      const band = plotW / opts.categories.length;
      const nS = opts.series.length;
      const barW = Math.min(24, (band * 0.76 - (nS - 1) * 2) / nS);
      const groupW = nS * barW + (nS - 1) * 2;

      for (const t of ticks(domain, 4)) {
        el("line", { class: "gridline", x1: margin.left, x2: width - margin.right, y1: y(t), y2: y(t) }, svg);
        text(svg, `${t}`, { class: "tick", x: margin.left - 8, y: y(t) + 4, "text-anchor": "end" });
      }
      if (opts.yLabel) {
        text(svg, opts.yLabel, { class: "axis-label", x: margin.left, y: 10 });
      }
      el("line", { class: "baseline-rule", x1: margin.left, x2: width - margin.right, y1: y(domain[0]), y2: y(domain[0]) }, svg);

      opts.categories.forEach((cat, ci) => {
        const cx = margin.left + band * ci + band / 2;
        const lx = cx;
        const lbl = text(svg, cat, { class: "tick", x: lx, y: height - margin.bottom + 16, "text-anchor": opts.rotateLabels ? "end" : "middle" });
        if (opts.rotateLabels) lbl.setAttribute("transform", `rotate(-30 ${lx} ${height - margin.bottom + 16})`);
        opts.series.forEach((s, si) => {
          const raw = opts.value(cat, s);
          if (raw === null || raw === undefined) return;
          const v = typeof raw === "number" ? raw : raw.v;
          const err = typeof raw === "number" ? null : raw.err;
          const x0 = cx - groupW / 2 + si * (barW + 2);
          const y0 = y(Math.max(domain[0], v));
          const h = y(domain[0]) - y0;
          const r = Math.min(4, barW / 2, h);
          const path = `M${x0},${y(domain[0])} V${y0 + r} Q${x0},${y0} ${x0 + r},${y0} H${x0 + barW - r} Q${x0 + barW},${y0} ${x0 + barW},${y0 + r} V${y(domain[0])} Z`;
          const bar = el("path", { d: path, fill: s.color }, svg);
          if (err) {
            const ex = x0 + barW / 2;
            el("line", { x1: ex, x2: ex, y1: y(v - err), y2: y(Math.min(domain[1], v + err)), stroke: "#20211f", "stroke-width": 1, opacity: 0.55 }, svg);
          }
          if (opts.labelSeries && opts.labelSeries.includes(s.key)) {
            text(svg, fmt(v, digits), { class: "value-label", x: x0 + barW / 2, y: y(v + (err || 0)) - 6, "text-anchor": "middle" });
          }
          const hit = el("rect", { class: "hit", x: x0 - 1, y: margin.top, width: barW + 2, height: y(domain[0]) - margin.top }, svg);
          bindTip(hit, () => ({
            title: cat,
            rows: opts.series.map((ss) => {
              const rr = opts.value(cat, ss);
              if (rr === null || rr === undefined) return null;
              const vv = typeof rr === "number" ? rr : rr.v;
              const ee = typeof rr === "number" ? null : rr.err;
              return { color: ss.color, value: fmt(vv, digits, unit) + (ee ? ` ± ${fmt(ee, digits)}` : ""), name: ss.label };
            }).filter(Boolean),
            foot: opts.foot ? opts.foot(cat) : undefined,
          }), {
            onEnter: () => bar.setAttribute("opacity", "0.8"),
            onLeave: () => bar.removeAttribute("opacity"),
          });
        });
      });
    });
  }

  // ---------- line chart with crosshair ----------
  //
  // opts: {x: [values], xLabels?: [str], series: [{key, label, color, values, errs?}],
  //        domain, unit, digits, xLabel, yLabel, xLog}
  function lineChart(host, opts) {
    return responsive(host, (width) => {
      const height = opts.height || 240;
      const margin = { top: 20, right: opts.endLabels ? 92 : 14, bottom: 40, left: 44 };
      const svg = svgRoot(host, width, height, opts.label || "Line chart");
      const unit = opts.unit ?? "";
      const digits = opts.digits ?? 1;
      const xs = opts.x;
      const categorical = !!opts.xCategorical;
      const xr = [margin.left, width - margin.right];
      const x = categorical
        ? (v) => xr[0] + (xs.indexOf(v) / Math.max(1, xs.length - 1)) * (xr[1] - xr[0])
        : opts.xLog ? log10Scale([xs[0], xs[xs.length - 1]], xr) : linear([xs[0], xs[xs.length - 1]], xr);
      const domain = opts.domain;
      const y = linear(domain, [height - margin.bottom, margin.top]);

      for (const t of ticks(domain, 4)) {
        el("line", { class: "gridline", x1: margin.left, x2: width - margin.right, y1: y(t), y2: y(t) }, svg);
        text(svg, `${t}`, { class: "tick", x: margin.left - 8, y: y(t) + 4, "text-anchor": "end" });
      }
      xs.forEach((v, i) => {
        text(svg, opts.xLabels ? opts.xLabels[i] : `${v}`, { class: "tick", x: x(v), y: height - margin.bottom + 16, "text-anchor": "middle" });
      });
      if (opts.xLabel) text(svg, opts.xLabel, { class: "axis-label", x: (xr[0] + xr[1]) / 2, y: height - 6, "text-anchor": "middle" });
      if (opts.yLabel) text(svg, opts.yLabel, { class: "axis-label", x: margin.left, y: 10 });

      for (const ref of opts.refLines || []) {
        el("line", { x1: margin.left, x2: width - margin.right, y1: y(ref.value), y2: y(ref.value), stroke: "#20211f", "stroke-width": 1, opacity: 0.45, "stroke-dasharray": ref.dashed ? "4 4" : null }, svg);
        if (ref.label) {
          const atStart = ref.labelAt === "start";
          text(svg, ref.label, { class: "axis-label", x: atStart ? margin.left + 4 : width - margin.right, y: y(ref.value) - 5, "text-anchor": atStart ? "start" : "end" });
        }
      }

      for (const s of opts.series) {
        if (s.lo && s.hi) {
          const up = xs.map((v, i) => `${x(v)},${y(s.hi[i])}`);
          const dn = xs.map((v, i) => `${x(v)},${y(s.lo[i])}`).reverse();
          el("polygon", { points: [...up, ...dn].join(" "), fill: s.color, opacity: 0.1 }, svg);
        }
        if (s.errs) {
          const up = xs.map((v, i) => `${x(v)},${y(s.values[i] + s.errs[i])}`);
          const dn = xs.map((v, i) => `${x(v)},${y(s.values[i] - s.errs[i])}`).reverse();
          el("polygon", { points: [...up, ...dn].join(" "), fill: s.color, opacity: 0.1 }, svg);
        }
        const pts = xs.map((v, i) => (s.values[i] === null ? null : `${x(v)},${y(s.values[i])}`)).filter(Boolean);
        el("polyline", { points: pts.join(" "), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": s.dashed ? "5 4" : null }, svg);
        xs.forEach((v, i) => {
          if (s.values[i] === null) return;
          el("circle", { cx: x(v), cy: y(s.values[i]), r: 4, fill: s.color, stroke: "#fff", "stroke-width": 2 }, svg);
        });
        if (opts.endLabels) {
          const last = s.values.length - 1;
          text(svg, s.label, { class: "row-label", x: x(xs[last]) + 10, y: y(s.values[last]) + 4 });
        }
      }

      // crosshair: one hit band per x position
      const cross = el("line", { x1: 0, x2: 0, y1: margin.top, y2: height - margin.bottom, stroke: "#20211f", "stroke-width": 1, opacity: 0 }, svg);
      xs.forEach((v, i) => {
        const left = i === 0 ? xr[0] - 10 : (x(xs[i - 1]) + x(v)) / 2;
        const right = i === xs.length - 1 ? xr[1] + 10 : (x(v) + x(xs[i + 1])) / 2;
        const hit = el("rect", { class: "hit", x: left, y: margin.top, width: right - left, height: height - margin.top - margin.bottom }, svg);
        bindTip(hit, () => ({
          title: `${opts.xName || opts.xLabel || "x"}: ${(opts.xLabels && opts.xLabels[i]) || v}`,
          rows: opts.series
            .filter((s) => s.values[i] !== null)
            .map((s) => ({
              color: s.color,
              value: fmt(s.values[i], digits, unit) + (s.errs ? ` ± ${fmt(s.errs[i], digits)}` : "") + (s.lo ? ` (${fmt(s.lo[i], digits)}–${fmt(s.hi[i], digits)})` : ""),
              name: s.label,
            })),
          foot: opts.foot ? opts.foot(i) : undefined,
        }), {
          onEnter: () => { cross.setAttribute("x1", x(v)); cross.setAttribute("x2", x(v)); cross.setAttribute("opacity", 0.25); },
          onLeave: () => cross.setAttribute("opacity", 0),
        });
      });
    });
  }

  // ---------- scatter with nearest-point hover ----------
  //
  // opts: {points: [{x, y, color, label, detail}], xDomain, yDomain, xLabel, yLabel,
  //        xLog, xUnit, yUnit, annotate(svg, x, y) optional overlay}
  function scatter(host, opts) {
    return responsive(host, (width) => {
      const height = opts.height || Math.min(360, Math.max(240, width * 0.55));
      const margin = { top: 20, right: 16, bottom: 42, left: 48 };
      const svg = svgRoot(host, width, height, opts.label || "Scatter plot");
      const xr = [margin.left, width - margin.right];
      const x = opts.xLog ? log10Scale(opts.xDomain, xr) : linear(opts.xDomain, xr);
      const y = linear(opts.yDomain, [height - margin.bottom, margin.top]);

      for (const t of ticks(opts.yDomain, 4)) {
        el("line", { class: "gridline", x1: xr[0], x2: xr[1], y1: y(t), y2: y(t) }, svg);
        text(svg, `${t}`, { class: "tick", x: margin.left - 8, y: y(t) + 4, "text-anchor": "end" });
      }
      const xt = opts.xTicks || (opts.xLog ? logTicks(opts.xDomain) : ticks(opts.xDomain, 5));
      for (const t of xt) {
        el("line", { class: "gridline", x1: x(t), x2: x(t), y1: margin.top, y2: height - margin.bottom }, svg);
        text(svg, opts.xTickFormat ? opts.xTickFormat(t) : `${t}`, { class: "tick", x: x(t), y: height - margin.bottom + 16, "text-anchor": "middle" });
      }
      if (opts.xLabel) text(svg, opts.xLabel, { class: "axis-label", x: (xr[0] + xr[1]) / 2, y: height - 6, "text-anchor": "middle" });
      if (opts.yLabel) text(svg, opts.yLabel, { class: "axis-label", x: margin.left, y: 10 });
      if (opts.annotate) opts.annotate(svg, x, y, { width, height, margin });

      const dots = opts.points.map((p) => ({
        p,
        cx: x(p.x),
        cy: y(p.y),
        node: el("circle", { cx: x(p.x), cy: y(p.y), r: p.r || 4.5, fill: p.color, stroke: "#fff", "stroke-width": 1.5, opacity: p.opacity ?? 0.9 }, svg),
      }));

      const overlay = el("rect", { class: "hit", x: xr[0], y: margin.top, width: xr[1] - xr[0], height: height - margin.top - margin.bottom }, svg);
      overlay.style.cursor = "crosshair";
      let active = null;
      const ring = el("circle", { r: 8, fill: "none", stroke: "#20211f", "stroke-width": 1.5, opacity: 0, "pointer-events": "none" }, svg);
      overlay.addEventListener("pointermove", (event) => {
        const box = svg.getBoundingClientRect();
        const px = ((event.clientX - box.left) / box.width) * width;
        const py = ((event.clientY - box.top) / box.height) * height;
        let best = null;
        let bestD = Infinity;
        for (const d of dots) {
          const dist = (d.cx - px) ** 2 + (d.cy - py) ** 2;
          if (dist < bestD) { bestD = dist; best = d; }
        }
        if (!best || bestD > 40 ** 2) {
          hideTooltip();
          ring.setAttribute("opacity", 0);
          active = null;
          return;
        }
        active = best;
        ring.setAttribute("cx", best.cx);
        ring.setAttribute("cy", best.cy);
        ring.setAttribute("opacity", 0.7);
        showTooltip(overlay, best.p.detail(), event);
      });
      overlay.addEventListener("pointerleave", () => {
        hideTooltip();
        ring.setAttribute("opacity", 0);
        active = null;
      });
      if (opts.onClick) {
        overlay.addEventListener("click", () => active && opts.onClick(active.p));
      }
    });
  }

  // ---------- horizontal bars, grouped, with an optional reference rule ----------
  //
  // opts: {groups: [{name, rows: [{label, value, color, detail()}]}], domain, unit,
  //        digits, ref: {value, label}, labelWidth}
  function hbars(host, opts) {
    return responsive(host, (width) => {
      const rowH = 28;
      const groupGap = 28;
      const labelW = Math.min(opts.labelWidth || 220, width * 0.5);
      const margin = { top: 14, right: 46, bottom: 26, left: labelW };
      const nRows = opts.groups.reduce((n, g) => n + g.rows.length, 0);
      const named = opts.groups.filter((g) => g.name).length;
      const height = margin.top + margin.bottom + nRows * rowH + named * groupGap;
      const svg = svgRoot(host, width, height, opts.label || "Bar chart");
      const unit = opts.unit ?? "";
      const digits = opts.digits ?? 1;
      const domain = opts.domain || [0, 100];
      const x = linear(domain, [margin.left, width - margin.right]);
      const bottom = height - margin.bottom;

      for (const t of ticks(domain, width < 520 ? 4 : 5)) {
        el("line", { class: "gridline", x1: x(t), x2: x(t), y1: margin.top, y2: bottom }, svg);
        text(svg, `${t}${unit}`, { class: "tick", x: x(t), y: bottom + 16, "text-anchor": "middle" });
      }

      let y = margin.top;
      for (const group of opts.groups) {
        if (group.name) {
          y += groupGap;
          text(svg, group.name, { class: "group-label", x: 0, y: y - 9 });
        }
        for (const row of group.rows) {
          const barH = 14;
          const cy = y + rowH / 2;
          const x0 = x(domain[0]);
          const x1 = x(Math.max(domain[0], row.value));
          const r = Math.min(4, (x1 - x0) / 2);
          const top = cy - barH / 2;
          text(svg, row.label, { class: "row-label", x: labelW - 10, y: cy + 4, "text-anchor": "end" });
          const path = `M${x0},${top} H${x1 - r} Q${x1},${top} ${x1},${top + r} V${top + barH - r} Q${x1},${top + barH} ${x1 - r},${top + barH} H${x0} Z`;
          const bar = el("path", { d: path, fill: row.color }, svg);
          text(svg, fmt(row.value, digits), { class: "value-label", x: x1 + 6, y: cy + 4 });
          const hit = el("rect", { class: "hit", x: 0, y, width, height: rowH }, svg);
          if (row.detail) {
            bindTip(hit, row.detail, {
              onEnter: () => bar.setAttribute("opacity", "0.8"),
              onLeave: () => bar.removeAttribute("opacity"),
            });
          }
          y += rowH;
        }
      }

      if (opts.ref) {
        // drawn behind the bars and labels
        const rx = x(opts.ref.value);
        const rule = el("line", { x1: rx, x2: rx, y1: margin.top + 4, y2: bottom, stroke: "#20211f", "stroke-width": 1, opacity: 0.3, "stroke-dasharray": "3 3" });
        svg.insertBefore(rule, svg.firstChild.nextSibling);
        if (opts.ref.label) text(svg, opts.ref.label, { class: "axis-label", x: rx + 4, y: margin.top + 10 });
      }
    });
  }

  function logTicks([d0, d1]) {
    const out = [];
    for (let e = Math.floor(Math.log10(d0)); e <= Math.ceil(Math.log10(d1)); e++) {
      for (const m of [1, 2, 5]) {
        const v = m * 10 ** e;
        if (v >= d0 * 0.999 && v <= d1 * 1.001) out.push(Number(v.toPrecision(3)));
      }
    }
    return out;
  }

  window.Viz = {
    el, text, linear, ticks, fmt, signed, mix, diverging, DIVERGING, DIVERGING_CAP, rampAt, inkFor, css,
    showTooltip, hideTooltip, bindTip, responsive, svgRoot,
    renderTable, renderLegend,
    dumbbell, heatmap, groupedBars, lineChart, scatter, hbars,
  };
})();
