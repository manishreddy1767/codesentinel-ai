/**
 * Rendering helpers for analysis results.
 *
 * Every value that originates from source code or from the backend is written
 * with `textContent`, never `innerHTML`. Analyzed code is attacker-controlled
 * by definition - a tool that renders `<img onerror=...>` from the snippet it
 * is analyzing would be self-XSSing its own operator. The only HTML this file
 * creates is elements it constructs itself.
 */

const SEVERITY_STYLE = {
  CRITICAL: { badge: "bg-error/20 text-error border-error/40", order: 0 },
  HIGH: { badge: "bg-error/15 text-error/90 border-error/30", order: 1 },
  MEDIUM: { badge: "bg-[#ffb59a]/15 text-[#ffb59a] border-[#ffb59a]/30", order: 2 },
  LOW: { badge: "bg-primary-container/15 text-primary-container border-primary-container/30", order: 3 },
  INFO: { badge: "bg-white/10 text-on-surface-variant border-white/20", order: 4 },
};

function styleFor(severity) {
  return SEVERITY_STYLE[String(severity || "").toUpperCase()] || SEVERITY_STYLE.INFO;
}

/** Create an element with classes and text. Text is always set safely. */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

/** A centred status block: loading, error, empty, or idle. */
export function renderStatus(container, { icon, title, message, tone = "neutral" }) {
  clear(container);

  const toneClass =
    tone === "error"
      ? "text-error"
      : tone === "success"
        ? "text-primary-container"
        : "text-on-surface-variant";

  const wrap = el("div", "flex flex-col items-center justify-center py-16 px-6 text-center gap-3");

  if (icon) {
    const i = el("span", `material-symbols-outlined text-[40px] ${toneClass}`, icon);
    if (icon === "progress_activity") i.classList.add("animate-spin");
    wrap.appendChild(i);
  }

  wrap.appendChild(el("p", `font-headline-md text-[16px] font-bold ${toneClass}`, title));

  if (message) {
    wrap.appendChild(
      el("p", "font-body-md text-[13px] text-on-surface-variant max-w-md leading-relaxed", message),
    );
  }

  container.appendChild(wrap);
}

/** Summary strip: score, risk level, severity counts. */
function renderSummary(result) {
  const risk = result.security_risk || {};
  const counts = risk.severity_counts || {};

  const bar = el("div", "flex flex-wrap items-center gap-3 p-4 rounded-xl bg-white/5 border border-white/10 mb-4");

  const score = el("div", "flex items-baseline gap-2");
  score.appendChild(el("span", "font-headline-md text-[28px] font-bold text-on-surface", risk.security_score ?? "—"));
  score.appendChild(el("span", "font-body-md text-[12px] text-on-surface-variant", "/ 100 security score"));
  bar.appendChild(score);

  const level = String(risk.risk_level || "UNKNOWN").toUpperCase();
  bar.appendChild(
    el("span", `px-3 py-1 rounded-full border text-[12px] font-bold ${styleFor(level).badge}`, level),
  );

  const total = (result.vulnerabilities || []).length;
  bar.appendChild(
    el("span", "font-body-md text-[13px] text-on-surface-variant",
      `${total} finding${total === 1 ? "" : "s"}`),
  );

  Object.entries(counts)
    .filter(([, n]) => n > 0)
    .forEach(([severity, n]) => {
      bar.appendChild(
        el("span", `px-2 py-0.5 rounded border text-[11px] font-medium ${styleFor(severity).badge}`,
          `${severity} ${n}`),
      );
    });

  const meta = el("div", "ml-auto font-code-md text-[11px] text-on-surface-variant/70");
  meta.textContent = `${result.language || "?"} · id ${String(result.analysis_id || "").slice(0, 8)}`;
  bar.appendChild(meta);

  return bar;
}

/** One finding card. */
function renderFinding(vulnerability, index) {
  const severity = String(vulnerability.severity || "INFO").toUpperCase();
  const card = el("div", "rounded-xl bg-white/5 border border-white/10 p-4 mb-3");

  const head = el("div", "flex items-start gap-3 mb-2");
  head.appendChild(
    el("span", `px-2 py-0.5 rounded border text-[11px] font-bold shrink-0 ${styleFor(severity).badge}`, severity),
  );

  const titleText =
    vulnerability.title || vulnerability.type || `Finding ${index + 1}`;
  head.appendChild(el("h4", "font-headline-md text-[15px] font-bold text-on-surface flex-1", titleText));

  if (typeof vulnerability.confidence === "number") {
    head.appendChild(
      el("span", "font-code-md text-[11px] text-on-surface-variant shrink-0",
        `confidence ${vulnerability.confidence.toFixed(2)}`),
    );
  }

  card.appendChild(head);

  // Location / CWE / detector chips
  const chips = el("div", "flex flex-wrap gap-2 mb-2");

  if (vulnerability.line !== null && vulnerability.line !== undefined) {
    chips.appendChild(el("span", "font-code-md text-[11px] text-on-surface-variant", `line ${vulnerability.line}`));
  }
  if (vulnerability.function) {
    chips.appendChild(el("span", "font-code-md text-[11px] text-on-surface-variant", `fn ${vulnerability.function}`));
  }
  if (vulnerability.cwe) {
    chips.appendChild(el("span", "font-code-md text-[11px] text-[#ffb59a]", vulnerability.cwe));
  }
  if (vulnerability.type) {
    chips.appendChild(el("span", "font-code-md text-[11px] text-primary-container/80", vulnerability.type));
  }
  if (vulnerability.sink) {
    chips.appendChild(el("span", "font-code-md text-[11px] text-on-surface-variant", `sink: ${vulnerability.sink}`));
  }

  if (chips.childNodes.length) card.appendChild(chips);

  if (vulnerability.description) {
    card.appendChild(
      el("p", "font-body-md text-[13px] text-on-surface-variant leading-relaxed mb-2",
        vulnerability.description),
    );
  }

  // Offending code: textContent only.
  if (vulnerability.code) {
    const pre = el("pre", "font-code-md text-[12px] bg-black/40 border border-white/10 rounded p-2 overflow-x-auto mb-2");
    pre.appendChild(el("code", "text-[#e5e2e1]", vulnerability.code));
    card.appendChild(pre);
  }

  if (vulnerability.recommendation) {
    const rec = el("div", "flex gap-2 items-start");
    rec.appendChild(el("span", "material-symbols-outlined text-[16px] text-primary-container shrink-0", "lightbulb"));
    rec.appendChild(
      el("p", "font-body-md text-[12px] text-on-surface-variant leading-relaxed",
        vulnerability.recommendation),
    );
    card.appendChild(rec);
  }

  const nodes = vulnerability.cpg_nodes || [];
  if (nodes.length) {
    card.appendChild(
      el("p", "font-code-md text-[11px] text-on-surface-variant/60 mt-2",
        `linked to ${nodes.length} code-property-graph node${nodes.length === 1 ? "" : "s"}`),
    );
  }

  return card;
}

/** Full result view, or an explicit "no findings" state. */
export function renderResult(container, result) {
  clear(container);
  container.appendChild(renderSummary(result));

  const findings = (result.vulnerabilities || [])
    .slice()
    .sort((a, b) => styleFor(a.severity).order - styleFor(b.severity).order);

  if (!findings.length) {
    const wrap = el("div", "flex flex-col items-center justify-center py-12 px-6 text-center gap-2");
    wrap.appendChild(el("span", "material-symbols-outlined text-[40px] text-primary-container", "verified"));
    wrap.appendChild(el("p", "font-headline-md text-[16px] font-bold text-on-surface", "No vulnerabilities detected"));
    wrap.appendChild(
      el("p", "font-body-md text-[13px] text-on-surface-variant max-w-md leading-relaxed",
        "The rule and taint analyzers found nothing in this snippet. " +
        "This is not a guarantee that the code is secure - it means these " +
        "detectors did not fire."),
    );
    container.appendChild(wrap);
    return;
  }

  findings.forEach((v, i) => container.appendChild(renderFinding(v, i)));
}
