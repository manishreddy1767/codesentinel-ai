/**
 * Controller for pages/analysis-results.html.
 *
 * Renders the most recent analysis from this browser session. It deliberately
 * does NOT re-run analysis: the results page should show what was actually
 * produced, not quietly generate something new.
 *
 * If there is no cached result, it says so. The previous version of this page
 * shipped a hardcoded "Buffer Overflow / CWE-120 / 78% security score" report
 * attributed to a "CodeSentinel R-GCN" model that does not exist in this
 * repository. That has been removed.
 */

import { renderResult, renderStatus } from "./render.js";

function init() {
  const container = document.getElementById("cs-results");
  if (!container) return;

  let cached = null;

  try {
    cached = sessionStorage.getItem("codesentinel.lastResult");
  } catch {
    /* Private mode: treated the same as "no result". */
  }

  if (!cached) {
    renderStatus(container, {
      icon: "science",
      title: "No analysis in this session",
      message:
        "Run an analysis on the Analyze Code page and the result will appear here. " +
        "This page never shows sample or placeholder findings.",
    });
    return;
  }

  try {
    renderResult(container, JSON.parse(cached));
  } catch {
    renderStatus(container, {
      icon: "error",
      title: "Stored result could not be read",
      message: "The cached analysis was malformed. Run the analysis again.",
      tone: "error",
    });
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
