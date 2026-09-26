/**
 * Controller for pages/vulnerability-details.html.
 *
 * Shows the findings from this session's most recent analysis. It previously
 * rendered a hardcoded CWE-120 buffer-overflow write-up, including a fabricated
 * code snippet and explanation, for code the user never submitted.
 */

import { renderResult, renderStatus } from "./render.js";

function init() {
  const container = document.getElementById("cs-results");
  if (!container) return;

  let cached = null;
  try {
    cached = sessionStorage.getItem("codesentinel.lastResult");
  } catch {
    /* Private mode: same as no result. */
  }

  if (!cached) {
    renderStatus(container, {
      icon: "bug_report",
      title: "No finding selected",
      message:
        "Run an analysis on the Analyze Code page first. This page only ever " +
        "shows findings returned by the backend.",
    });
    return;
  }

  try {
    renderResult(container, JSON.parse(cached));
  } catch {
    renderStatus(container, {
      icon: "error",
      title: "Stored result could not be read",
      message: "Run the analysis again.",
      tone: "error",
    });
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
