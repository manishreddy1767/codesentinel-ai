/**
 * Repository analysis — honest empty state.
 *
 * This page previously rendered fabricated repository scan progress and findings. There is no
 * repository ingestion endpoint in the backend, so there is nothing real to show. Rather than
 * display invented data, it states the limitation.
 */

import { renderStatus } from "./render.js";

function init() {
  const container = document.getElementById("cs-results");
  if (!container) return;

  renderStatus(container, {
    icon: "folder_off",
    title: "Repository analysis is not implemented",
    message:
      "Scanning a whole repository requires safe traversal, per-file limits and a POST /analyze/repository endpoint. It is deliberately not implemented rather than faked. Single files work on the Analyze Code page.",
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
