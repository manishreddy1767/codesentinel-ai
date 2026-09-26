/**
 * Controller for pages/analyze-code.html.
 *
 * Wires the language selector, the paste/upload tabs, and the Analyze button
 * to the real backend. Nothing on this page renders a finding that did not
 * come from an API response.
 */

import { analyze, analyzeFile, getSupportedLanguages, getHealth, ApiError } from "./api.js";
import { renderResult, renderStatus } from "./render.js";

const $ = (id) => document.getElementById(id);

const state = {
  mode: "paste", // "paste" | "upload"
  file: null,
  busy: false,
};

function setStatusLine(text, tone = "neutral") {
  const line = $("cs-status-line");
  if (!line) return;

  line.textContent = text;
  line.className =
    "font-body-md " +
    (tone === "error"
      ? "text-error"
      : tone === "success"
        ? "text-primary-container"
        : "text-on-surface-variant");
}

function setBusy(busy) {
  state.busy = busy;
  const button = $("cs-analyze-button");
  if (!button) return;

  button.disabled = busy;
  button.classList.toggle("opacity-50", busy);
  button.classList.toggle("cursor-not-allowed", busy);

  const label = $("cs-analyze-label");
  if (label) label.textContent = busy ? "Analyzing..." : "Analyze Code";
}

/** Populate the language selector from the backend, never from a static list. */
async function loadLanguages() {
  const select = $("cs-language");
  if (!select) return;

  try {
    const data = await getSupportedLanguages();

    select.replaceChildren();

    data.languages.forEach((language) => {
      const option = document.createElement("option");
      option.value = language.id;
      option.textContent = language.parser_available
        ? language.label
        : `${language.label} (parser unavailable)`;
      option.disabled = !language.parser_available;
      select.appendChild(option);
    });

    // Default to C++ when present: the bundled sample snippet is C++.
    const preferred = data.languages.find((l) => l.id === "cpp" && l.parser_available);
    if (preferred) select.value = "cpp";

    setStatusLine("Ready to process source file");
  } catch (error) {
    select.replaceChildren();
    const option = document.createElement("option");
    option.textContent = "Backend unavailable";
    option.value = "";
    select.appendChild(option);
    select.disabled = true;

    setStatusLine(
      error instanceof ApiError ? error.message : "Backend unavailable",
      "error",
    );

    renderStatus($("cs-results"), {
      icon: "cloud_off",
      title: "Backend unavailable",
      message:
        error instanceof ApiError
          ? error.message
          : "Could not load supported languages.",
      tone: "error",
    });
  }
}

function readEditorCode() {
  const editor = $("cs-code-editor");
  if (!editor) return "";

  // innerText preserves the line breaks the highlighted <span>s sit inside;
  // textContent would run the lines together.
  return (editor.innerText || "").replace(/ /g, " ").trimEnd();
}

function switchMode(mode) {
  state.mode = mode;

  $("cs-tab-paste")?.classList.toggle("bg-white/10", mode === "paste");
  $("cs-tab-paste")?.classList.toggle("text-on-surface", mode === "paste");
  $("cs-tab-upload")?.classList.toggle("bg-white/10", mode === "upload");
  $("cs-tab-upload")?.classList.toggle("text-on-surface", mode === "upload");

  const picker = $("cs-file-row");
  if (picker) picker.classList.toggle("hidden", mode !== "upload");

  setStatusLine(
    mode === "upload"
      ? state.file
        ? `Selected ${state.file.name}`
        : "Choose a source file to analyze"
      : "Ready to process source file",
  );
}

async function runAnalysis() {
  if (state.busy) return;

  const results = $("cs-results");
  const language = $("cs-language")?.value || "";

  // ---- validate before spending a request --------------------------------
  if (state.mode === "upload") {
    if (!state.file) {
      renderStatus(results, {
        icon: "upload_file",
        title: "No file selected",
        message: "Choose a source file, then run the analysis.",
        tone: "error",
      });
      return;
    }
  } else {
    const code = readEditorCode();

    if (!code.trim()) {
      renderStatus(results, {
        icon: "edit_note",
        title: "No code to analyze",
        message: "Paste or type some source code into the editor first.",
        tone: "error",
      });
      return;
    }

    if (!language) {
      renderStatus(results, {
        icon: "help",
        title: "No language selected",
        message: "Pick a language the backend supports.",
        tone: "error",
      });
      return;
    }
  }

  setBusy(true);
  setStatusLine("Analyzing...");
  renderStatus(results, {
    icon: "progress_activity",
    title: "Analyzing source code",
    message: "Running parser, rule engine and taint analysis.",
  });

  const started = performance.now();

  try {
    const payload =
      state.mode === "upload"
        ? await analyzeFile({ file: state.file, language: language || null })
        : await analyze({
            code: readEditorCode(),
            language,
            filename: $("cs-filename")?.value || null,
          });

    const elapsed = Math.round(performance.now() - started);

    renderResult(results, payload);

    const count = (payload.vulnerabilities || []).length;
    setStatusLine(
      `Analysis complete — ${count} finding${count === 1 ? "" : "s"} in ${elapsed} ms`,
      count ? "neutral" : "success",
    );

    try {
      sessionStorage.setItem("codesentinel.lastResult", JSON.stringify(payload));
    } catch {
      /* Non-fatal: the details page just will not have a cached result. */
    }
  } catch (error) {
    const isApi = error instanceof ApiError;

    renderStatus(results, {
      icon:
        isApi && error.kind === "offline"
          ? "cloud_off"
          : isApi && error.kind === "timeout"
            ? "timer_off"
            : "error",
      title:
        isApi && error.kind === "offline"
          ? "Backend unavailable"
          : isApi && error.kind === "timeout"
            ? "Analysis timed out"
            : "Analysis failed",
      message: isApi ? error.message : String(error),
      tone: "error",
    });

    setStatusLine("Analysis failed", "error");
  } finally {
    setBusy(false);
  }
}

function init() {
  const results = $("cs-results");

  if (results) {
    renderStatus(results, {
      icon: "policy",
      title: "No analysis run yet",
      message:
        "Paste code or upload a file, then press Analyze Code. " +
        "Results shown here always come from the backend.",
    });
  }

  $("cs-analyze-button")?.addEventListener("click", runAnalysis);
  $("cs-tab-paste")?.addEventListener("click", () => switchMode("paste"));
  $("cs-tab-upload")?.addEventListener("click", () => switchMode("upload"));

  $("cs-file-input")?.addEventListener("change", (event) => {
    state.file = event.target.files?.[0] || null;
    setStatusLine(state.file ? `Selected ${state.file.name}` : "No file selected");
  });

  loadLanguages();

  // Surface backend availability immediately rather than at first click.
  getHealth().catch(() => {
    setStatusLine("Backend unavailable — start the API on port 8000", "error");
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
