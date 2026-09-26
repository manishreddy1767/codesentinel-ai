/**
 * CodeSentinel-AI — backend API client.
 *
 * Every network call goes through here so that timeout handling, error
 * shaping and the base-URL setting exist in exactly one place.
 *
 * Error contract: every method either resolves with parsed JSON, or rejects
 * with an ApiError carrying a `kind` the UI can branch on. The UI must never
 * have to parse an error message string to decide what to render.
 */

const DEFAULT_BASE_URL = "http://127.0.0.1:8000";
const DEFAULT_TIMEOUT_MS = 60000;

/** Error with a machine-readable `kind`, so callers never regex a message. */
export class ApiError extends Error {
  constructor(kind, message, { status = null, detail = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.kind = kind; // "offline" | "timeout" | "http" | "malformed"
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Base URL, overridable from the settings page and remembered per browser.
 * Reads are wrapped because localStorage throws in some privacy modes.
 */
export function getBaseUrl() {
  try {
    return localStorage.getItem("codesentinel.apiBaseUrl") || DEFAULT_BASE_URL;
  } catch {
    return DEFAULT_BASE_URL;
  }
}

export function setBaseUrl(url) {
  try {
    localStorage.setItem("codesentinel.apiBaseUrl", url);
  } catch {
    /* Non-fatal: the setting simply will not persist. */
  }
}

async function request(path, { method = "GET", body, headers, timeoutMs } = {}) {
  const controller = new AbortController();
  const limit = timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const timer = setTimeout(() => controller.abort(), limit);

  let response;

  try {
    response = await fetch(`${getBaseUrl()}${path}`, {
      method,
      body,
      headers,
      signal: controller.signal,
    });
  } catch (error) {
    clearTimeout(timer);

    // An aborted fetch and an unreachable server both land here, but they
    // need different messages: one is "wait less", the other is "start the
    // backend".
    if (error.name === "AbortError") {
      throw new ApiError(
        "timeout",
        `The backend did not respond within ${Math.round(limit / 1000)}s. ` +
          `Large files can exceed this.`,
      );
    }

    throw new ApiError(
      "offline",
      `Cannot reach the backend at ${getBaseUrl()}. Start it with: ` +
        `uvicorn app.main:app --reload --port 8000 (from the backend/ directory).`,
    );
  } finally {
    clearTimeout(timer);
  }

  let payload = null;

  try {
    payload = await response.json();
  } catch {
    if (response.ok) {
      throw new ApiError("malformed", "The backend returned a non-JSON response.");
    }
  }

  if (!response.ok) {
    // FastAPI puts the useful text in `detail`, which is either a string or a
    // list of validation objects.
    const detail = payload && payload.detail;
    let message;

    if (typeof detail === "string") {
      message = detail;
    } else if (Array.isArray(detail)) {
      message = detail
        .map((d) => d.msg || JSON.stringify(d))
        .join("; ");
    } else {
      message = `Request failed with HTTP ${response.status}.`;
    }

    throw new ApiError("http", message, { status: response.status, detail });
  }

  return payload;
}

export function getHealth() {
  return request("/health", { timeoutMs: 5000 });
}

export function getSupportedLanguages() {
  return request("/supported-languages", { timeoutMs: 10000 });
}

export function analyze({ code, language, filename = null }) {
  return request("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code, language, filename }),
  });
}

export function analyzeFile({ file, language = null }) {
  const form = new FormData();
  form.append("file", file);

  if (language) {
    form.append("language", language);
  }

  // No Content-Type header: the browser must set the multipart boundary.
  return request("/analyze/file", { method: "POST", body: form });
}
