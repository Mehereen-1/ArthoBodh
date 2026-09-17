const config = window.ARTHOBODH_CONFIG || {};
const api = (path) => `${config.apiBase || ""}${path}`;

const form = document.querySelector("#prediction-form");
const sentenceInput = document.querySelector("#sentence");
const targetInput = document.querySelector("#target-word");
const exampleSelect = document.querySelector("#example-select");
const characterCount = document.querySelector("#character-count");
const formMessage = document.querySelector("#form-message");
const predictButton = document.querySelector("#predict-button");
const emptyResult = document.querySelector("#empty-result");
const resultContent = document.querySelector("#result-content");
let examples = [];

sentenceInput.addEventListener("input", () => { characterCount.textContent = `${sentenceInput.value.length} / 500`; clearMessage(); });
targetInput.addEventListener("input", clearMessage);
exampleSelect.addEventListener("change", () => {
  const example = examples[Number(exampleSelect.value)];
  if (!example) return;
  sentenceInput.value = example.sentence.slice(0, 500);
  targetInput.value = example.target_word;
  sentenceInput.dispatchEvent(new Event("input"));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const sentence = sentenceInput.value.trim();
  const targetWord = targetInput.value.trim();
  const validationError = validateInput(sentence, targetWord);
  if (validationError) { showMessage(validationError, "error"); return; }
  setLoading(true); clearMessage();
  try {
    const result = await requestPrediction(sentence, targetWord);
    renderResult(result, sentence);
  } catch (error) { showMessage(error.message || "The prediction could not be completed. Please try again.", "error"); } finally { setLoading(false); }
});

loadMetadata();
loadExamples();

function validateInput(sentence, targetWord) { if (!sentence) return "Please enter a Bengali sentence to analyze."; if (!targetWord) return "Please enter the ambiguous target word."; if (!sentence.includes(targetWord)) return `The target word “${targetWord}” was not found in the sentence.`; return ""; }

async function requestPrediction(sentence, targetWord) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.requestTimeoutMs || 30000);
  try {
    const response = await fetch(api("/predict"), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sentence, target_word: targetWord }), signal: controller.signal });
    if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `Backend request failed (${response.status}).`); }
    return await response.json();
  } catch (error) { if (error.name === "AbortError") throw new Error("The request timed out. Please check the backend and try again."); if (error instanceof TypeError) throw new Error("The backend could not be reached. Is the server running?"); throw error; } finally { clearTimeout(timeout); }
}

async function loadMetadata() {
  try {
    const meta = await (await fetch(api("/metadata"))).json();
    document.querySelector("#supported-words").innerHTML = meta.supported_words.map((w) => `<option value="${escapeHtml(w)}"></option>`).join("");
    document.querySelector("#meta-model").textContent = meta.model_ready ? meta.model : "Not trained yet";
    document.querySelector("#meta-accuracy").textContent = meta.test_metrics ? formatProbability(meta.test_metrics.accuracy) : "Run src.evaluate";
    document.querySelector("#meta-f1").textContent = meta.test_metrics ? formatProbability(meta.test_metrics.macro_f1) : "Run src.evaluate";
    if (!meta.model_ready) showMessage(meta.model_error, "error");
  } catch { document.querySelector("#meta-model").textContent = "Backend unreachable"; }
}

async function loadExamples() {
  try {
    examples = await (await fetch(api("/examples"))).json();
    exampleSelect.innerHTML = `<option value="">Try an example</option>` + examples.map((e, i) => `<option value="${i}">${escapeHtml(e.target_word)} — ${escapeHtml(e.sentence.slice(0, 50))}…</option>`).join("");
  } catch { /* examples are optional */ }
}

function renderResult(result, sentence) {
  const confidence = Math.max(0, Math.min(1, Number(result.confidence) || 0));
  document.querySelector("#result-target").textContent = result.target_word;
  document.querySelector("#predicted-sense").textContent = result.predicted_sense;
  document.querySelector("#confidence-value").textContent = formatProbability(confidence);
  document.querySelector("#confidence-bar").style.width = `${confidence * 100}%`;
  document.querySelector("#result-sentence").innerHTML = highlightTarget(sentence, result.target_word);
  document.querySelector("#alternatives-list").innerHTML = (result.alternative_senses || []).map((alternative, index) => `<div class="alternative-row ${index === 0 ? "is-predicted" : ""}"><strong lang="bn">${escapeHtml(alternative.sense)}</strong><span class="alternative-meter"><i style="width: ${alternative.probability * 100}%"></i></span><b>${formatProbability(alternative.probability)}</b></div>`).join("");
  emptyResult.hidden = true; resultContent.hidden = false;
}
function highlightTarget(sentence, targetWord) { const safeSentence = escapeHtml(sentence); const safeTarget = escapeHtml(targetWord); return safeSentence.replaceAll(safeTarget, `<mark>${safeTarget}</mark>`); }
function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[character])); }
function formatProbability(value) { return `${(value * 100).toFixed(1)}%`; }
function showMessage(message, type) { formMessage.textContent = message; formMessage.className = `form-message ${type}`; }
function clearMessage() { formMessage.textContent = ""; formMessage.className = "form-message"; }
function setLoading(isLoading) { predictButton.classList.toggle("is-loading", isLoading); predictButton.disabled = isLoading; }
