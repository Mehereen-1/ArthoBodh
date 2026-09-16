const config = window.ARTHOBODH_CONFIG || {};
const examples = {
  "phol-fruit": { sentence: "সে বাজার থেকে তাজা ফল কিনেছে।", target: "ফল" },
  "phol-result": { sentence: "অনেক পরিশ্রমের ফল অবশেষে পেলাম।", target: "ফল" },
  "har-necklace": { sentence: "দিদি গলায় সোনার হার পরেছে।", target: "হার" },
  "har-rate": { sentence: "গ্রামে শিক্ষার হার বেড়েছে।", target: "হার" }
};
const form = document.querySelector("#prediction-form");
const sentenceInput = document.querySelector("#sentence");
const targetInput = document.querySelector("#target-word");
const exampleSelect = document.querySelector("#example-select");
const characterCount = document.querySelector("#character-count");
const formMessage = document.querySelector("#form-message");
const predictButton = document.querySelector("#predict-button");
const emptyResult = document.querySelector("#empty-result");
const resultContent = document.querySelector("#result-content");

sentenceInput.addEventListener("input", () => { characterCount.textContent = `${sentenceInput.value.length} / 500`; clearMessage(); });
targetInput.addEventListener("input", clearMessage);
exampleSelect.addEventListener("change", () => { const example = examples[exampleSelect.value]; if (!example) return; sentenceInput.value = example.sentence; targetInput.value = example.target; sentenceInput.dispatchEvent(new Event("input")); clearMessage(); });

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const sentence = sentenceInput.value.trim();
  const targetWord = targetInput.value.trim();
  const validationError = validateInput(sentence, targetWord);
  if (validationError) { showMessage(validationError, "error"); return; }
  setLoading(true); clearMessage();
  try {
    const result = config.useMock ? await mockPrediction(sentence, targetWord) : await requestPrediction(sentence, targetWord);
    renderResult({ ...result, target_word: result.target_word || targetWord }, sentence, config.useMock);
  } catch (error) { showMessage(error.message || "The prediction could not be completed. Please try again.", "error"); } finally { setLoading(false); }
});

function validateInput(sentence, targetWord) { if (!sentence) return "Please enter a Bengali sentence to analyze."; if (!targetWord) return "Please enter the ambiguous target word."; if (!sentence.includes(targetWord)) return `The target word “${targetWord}” was not found in the sentence.`; return ""; }

async function requestPrediction(sentence, targetWord) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.requestTimeoutMs || 12000);
  try {
    const response = await fetch(config.apiEndpoint || "/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sentence, target_word: targetWord }), signal: controller.signal });
    if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `Backend request failed (${response.status}).`); }
    return await response.json();
  } catch (error) { if (error.name === "AbortError") throw new Error("The request timed out. Please check the backend and try again."); if (error instanceof TypeError) throw new Error("The backend could not be reached. Check the API endpoint and server."); throw error; } finally { clearTimeout(timeout); }
}

async function mockPrediction(sentence, targetWord) {
  await new Promise((resolve) => setTimeout(resolve, 850));
  const isRiver = sentence.includes("নদী") || sentence.includes("তীর");
  const isHole = sentence.includes("সুঁই") || sentence.includes("সুতো");
  const senses = targetWord === "ব্যাংক" && isRiver ? [{ sense: "নদীর তীর", probability: 0.947 }, { sense: "আর্থিক প্রতিষ্ঠান", probability: 0.038 }, { sense: "অন্যান্য", probability: 0.015 }] : targetWord === "চোখ" && isHole ? [{ sense: "ছিদ্র বা ফাঁক", probability: 0.912 }, { sense: "দৃষ্টির অঙ্গ", probability: 0.067 }, { sense: "অন্যান্য", probability: 0.021 }] : targetWord === "চোখ" ? [{ sense: "দৃষ্টির অঙ্গ", probability: 0.947 }, { sense: "ছিদ্র বা ফাঁক", probability: 0.038 }, { sense: "অন্যান্য", probability: 0.015 }] : [{ sense: "আর্থিক প্রতিষ্ঠান", probability: 0.947 }, { sense: "নদীর তীর", probability: 0.038 }, { sense: "অন্যান্য", probability: 0.015 }];
  return { target_word: targetWord, predicted_sense: senses[0].sense, confidence: senses[0].probability, alternative_senses: senses };
}

function renderResult(result, sentence, isMock) {
  const confidence = Math.max(0, Math.min(1, Number(result.confidence) || 0));
  document.querySelector("#result-mode").textContent = isMock ? "MOCK MODE" : "API RESPONSE";
  document.querySelector("#result-target").textContent = result.target_word;
  document.querySelector("#predicted-sense").textContent = result.predicted_sense;
  document.querySelector("#confidence-value").textContent = formatProbability(confidence);
  document.querySelector("#confidence-bar").style.width = `${confidence * 100}%`;
  document.querySelector("#result-sentence").innerHTML = highlightTarget(sentence, result.target_word);
  document.querySelector("#alternatives-list").innerHTML = (result.alternative_senses || []).sort((a, b) => b.probability - a.probability).map((alternative, index) => `<div class="alternative-row ${index === 0 ? "is-predicted" : ""}"><strong lang="bn">${escapeHtml(alternative.sense)}</strong><span class="alternative-meter"><i style="width: ${alternative.probability * 100}%"></i></span><b>${formatProbability(alternative.probability)}</b></div>`).join("");
  emptyResult.hidden = true; resultContent.hidden = false;
}
function highlightTarget(sentence, targetWord) { const safeSentence = escapeHtml(sentence); const safeTarget = escapeHtml(targetWord); return safeSentence.replaceAll(safeTarget, `<mark>${safeTarget}</mark>`); }
function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[character])); }
function formatProbability(value) { return `${(value * 100).toFixed(1)}%`; }
function showMessage(message, type) { formMessage.textContent = message; formMessage.className = `form-message ${type}`; }
function clearMessage() { formMessage.textContent = ""; formMessage.className = "form-message"; }
function setLoading(isLoading) { predictButton.classList.toggle("is-loading", isLoading); predictButton.disabled = isLoading; }