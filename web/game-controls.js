import { request } from "./api.js";

const modeSelect = document.querySelector("#game-mode");
const modelSelect = document.querySelector("#ai-model");
const colorSelect = document.querySelector("#human-color");
const simulationsInput = document.querySelector("#mcts-simulations");
const refreshButton = document.querySelector("#refresh-models");
const modelHint = document.querySelector("#model-hint");
const aiFields = document.querySelector("#ai-settings");
const modelFields = document.querySelector("#model-fields");
const searchLabel = document.querySelector("#search-label");
const searchHint = document.querySelector("#search-hint");

const PIKAFISH_MAX_DEPTH = 30;

let busy = false;
let refreshing = false;
let modelNames = [];

function updateControls() {
  const aiMode = modeSelect.value === "ai";
  const pikafishMode = modeSelect.value === "pikafish";
  aiFields.hidden = !aiMode && !pikafishMode;
  modelFields.hidden = !aiMode;
  modeSelect.disabled = busy;
  modelSelect.disabled = busy || refreshing || modelNames.length === 0;
  colorSelect.disabled = busy;
  simulationsInput.disabled = busy;
  refreshButton.disabled = busy || refreshing;
  if (pikafishMode) {
    searchLabel.textContent = "PIKAFISH DEPTH";
    searchHint.textContent = "Start with 8. Higher depth is stronger and slower. Moves still go through project Rules.";
    simulationsInput.max = String(PIKAFISH_MAX_DEPTH);
    const depth = Number(simulationsInput.value);
    if (!Number.isInteger(depth) || depth < 1 || depth > PIKAFISH_MAX_DEPTH || depth === 16) {
      simulationsInput.value = "8";
    }
  } else {
    searchLabel.textContent = "MCTS SIMULATIONS";
    searchHint.textContent = "Start with 16. More simulations are slower. CUDA is used when available.";
    simulationsInput.max = "400";
  }
  document.querySelector("#new-game-button").disabled =
    busy || (aiMode && (refreshing || modelNames.length === 0));
}

export function setControlsBusy(value) {
  busy = value;
  updateControls();
}

export function gameSettings() {
  const simulations = Number.parseInt(simulationsInput.value, 10);
  return {
    mode: modeSelect.value || "human",
    model: modelSelect.value || null,
    humanColor: colorSelect.value || "red",
    simulations: Number.isInteger(simulations) ? simulations : 0,
  };
}

async function refreshModels(preferred = modelSelect.value) {
  refreshing = true;
  updateControls();
  try {
    const response = await request("/api/models");
    modelNames = response.models;
    const options = modelNames.map((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      return option;
    });
    modelSelect.replaceChildren(...options);
    if (modelNames.includes(preferred)) modelSelect.value = preferred;
    modelHint.textContent = modelNames.length
      ? "Versioned .pt/.pth model weights from models/. Apply with New game."
      : "No models found. Run scripts/quick_start.py, then refresh this list.";
  } catch (error) {
    modelHint.textContent = error.message;
  } finally {
    refreshing = false;
    updateControls();
  }
}

export async function initializeControls(state) {
  modeSelect.value = state.mode;
  colorSelect.value = state.humanColor;
  simulationsInput.value = state.simulations;
  await refreshModels(state.model || "test_model.pt");
}

modeSelect.addEventListener("change", updateControls);
refreshButton.addEventListener("click", () => refreshModels());
