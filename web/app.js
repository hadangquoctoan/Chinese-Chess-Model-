import { post, request } from "./api.js";
import { drawBoard, positionKey } from "./board-view.js";
import { gameSettings, initializeControls, setControlsBusy } from "./game-controls.js";

const boardFrame = document.querySelector(".board-frame");
const turnLabel = document.querySelector("#turn-label");
const statusMessage = document.querySelector("#status-message");
const moveCount = document.querySelector("#move-count");
const lastMove = document.querySelector("#last-move");
const undoButton = document.querySelector("#undo-button");
const newGameButton = document.querySelector("#new-game-button");
const retryAiButton = document.querySelector("#retry-ai-button");
const opponentLabel = document.querySelector("#opponent-label");
const themeToggle = document.querySelector("#theme-toggle");
const selectedCoord = document.querySelector("#selected-coord");

const THEME_STORAGE_KEY = "xiangqi-theme";
const BOARD_ROWS = 10;
const BOARD_COLS = 9;

function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "dark"
    ? "dark"
    : "light";
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch (err) {
    // localStorage may be unavailable in private mode; the UI still works.
  }
}

let gameState = null;
let selectedOrigin = null;
let busy = true;
let pendingMessage = "Loading game...";

function formatCoord(row, col) {
  return `(${row},${col})`;
}

function moveNotation(move) {
  if (!move) {
    return "-";
  }

  const [fromRow, fromCol, toRow, toCol] = move;
  return `${formatCoord(fromRow, fromCol)} -> ${formatCoord(toRow, toCol)}`;
}

function buildAxisLabels() {
  const colTop = boardFrame.querySelector(".board-axis-top");
  const colBottom = boardFrame.querySelector(".board-axis-bottom");
  const rowLeft = boardFrame.querySelector(".board-axis-left");
  const rowRight = boardFrame.querySelector(".board-axis-right");
  if (!colTop || !colBottom || !rowLeft || !rowRight) {
    return;
  }

  colTop.replaceChildren();
  colBottom.replaceChildren();
  rowLeft.replaceChildren();
  rowRight.replaceChildren();

  // Column labels (0-8). Top and bottom match so users can read either side.
  for (let col = 0; col < BOARD_COLS; col += 1) {
    for (const target of [colTop, colBottom]) {
      const label = document.createElement("span");
      label.className = "axis-label axis-col";
      label.textContent = String(col);
      target.append(label);
    }
  }

  // Row labels (0-9). Red sits at row 0, black at row 9.
  for (let row = BOARD_ROWS - 1; row >= 0; row -= 1) {
    for (const target of [rowLeft, rowRight]) {
      const label = document.createElement("span");
      label.className = "axis-label axis-row";
      label.textContent = String(row);
      target.append(label);
    }
  }
}

function selectedMoves() {
  if (!selectedOrigin || !gameState) {
    return [];
  }

  return gameState.legalMoves.filter(
    ([fromRow, fromCol]) =>
      fromRow === selectedOrigin.row && fromCol === selectedOrigin.col,
  );
}

function statusText(state) {
  if (state.isTerminal) {
    if (state.winner === "red") {
      return "Red wins";
    }
    if (state.winner === "black") {
      return "Black wins";
    }
    return "Draw";
  }

  if (state.isAiTurn) return "AI to move";
  return state.isRedTurn ? "Red to move" : "Black to move";
}

function updatePanel() {
  const text = statusText(gameState);
  turnLabel.textContent = text;
  statusMessage.textContent = busy ? pendingMessage : (
    gameState.isTerminal ? gameState.terminationReason : text
  );
  if (gameState.mode === "pikafish") {
    opponentLabel.textContent =
      `Pikafish | depth ${gameState.simulations} | You: ${gameState.humanColor}`;
  } else if (gameState.mode === "ai") {
    opponentLabel.textContent =
      `${gameState.model} | ${gameState.device} | MCTS ${gameState.simulations} | You: ${gameState.humanColor}`;
  } else {
    opponentLabel.textContent = "Two human players";
  }
  retryAiButton.hidden = !gameState.isAiTurn || busy;
  moveCount.textContent = String(gameState.moveCount);
  lastMove.textContent = moveNotation(gameState.lastMove);
  selectedCoord.textContent = selectedOrigin
    ? formatCoord(selectedOrigin.row, selectedOrigin.col)
    : "-";
  undoButton.disabled = busy || !gameState.canUndo;
}

function renderBoard() {
  drawBoard(gameState, selectedOrigin, selectedMoves(), busy, selectSquare);
}

function renderState(state = gameState) {
  gameState = state;
  selectedOrigin = null;
  updatePanel();
  renderBoard();
}

async function playAiTurn() {
  if (!gameState.isAiTurn) return;
  pendingMessage = gameState.mode === "pikafish"
    ? "Pikafish is thinking..."
    : "AI is thinking (NN + MCTS)...";
  updatePanel();
  const response = await post("/api/ai-move");
  renderState(response.state);
}

async function runAction(action, message) {
  if (busy) return;
  busy = true;
  pendingMessage = message;
  setControlsBusy(true);
  renderState();
  let failure = null;
  try {
    await action();
    await playAiTurn();
  } catch (error) {
    failure = error.message;
    if (error.state) renderState(error.state);
  } finally {
    busy = false;
    setControlsBusy(false);
    renderState();
    if (failure) statusMessage.textContent = failure;
  }
}

async function loadState() {
  setControlsBusy(true);
  renderState(await request("/api/state"));
  await initializeControls(gameState);
  busy = false;
  setControlsBusy(false);
  renderState();
  await runAction(async () => {}, "Restoring game...");
}

async function submitMove(move) {
  await runAction(async () => {
    const response = await post("/api/move", {
      fromRow: move[0], fromCol: move[1], toRow: move[2], toCol: move[3],
    });
    renderState(response.state);
  }, "Sending move...");
}

async function selectSquare(row, col) {
  if (busy || !gameState || gameState.isAiTurn || gameState.isTerminal) {
    return;
  }

  const matchingMove = selectedMoves().find(
    ([, , toRow, toCol]) => toRow === row && toCol === col,
  );
  if (matchingMove) {
    try {
      await submitMove(matchingMove);
    } catch (error) {
      statusMessage.textContent = error.message;
    }
    return;
  }

  const key = positionKey(row, col);
  const canSelect = gameState.legalMoves.some(
    ([fromRow, fromCol]) => positionKey(fromRow, fromCol) === key,
  );
  selectedOrigin = canSelect ? { row, col } : null;
  updatePanel();
  renderBoard();
}

undoButton.addEventListener("click", () => runAction(async () => {
  const response = await post("/api/undo");
  renderState(response.state);
}, "Undoing turn..."));

newGameButton.addEventListener("click", () => runAction(async () => {
  const response = await post("/api/new-game", gameSettings());
  renderState(response.state);
}, "Starting game / loading model..."));

retryAiButton.addEventListener("click", () => runAction(async () => {}, "Retrying AI move..."));

themeToggle.addEventListener("click", () => {
  setTheme(currentTheme() === "dark" ? "light" : "dark");
});

if (window.matchMedia) {
  window
    .matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", (event) => {
      let stored = null;
      try {
        stored = localStorage.getItem(THEME_STORAGE_KEY);
      } catch (err) {
        stored = null;
      }
      if (stored) {
        return;
      }
      setTheme(event.matches ? "dark" : "light");
    });
}

loadState().catch((error) => {
  turnLabel.textContent = "Unable to load game";
  statusMessage.textContent = error.message;
});

buildAxisLabels();
