const boardElement = document.querySelector("#board");
const boardFrame = document.querySelector(".board-frame");
const turnLabel = document.querySelector("#turn-label");
const statusMessage = document.querySelector("#status-message");
const moveCount = document.querySelector("#move-count");
const lastMove = document.querySelector("#last-move");
const undoButton = document.querySelector("#undo-button");
const newGameButton = document.querySelector("#new-game-button");
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

function positionKey(row, col) {
  return `${row}:${col}`;
}

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
  for (let row = 0; row < BOARD_ROWS; row += 1) {
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

  return state.isRedTurn ? "Red to move" : "Black to move";
}

function updatePanel() {
  const text = statusText(gameState);
  turnLabel.textContent = text;
  statusMessage.textContent = gameState.isTerminal
    ? gameState.terminationReason
    : text;
  moveCount.textContent = String(gameState.moveCount);
  lastMove.textContent = moveNotation(gameState.lastMove);
  selectedCoord.textContent = selectedOrigin
    ? formatCoord(selectedOrigin.row, selectedOrigin.col)
    : "-";
  undoButton.disabled = gameState.moveCount === 0;
}

function renderBoard() {
  const legalOrigins = new Set(
    gameState.legalMoves.map(([row, col]) => positionKey(row, col)),
  );
  const destinations = new Set(
    selectedMoves().map(([, , row, col]) => positionKey(row, col)),
  );
  const lastMoveSquares = new Set(
    gameState.lastMove
      ? [
          positionKey(gameState.lastMove[0], gameState.lastMove[1]),
          positionKey(gameState.lastMove[2], gameState.lastMove[3]),
        ]
      : [],
  );

  boardElement.replaceChildren();
  for (let displayRow = 9; displayRow >= 0; displayRow -= 1) {
    for (let col = 0; col < 9; col += 1) {
      const cell = document.createElement("button");
      const key = positionKey(displayRow, col);
      const piece = gameState.board[displayRow][col];
      cell.type = "button";
      cell.className = "board-cell";
      cell.dataset.row = String(displayRow);
      cell.dataset.col = String(col);
      cell.title = `r=${displayRow}, c=${col}`;
      cell.setAttribute("aria-label", `${String.fromCharCode(97 + col)}${displayRow}`);

      if (selectedOrigin && key === positionKey(selectedOrigin.row, selectedOrigin.col)) {
        cell.classList.add("selected");
      }
      if (destinations.has(key)) {
        cell.classList.add("destination");
      }
      if (lastMoveSquares.has(key)) {
        cell.classList.add("last-move");
      }
      if (piece) {
        const token = document.createElement("span");
        token.className = `piece ${piece.isRed ? "red" : "black"}`;
        token.textContent = piece.symbol;
        cell.append(token);
      }

      const coordTag = document.createElement("span");
      coordTag.className = "cell-coord";
      coordTag.textContent = `${displayRow},${col}`;
      coordTag.setAttribute("aria-hidden", "true");
      cell.append(coordTag);

      cell.disabled = gameState.isTerminal || (
        !legalOrigins.has(key) && !destinations.has(key)
      );
      cell.addEventListener("click", () => selectSquare(displayRow, col));
      boardElement.append(cell);
    }
  }
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "The request could not be completed.");
  }
  return payload;
}

async function loadState() {
  gameState = await request("/api/state");
  selectedOrigin = null;
  updatePanel();
  renderBoard();
}

async function submitMove(move) {
  const response = await request("/api/move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      fromRow: move[0],
      fromCol: move[1],
      toRow: move[2],
      toCol: move[3],
    }),
  });
  gameState = response.state;
  selectedOrigin = null;
  updatePanel();
  renderBoard();
}

async function selectSquare(row, col) {
  if (gameState.isTerminal) {
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
  renderBoard();
}

undoButton.addEventListener("click", async () => {
  try {
    const response = await request("/api/undo", { method: "POST" });
    gameState = response.state;
    selectedOrigin = null;
    updatePanel();
    renderBoard();
  } catch (error) {
    statusMessage.textContent = error.message;
  }
});

newGameButton.addEventListener("click", async () => {
  const response = await request("/api/new-game", { method: "POST" });
  gameState = response.state;
  selectedOrigin = null;
  updatePanel();
  renderBoard();
});

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
