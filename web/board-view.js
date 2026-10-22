const boardElement = document.querySelector("#board");

export function positionKey(row, col) {
  return `${row}:${col}`;
}

function coordinateTag(row, col) {
  const tag = document.createElement("span");
  tag.className = "cell-coord";
  tag.textContent = `${row},${col}`;
  tag.setAttribute("aria-hidden", "true");
  return tag;
}

function createCell(row, col, context) {
  const { state, selectedOrigin, destinations, lastSquares, origins, disabled, onSelect } = context;
  const key = positionKey(row, col);
  const cell = document.createElement("button");
  cell.type = "button";
  cell.className = "board-cell";
  cell.dataset.row = String(row);
  cell.dataset.col = String(col);
  cell.title = `r=${row}, c=${col}`;
  cell.setAttribute("aria-label", `${String.fromCharCode(97 + col)}${row}`);
  if (selectedOrigin && key === positionKey(selectedOrigin.row, selectedOrigin.col)) {
    cell.classList.add("selected");
  }
  if (destinations.has(key)) cell.classList.add("destination");
  if (lastSquares.has(key)) cell.classList.add("last-move");
  const piece = state.board[row][col];
  if (piece) {
    const token = document.createElement("span");
    token.className = `piece ${piece.isRed ? "red" : "black"}`;
    token.textContent = piece.symbol;
    cell.append(token);
  }
  cell.append(coordinateTag(row, col));
  cell.disabled = disabled || (!origins.has(key) && !destinations.has(key));
  cell.addEventListener("click", () => onSelect(row, col));
  return cell;
}

export function drawBoard(state, selectedOrigin, selectedMoves, busy, onSelect) {
  const context = {
    state, selectedOrigin, onSelect,
    destinations: new Set(selectedMoves.map(([, , row, col]) => positionKey(row, col))),
    lastSquares: new Set(state.lastMove ? [
      positionKey(state.lastMove[0], state.lastMove[1]),
      positionKey(state.lastMove[2], state.lastMove[3]),
    ] : []),
    origins: new Set(state.legalMoves.map(([row, col]) => positionKey(row, col))),
    disabled: busy || state.isAiTurn || state.isTerminal,
  };
  boardElement.replaceChildren();
  for (let row = 9; row >= 0; row -= 1) {
    for (let col = 0; col < 9; col += 1) {
      boardElement.append(createCell(row, col, context));
    }
  }
}
