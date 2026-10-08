"use strict";

const BOARD_ROWS = 10;
const BOARD_COLS = 9;
const TOTAL_BOARD_CELLS = BOARD_ROWS * BOARD_COLS;
const TOTAL_PIECES = 32;
const PLAY_INTERVAL_MS = 1900;
const TAKEAWAY = "Sau Input Conv và hai ResidualBlock, nhánh tích chập dài nhất có trường cảm thụ lý thuyết 11×11. Với một nơ-ron ở vùng tâm, footprint này đã bao trùm toàn bộ 90 ô của bàn cờ 10×9.";

const STAGES = [
  {
    shortName: "Tầng 0",
    title: "Bàn cờ gốc · Input Tensor",
    code: "tensor (19, 10, 9)",
    convCount: 0,
    rf: 1,
    color: "#cde2fb",
    summary: "Mỗi vị trí chỉ chứa vector 19 kênh đặc trưng của chính ô đó. Chưa có phép tích chập nào truyền thông tin từ các ô lân cận.",
    insight: "RF 1×1 là điểm xuất phát toán học, đại diện cho trạng thái ban đầu của bàn cờ khi chưa qua xử lý nơ-ron.",
    tacticalNote: "<strong>Ý nghĩa cờ tướng:</strong> Nơ-ron chỉ biết đúng quân cờ tại chỗ (ví dụ Pháo, Mã hay ô trống). Nó hoàn toàn 'mù' về môi trường xung quanh, chưa biết ô kế bên an toàn hay đang bị uy hiếp.",
  },
  {
    shortName: "Tầng 1",
    title: "Input Conv · 1 tầng Conv 3×3",
    code: "conv_input (3×3, pad 1)",
    convCount: 1,
    rf: 3,
    color: "#86b6ef",
    summary: "Kernel 3×3 trượt qua ô tâm cùng 8 ô tiếp giáp. Tầng này thu thập ngữ cảnh không gian cục bộ tức thì.",
    insight: "Bán kính quan sát là 1 ô. Đủ để phát hiện quân cản chân Mã đứng ngay sát cạnh, hoặc hai quân che chở cho nhau.",
    tacticalNote: "<strong>Ý nghĩa cờ tướng:</strong> Đủ phát hiện cản chân Mã sát nách, thế Tướng đối mặt cự ly 1 ô, hoặc quân đang áp sát. Nhưng chưa thể thấy nước Mã nhảy (cách 2 ô) hay tầm bắn của Pháo.",
  },
  {
    shortName: "Tầng 2",
    title: "Block 1 · Conv 1",
    code: "blocks[0].conv1 (3×3)",
    convCount: 2,
    rf: 5,
    color: "#5598e7",
    summary: "Hiệu ứng bắc cầu bắt đầu: Mỗi ô trong vùng 3×3 mà Conv 1 đọc đã tích lũy sẵn RF 3×3 từ tầng trước, mở rộng biên lên 5×5.",
    insight: "Tầng sau đọc đặc trưng đã được tầng trước tổng hợp, không đọc dữ liệu thô. RF tăng thêm đúng (3-1) = 2 ô.",
    tacticalNote: "<strong>Ý nghĩa cờ tướng:</strong> Bao trọn vẹn toàn bộ 8 điểm đến của một quân Mã (tối đa cách 2 ô theo cả hàng và cột). Mạng bắt đầu 'hiểu' đường đi nước bước hoàn chỉnh của Mã và Tượng trong phạm vi 5×5.",
  },
  {
    shortName: "Tầng 3",
    title: "Block 1 · Conv 2",
    code: "blocks[0].conv2 (3×3)",
    convCount: 3,
    rf: 7,
    color: "#2a78d6",
    summary: "Nhánh tích chập của Block 1 đạt RF tối đa 7×7 trước khi cộng phần tử với đường tắt Identity skip-connection.",
    insight: "Đầu ra Block 1 kết hợp song song: đường tắt giữ lại ngữ cảnh cục bộ sắc nét, nhánh chính mở rộng tầm nhìn lên 7×7.",
    tacticalNote: "<strong>Ý nghĩa cờ tướng:</strong> Kích thước 7×7 bao trùm toàn bộ Cung tướng (3×3) cùng hệ thống phòng thủ Sĩ - Tượng xung quanh. Nhận diện các thế phối hợp phòng ngự liên hoàn.",
  },
  {
    shortName: "Tầng 4",
    title: "Block 2 · Conv 1",
    code: "blocks[1].conv1 (3×3)",
    convCount: 4,
    rf: 9,
    color: "#1c5cab",
    summary: "Nhánh dài nhất tiếp tục mở rộng lên RF 9×9. Tại các cột giữa, footprint đã chạm tới cả 2 mép biên trái và phải của bàn cờ.",
    insight: "Bán kính quan sát lên tới 4 ô. Độ bao phủ ngang đạt 100% chiều rộng bàn cờ Tướng (9 cột).",
    tacticalNote: "<strong>Ý nghĩa cờ tướng:</strong> Đủ sức quan sát toàn bộ một hàng ngang 9 ô. Bắt đầu nhận diện đường chuyển cánh của Xe từ lộ 1 sang lộ 9 và các đòn Pháo bắn xuyên sông tầm xa.",
  },
  {
    shortName: "Tầng 5",
    title: "Block 2 · Conv 2",
    code: "blocks[1].conv2 (3×3)",
    convCount: 5,
    rf: 11,
    color: "#104281",
    summary: "Sau 5 tầng Conv 3×3 liên tiếp trên nhánh chính, trường cảm thụ lý thuyết đạt 11×11 (121 ô không gian).",
    insight: "Nếu nơ-ron nằm ở vùng trung tâm bàn cờ, footprint 11×11 bao trọn toàn bộ 90 ô (10 hàng × 9 cột) của bàn cờ Tướng!",
    tacticalNote: "<strong>Ý nghĩa cờ tướng:</strong> Tầm nhìn toàn cục (Global Context)! Một nơ-ron trung tâm có thể kết nối chiến lược giữa Xe cánh trái, Pháo trung lộ và Tướng trong cung để đưa ra quyết định tối ưu cho Policy & Value head.",
  },
];

const PIECES = new Map([
  ["0,0", ["Xe", "red"]], ["0,1", ["Mã", "red"]], ["0,2", ["Tượng", "red"]],
  ["0,3", ["Sĩ", "red"]], ["0,4", ["Tướng", "red"]], ["0,5", ["Sĩ", "red"]],
  ["0,6", ["Tượng", "red"]], ["0,7", ["Mã", "red"]], ["0,8", ["Xe", "red"]],
  ["2,1", ["Pháo", "red"]], ["2,7", ["Pháo", "red"]],
  ["3,0", ["Tốt", "red"]], ["3,2", ["Tốt", "red"]], ["3,4", ["Tốt", "red"]],
  ["3,6", ["Tốt", "red"]], ["3,8", ["Tốt", "red"]],
  ["6,0", ["Tốt", "black"]], ["6,2", ["Tốt", "black"]], ["6,4", ["Tốt", "black"]],
  ["6,6", ["Tốt", "black"]], ["6,8", ["Tốt", "black"]],
  ["7,1", ["Pháo", "black"]], ["7,7", ["Pháo", "black"]],
  ["9,0", ["Xe", "black"]], ["9,1", ["Mã", "black"]], ["9,2", ["Tượng", "black"]],
  ["9,3", ["Sĩ", "black"]], ["9,4", ["Tướng", "black"]], ["9,5", ["Sĩ", "black"]],
  ["9,6", ["Tượng", "black"]], ["9,7", ["Mã", "black"]], ["9,8", ["Xe", "black"]],
]);

const elements = {
  stageRail: document.querySelector("#stage-rail"),
  boardGrid: document.querySelector("#board-grid"),
  columnAxis: document.querySelector("#column-axis"),
  rowAxis: document.querySelector("#row-axis"),
  tooltip: document.querySelector("#cell-tooltip"),
  previousButton: document.querySelector("#previous-button"),
  nextButton: document.querySelector("#next-button"),
  playButton: document.querySelector("#play-button"),
  playIcon: document.querySelector("#play-icon"),
  playLabel: document.querySelector("#play-label"),
  fullscreenButton: document.querySelector("#fullscreen-button"),
  copyTakeaway: document.querySelector("#copy-takeaway"),
  toast: document.querySelector("#toast"),
  layerNumber: document.querySelector("#layer-number"),
  layerCode: document.querySelector("#layer-code"),
  layerTitle: document.querySelector("#layer-title"),
  layerSummary: document.querySelector("#layer-summary"),
  rfValue: document.querySelector("#rf-value"),
  rfRadius: document.querySelector("#rf-radius"),
  realCellCount: document.querySelector("#real-cell-count"),
  paddingCellCount: document.querySelector("#padding-cell-count"),
  pieceCount: document.querySelector("#piece-count"),
  rfFormula: document.querySelector("#rf-formula"),
  positionMessage: document.querySelector("#position-message"),
  layerInsight: document.querySelector("#layer-insight"),
  chessTacticalNote: document.querySelector("#chess-tactical-note"),
  previousFootprint: document.querySelector("#previous-footprint"),
  currentFootprint: document.querySelector("#current-footprint"),
  previousRfLabel: document.querySelector("#previous-rf-label"),
  currentRfLabel: document.querySelector("#current-rf-label"),
  bridgeEquation: document.querySelector("#bridge-equation"),
  bridgeHeading: document.querySelector("#bridge-heading"),
  bridgeDescription: document.querySelector("#bridge-description"),
  residualInputRf: document.querySelector("#residual-input-rf"),
  mainBranchRf: document.querySelector("#main-branch-rf"),
  skipBranchRf: document.querySelector("#skip-branch-rf"),
  residualOutputRf: document.querySelector("#residual-output-rf"),
  residualExplanation: document.querySelector("#residual-explanation"),
  tableBody: document.querySelector("#rf-table-body"),
  deckTabs: document.querySelectorAll(".deck-tab"),
  tabPanels: document.querySelectorAll(".tab-panel"),
};

let currentStage = 0;
let centerRow = 4;
let centerCol = 4;
let playing = false;
let playTimer = null;
let toastTimer = null;
let selectedResidualBlock = 1;

function boardKey(row, col) {
  return `${row},${col}`;
}

function footprintBounds(stage, row = centerRow, col = centerCol) {
  const radius = Math.floor(stage.rf / 2);
  return {
    radius,
    minRow: row - radius,
    maxRow: row + radius,
    minCol: col - radius,
    maxCol: col + radius,
  };
}

function footprintStats(stage) {
  const bounds = footprintBounds(stage);
  let realCells = 0;
  let pieces = 0;

  for (let row = bounds.minRow; row <= bounds.maxRow; row += 1) {
    for (let col = bounds.minCol; col <= bounds.maxCol; col += 1) {
      if (row < 0 || row >= BOARD_ROWS || col < 0 || col >= BOARD_COLS) {
        continue;
      }
      realCells += 1;
      if (PIECES.has(boardKey(row, col))) {
        pieces += 1;
      }
    }
  }

  return {
    ...bounds,
    realCells,
    paddingCells: stage.rf * stage.rf - realCells,
    pieces,
  };
}

function renderStages() {
  elements.stageRail.replaceChildren(...STAGES.map((stage, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `stage-button${index === currentStage ? " active" : ""}`;
    button.dataset.stage = String(index);
    button.setAttribute("aria-label", `${stage.title}, trường cảm thụ ${stage.rf} nhân ${stage.rf}`);
    button.setAttribute("aria-pressed", String(index === currentStage));
    button.innerHTML = `
      <span class="stage-dot">${index}</span>
      <span class="stage-name">${stage.shortName}</span>
      <span class="stage-rf">${stage.rf}×${stage.rf}</span>
    `;
    return button;
  }));
}

function renderAxes() {
  elements.columnAxis.replaceChildren(...Array.from({ length: BOARD_COLS }, (_, col) => {
    const label = document.createElement("span");
    label.textContent = String(col);
    return label;
  }));
  elements.rowAxis.replaceChildren(...Array.from({ length: BOARD_ROWS }, (_, row) => {
    const label = document.createElement("span");
    label.textContent = String(row);
    return label;
  }));
}

function createPiece(piece) {
  const [name, side] = piece;
  const token = document.createElement("span");
  token.className = `piece ${side}`;
  token.textContent = name;
  token.setAttribute("aria-hidden", "true");
  return token;
}

function renderBoard() {
  const stage = STAGES[currentStage];
  const stats = footprintStats(stage);
  const cells = [];

  for (let row = 0; row < BOARD_ROWS; row += 1) {
    for (let col = 0; col < BOARD_COLS; col += 1) {
      const key = boardKey(row, col);
      const piece = PIECES.get(key);
      const inField = row >= stats.minRow && row <= stats.maxRow
        && col >= stats.minCol && col <= stats.maxCol;
      const selected = row === centerRow && col === centerCol;
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = [
        "board-cell",
        inField ? "in-field" : "outside-field",
        selected ? "selected" : "",
        row === 4 ? "river-top" : "",
        row === 5 ? "river-bottom" : "",
      ].filter(Boolean).join(" ");
      cell.dataset.row = String(row);
      cell.dataset.col = String(col);
      cell.style.setProperty("--field-color", stage.color);
      cell.setAttribute("aria-label", `Hàng ${row}, cột ${col}${piece ? `, ${piece[0]} ${piece[1] === "red" ? "đỏ" : "đen"}` : ", ô trống"}${inField ? ", nằm trong trường cảm thụ" : ", ngoài trường cảm thụ"}`);
      cell.setAttribute("aria-pressed", String(selected));
      if (piece) {
        cell.append(createPiece(piece));
      }
      cells.push(cell);
    }
  }

  elements.boardGrid.replaceChildren(...cells);
  updateExplanation(stage, stats);
}

function updateExplanation(stage, stats) {
  const theoreticalCells = stage.rf * stage.rf;
  const coverage = Math.round((stats.realCells / TOTAL_BOARD_CELLS) * 100);
  const fullBoard = stats.realCells === TOTAL_BOARD_CELLS;

  elements.layerNumber.textContent = currentStage === 0 ? "Tầng 0 · Chưa qua Conv" : `Tầng ${currentStage} · Conv thứ ${stage.convCount}`;
  elements.layerCode.textContent = stage.code;
  elements.layerTitle.textContent = stage.title;
  elements.layerSummary.textContent = stage.summary;
  elements.rfValue.textContent = `${stage.rf} × ${stage.rf}`;
  elements.rfRadius.textContent = `Bán kính r = ${stats.radius} ô`;
  elements.realCellCount.textContent = `${stats.realCells} / ${TOTAL_BOARD_CELLS}`;
  elements.paddingCellCount.textContent = `${stats.paddingCells} ô`;
  elements.pieceCount.textContent = `${stats.pieces} / ${TOTAL_PIECES}`;
  elements.rfFormula.textContent = `RF = 1 + 2×${stage.convCount} = ${stage.rf}`;
  elements.layerInsight.textContent = stage.insight;

  if (elements.chessTacticalNote) {
    elements.chessTacticalNote.innerHTML = stage.tacticalNote;
  }

  if (fullBoard) {
    elements.positionMessage.textContent = `Tâm (${centerRow}, ${centerCol}) · Footprint 11×11 bao trọn 100% (90 ô) bàn cờ!`;
  } else if (stats.paddingCells > 0) {
    elements.positionMessage.textContent = `Tâm (${centerRow}, ${centerCol}) · ${stats.paddingCells} ô ngoài biên (padding 0); Vùng thật: ${stats.realCells}/90 ô (${coverage}%).`;
  } else {
    elements.positionMessage.textContent = `Tâm (${centerRow}, ${centerCol}) · Trọn vẹn ${theoreticalCells} ô nằm trên bàn cờ (${coverage}%).`;
  }
}

function renderFootprint(container, size, color) {
  const displaySize = Math.min(size, 11);
  container.style.gridTemplateColumns = `repeat(${displaySize}, 1fr)`;
  container.style.gridTemplateRows = `repeat(${displaySize}, 1fr)`;
  container.replaceChildren(...Array.from({ length: displaySize * displaySize }, (_, index) => {
    const cell = document.createElement("span");
    const center = Math.floor(displaySize / 2);
    const row = Math.floor(index / displaySize);
    const col = index % displaySize;
    cell.className = `footprint-cell${row === center && col === center ? " center" : ""}`;
    if (!(row === center && col === center)) {
      cell.style.background = color;
    }
    return cell;
  }));
}

function renderBridge() {
  const targetIndex = currentStage === 0 ? 1 : currentStage;
  const current = STAGES[targetIndex];
  const previous = STAGES[targetIndex - 1];

  renderFootprint(elements.previousFootprint, previous.rf, previous.color);
  renderFootprint(elements.currentFootprint, current.rf, current.color);
  elements.previousRfLabel.textContent = `${previous.rf} × ${previous.rf}`;
  elements.currentRfLabel.textContent = `${current.rf} × ${current.rf}`;
  elements.bridgeEquation.textContent = `${previous.rf} + 2 = ${current.rf}`;
  elements.bridgeHeading.textContent = `Từ ${previous.rf}×${previous.rf} lên ${current.rf}×${current.rf}`;
  elements.bridgeDescription.textContent = `Conv thứ ${current.convCount} quét vùng kernel 3×3 trên feature map trước. Mỗi điểm trong vùng đó đã đại diện cho một footprint ${previous.rf}×${previous.rf} của bàn cờ gốc. Khi các footprint này lệch nhau 1 ô và hợp lại, biên trái, phải, trên, dưới cùng mở thêm 1 ô, giúp RF tăng thêm đúng 2 ô (thành ${current.rf}×${current.rf}).`;
}

function renderResidualBlock() {
  const block = selectedResidualBlock;
  const inputRf = block === 1 ? 3 : 7;
  const outputRf = block === 1 ? 7 : 11;
  const convOneRf = inputRf + 2;

  elements.residualInputRf.textContent = `RF ${inputRf}×${inputRf}`;
  elements.mainBranchRf.textContent = `RF ${outputRf}×${outputRf}`;
  elements.skipBranchRf.textContent = `RF ${inputRf}×${inputRf}`;
  elements.residualOutputRf.textContent = `RF tối đa ${outputRf}×${outputRf}`;
  elements.residualExplanation.textContent = `Block ${block}: Nhánh chính đi qua 2 tầng Conv 3×3 (conv1 và conv2) nên RF tăng liên tục: ${inputRf}×${inputRf} → ${convOneRf}×${convOneRf} → ${outputRf}×${outputRf}. Nhánh identity (skip-connection) truyền thẳng tensor đầu vào nên bảo lưu nguyên vẹn trường cảm thụ ban đầu ${inputRf}×${inputRf}. Phép cộng (out + residual) giúp mạng nơ-ron vừa nắm bắt được thông tin ngữ cảnh diện rộng (${outputRf}×${outputRf}) vừa không làm mất thông tin định vị quân cờ cục bộ sắc nét.`;
}

function renderTable() {
  elements.tableBody.replaceChildren(...STAGES.map((stage, index) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${index}. ${stage.shortName}</td>
      <td><code>${stage.code}</code></td>
      <td>${stage.convCount}</td>
      <td><strong>${stage.rf} × ${stage.rf}</strong></td>
      <td>Bán kính ${Math.floor(stage.rf / 2)} ô</td>
    `;
    return row;
  }));
}

function updatePresetButtons() {
  document.querySelectorAll("[data-position]").forEach((button) => {
    const [row, col] = button.dataset.position.split(",").map(Number);
    button.classList.toggle("active", row === centerRow && col === centerCol);
  });
}

function setStage(index) {
  currentStage = Math.max(0, Math.min(STAGES.length - 1, index));
  renderStages();
  renderBoard();
  renderBridge();
}

function setCenter(row, col) {
  centerRow = row;
  centerCol = col;
  updatePresetButtons();
  renderBoard();
}

function stopPlayback() {
  playing = false;
  window.clearInterval(playTimer);
  playTimer = null;
  if (elements.playIcon) elements.playIcon.textContent = "▶";
  if (elements.playLabel) elements.playLabel.textContent = "Trình diễn";
}

function startPlayback() {
  stopPlayback();
  playing = true;
  if (elements.playIcon) elements.playIcon.textContent = "⏸";
  if (elements.playLabel) elements.playLabel.textContent = "Tạm dừng";
  setStage(0);
  playTimer = window.setInterval(() => {
    if (currentStage === STAGES.length - 1) {
      stopPlayback();
      return;
    }
    setStage(currentStage + 1);
  }, PLAY_INTERVAL_MS);
}

function togglePlayback() {
  if (playing) {
    stopPlayback();
    return;
  }
  startPlayback();
}

function showTooltip(event, cell) {
  const row = Number(cell.dataset.row);
  const col = Number(cell.dataset.col);
  const stage = STAGES[currentStage];
  const bounds = footprintBounds(stage);
  const inside = row >= bounds.minRow && row <= bounds.maxRow
    && col >= bounds.minCol && col <= bounds.maxCol;
  const piece = PIECES.get(boardKey(row, col));
  const pieceText = piece ? `${piece[0]} ${piece[1] === "red" ? "đỏ" : "đen"}` : "Ô trống";
  elements.tooltip.textContent = `(${row}, ${col}) · ${pieceText} · ${inside ? "Nằm TRONG trường cảm thụ" : "Ngoài RF ở tầng này"}`;
  elements.tooltip.hidden = false;
  moveTooltip(event);
}

function moveTooltip(event) {
  const gap = 14;
  const width = elements.tooltip.offsetWidth;
  const height = elements.tooltip.offsetHeight;
  const left = Math.min(event.clientX + gap, window.innerWidth - width - gap);
  const top = Math.min(event.clientY + gap, window.innerHeight - height - gap);
  elements.tooltip.style.left = `${Math.max(gap, left)}px`;
  elements.tooltip.style.top = `${Math.max(gap, top)}px`;
}

function hideTooltip() {
  elements.tooltip.hidden = true;
}

function showToast(message) {
  window.clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  toastTimer = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 2200);
}

async function copyTakeaway() {
  try {
    await navigator.clipboard.writeText(TAKEAWAY);
    showToast("Đã sao chép câu chốt thuyết trình!");
  } catch {
    showToast("Không thể truy cập clipboard; hãy sao chép thủ công.");
  }
}

async function toggleFullscreen() {
  try {
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      await document.documentElement.requestFullscreen();
    }
  } catch {
    showToast("Trình duyệt không cho phép chuyển chế độ toàn màn hình.");
  }
}

function switchTab(targetTab) {
  elements.deckTabs.forEach((tab) => {
    const isTarget = tab.dataset.tab === targetTab;
    tab.classList.toggle("active", isTarget);
    tab.setAttribute("aria-selected", String(isTarget));
  });

  elements.tabPanels.forEach((panel) => {
    const isTarget = panel.id === `tab-panel-${targetTab}`;
    panel.classList.toggle("active", isTarget);
    panel.hidden = !isTarget;
  });
}

function bindEvents() {
  // Tab switcher
  elements.deckTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      switchTab(tab.dataset.tab);
    });
  });

  // Stage buttons
  elements.stageRail.addEventListener("click", (event) => {
    const button = event.target.closest("[data-stage]");
    if (!button) {
      return;
    }
    stopPlayback();
    setStage(Number(button.dataset.stage));
  });

  // Board cell click
  elements.boardGrid.addEventListener("click", (event) => {
    const cell = event.target.closest(".board-cell");
    if (cell) {
      setCenter(Number(cell.dataset.row), Number(cell.dataset.col));
    }
  });

  // Tooltips
  elements.boardGrid.addEventListener("pointerover", (event) => {
    const cell = event.target.closest(".board-cell");
    if (cell) {
      showTooltip(event, cell);
    }
  });
  elements.boardGrid.addEventListener("pointermove", (event) => {
    if (!elements.tooltip.hidden) {
      moveTooltip(event);
    }
  });
  elements.boardGrid.addEventListener("pointerleave", hideTooltip);
  elements.boardGrid.addEventListener("focusin", (event) => {
    const cell = event.target.closest(".board-cell");
    if (!cell) {
      return;
    }
    const rect = cell.getBoundingClientRect();
    showTooltip({ clientX: rect.right, clientY: rect.top }, cell);
  });
  elements.boardGrid.addEventListener("focusout", hideTooltip);

  // Position presets
  document.querySelectorAll("[data-position]").forEach((button) => {
    button.addEventListener("click", () => {
      const [row, col] = button.dataset.position.split(",").map(Number);
      setCenter(row, col);
    });
  });

  // Navigation actions
  elements.previousButton.addEventListener("click", () => {
    stopPlayback();
    setStage((currentStage - 1 + STAGES.length) % STAGES.length);
  });
  elements.nextButton.addEventListener("click", () => {
    stopPlayback();
    setStage((currentStage + 1) % STAGES.length);
  });
  elements.playButton.addEventListener("click", togglePlayback);
  elements.fullscreenButton.addEventListener("click", toggleFullscreen);
  elements.copyTakeaway.addEventListener("click", copyTakeaway);

  // Residual block switchers
  document.querySelectorAll("[data-block]").forEach((button) => {
    button.addEventListener("click", () => {
      selectedResidualBlock = Number(button.dataset.block);
      document.querySelectorAll("[data-block]").forEach((tab) => {
        const selected = tab === button;
        tab.classList.toggle("active", selected);
        tab.setAttribute("aria-selected", String(selected));
      });
      renderResidualBlock();
    });
  });

  document.addEventListener("fullscreenchange", () => {
    elements.fullscreenButton.textContent = document.fullscreenElement ? "Thu nhỏ" : "Toàn màn hình";
  });

  document.addEventListener("keydown", (event) => {
    if (event.target.matches("button, a, summary, input")) {
      return;
    }
    if (event.key === "ArrowLeft") {
      stopPlayback();
      setStage((currentStage - 1 + STAGES.length) % STAGES.length);
    } else if (event.key === "ArrowRight") {
      stopPlayback();
      setStage((currentStage + 1) % STAGES.length);
    } else if (event.key === " ") {
      event.preventDefault();
      togglePlayback();
    }
  });
}

// Initial boot
renderAxes();
renderTable();
renderResidualBlock();
updatePresetButtons();
setStage(0);
bindEvents();
