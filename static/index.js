/**
 * index.js — Cover Type Explorer Frontend
 * Renders terrain grid, smooth water/road overlays, lazy predictions.
 */

// ── Tree image filenames ──────────────────────────────────────────────────
const TREE_IMAGES = {
    "Spruce/Fir":         "spruce_fir.png",
    "Lodgepole Pine":     "lodgepole_pine.png",
    "Ponderosa Pine":     "ponderosa_pine.png",
    "Cottonwood/Willow":  "cottonwood_willow.png",
    "Aspen":              "aspen.png",
    "Douglas-fir":        "douglas_fir.png",
    "Krummholz":          "krummholz.png",
};

// ── State ──────────────────────────────────────────────────────────────────
let mapData = null;
let cellGrid = [];
let selectedCell = null;
let highlightedType = null;
let batchPredictions = null;  // Lazy: only populated when legend clicked
let batchLoading = false;
let cellSize = 5;  // Computed dynamically

// ── DOM refs ───────────────────────────────────────────────────────────────
const btnNewMap       = document.getElementById('btn-new-map');
const seedDisplay     = document.getElementById('seed-display');
const canvas          = document.getElementById('grid-canvas');
const ctx             = canvas.getContext('2d');
const loadingOverlay  = document.getElementById('loading-overlay');
const tooltip         = document.getElementById('tooltip');
const legendItems     = document.getElementById('legend-items');
const panelEmpty      = document.getElementById('panel-empty');
const panelContent    = document.getElementById('panel-content');
const panelSimple     = document.getElementById('panel-simple');

// ── Init ───────────────────────────────────────────────────────────────────
btnNewMap.addEventListener('click', () => fetchMap());
canvas.addEventListener('click', onCanvasClick);
canvas.addEventListener('mousemove', onCanvasHover);
canvas.addEventListener('mouseleave', () => tooltip.classList.remove('visible'));
fetchMap();


// ── API ────────────────────────────────────────────────────────────────────
async function fetchMap(seed) {
    showLoading(true);
    btnNewMap.disabled = true;

    try {
        const url = seed != null
            ? `/api/generate_map?seed=${seed}`
            : '/api/generate_map';
        const resp = await fetch(url);
        mapData = await resp.json();

        seedDisplay.textContent = `Seed: ${mapData.seed}`;
        buildCellGrid();
        computeCellSize();
        resizeCanvas();
        renderGrid();
        showPanel('empty');
        selectedCell = null;
        highlightedType = null;
        batchPredictions = null;  // Reset lazy cache on new map
    } catch (err) {
        console.error('Failed to generate map:', err);
    } finally {
        showLoading(false);
        btnNewMap.disabled = false;
    }
}


// ── Grid data ──────────────────────────────────────────────────────────────
function buildCellGrid() {
    const { grid_width, grid_height, cells } = mapData;
    cellGrid = Array.from({ length: grid_height }, () =>
        new Array(grid_width).fill(null)
    );
    for (const cell of cells) {
        cellGrid[cell.r][cell.c] = cell;
    }
}


// ── Canvas rendering ───────────────────────────────────────────────────────
function computeCellSize() {
    // Fill available space while maintaining square cells
    const wrapper = document.getElementById('canvas-wrapper');
    const availW = wrapper.clientWidth - 20;
    const availH = wrapper.clientHeight - 20;
    const gw = mapData.grid_width;
    const gh = mapData.grid_height;
    cellSize = Math.max(2, Math.floor(Math.min(availW / gw, availH / gh)));
}

function resizeCanvas() {
    const w = mapData.grid_width * cellSize;
    const h = mapData.grid_height * cellSize;
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
}

function renderGrid() {
    if (!mapData) return;
    const { grid_width, grid_height } = mapData;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Draw all terrain cells
    for (let r = 0; r < grid_height; r++) {
        for (let c = 0; c < grid_width; c++) {
            const cell = cellGrid[r][c];
            if (!cell) continue;

            const x = c * cellSize;
            const y = r * cellSize;

            ctx.fillStyle = cell.color;
            ctx.fillRect(x, y, cellSize, cellSize);

            // Dim cells not matching highlighted type (from lazy batch)
            if (highlightedType !== null && batchPredictions) {
                const pred = batchPredictions[`${r},${c}`];
                if (pred && pred.cover_name !== highlightedType) {
                    ctx.fillStyle = 'rgba(10, 13, 20, 0.6)';
                    ctx.fillRect(x, y, cellSize, cellSize);
                } else if (pred && pred.cover_name === highlightedType) {
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
                    ctx.lineWidth = 1;
                    ctx.strokeRect(x + 0.5, y + 0.5, cellSize - 1, cellSize - 1);
                }
            }

            // Selected cell outline
            if (selectedCell && selectedCell.r === r && selectedCell.c === c) {
                ctx.strokeStyle = '#FBBF24';
                ctx.lineWidth = 2;
                ctx.strokeRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
            }
        }
    }
}


// ── Legend (lazy batch prediction) ─────────────────────────────────────────
const CLASS_NAMES_LIST = [
    "Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine",
    "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"
];
const CLASS_COLORS_LIST = [
    "#2D5F3E", "#5B8C5A", "#C4A35A", "#7ABF7E", "#E8C547", "#3E7A5E", "#8B6F47"
];

// Attach legend click handlers
document.querySelectorAll('.legend-item').forEach(item => {
    item.addEventListener('click', () => {
        const label = item.textContent.trim();
        if (CLASS_NAMES_LIST.includes(label)) {
            // Not a terrain legend item; skip
        }
        // For now legend items are static terrain labels, so we skip
    });
});

async function doBatchPredict() {
    if (batchPredictions || batchLoading) return;
    batchLoading = true;

    const allVectors = [];
    const indices = [];
    const { grid_width, grid_height } = mapData;
    for (let r = 0; r < grid_height; r++) {
        for (let c = 0; c < grid_width; c++) {
            const cell = cellGrid[r][c];
            if (cell && !cell.is_water && !cell.is_road && cell.feature_vector) {
                allVectors.push(cell.feature_vector);
                indices.push({ r, c });
            }
        }
    }

    try {
        const resp = await fetch('/api/batch_predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ vectors: allVectors })
        });
        const data = await resp.json();
        batchPredictions = {};
        for (let i = 0; i < indices.length; i++) {
            const { r, c } = indices[i];
            batchPredictions[`${r},${c}`] = data.predictions[i];
        }
    } catch (err) {
        console.error('Batch prediction failed:', err);
    } finally {
        batchLoading = false;
    }
}


// ── Canvas interaction ─────────────────────────────────────────────────────
function getCellAt(event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    const c = Math.floor(x / cellSize);
    const r = Math.floor(y / cellSize);
    if (r >= 0 && r < mapData.grid_height && c >= 0 && c < mapData.grid_width) {
        return cellGrid[r][c];
    }
    return null;
}

function onCanvasClick(event) {
    const cell = getCellAt(event);
    if (!cell) return;

    selectedCell = { r: cell.r, c: cell.c };
    renderGrid();

    if (cell.is_water) {
        showWaterPanel(cell);
    } else if (cell.is_road) {
        showRoadPanel(cell);
    } else {
        showLandPanel(cell);
    }
}

function onCanvasHover(event) {
    const cell = getCellAt(event);
    if (!cell) {
        tooltip.classList.remove('visible');
        return;
    }

    const rect = canvas.getBoundingClientRect();
    let text = '';
    if (cell.is_water) {
        text = `Water · (${cell.r}, ${cell.c})`;
    } else if (cell.is_road) {
        text = `Road · (${cell.r}, ${cell.c})`;
    } else {
        text = `Terrain · (${cell.r}, ${cell.c}) · Elev: ${cell.features?.Elevation ?? '—'}`;
    }

    tooltip.textContent = text;
    tooltip.classList.add('visible');

    const tx = event.clientX - rect.left + 14;
    const ty = event.clientY - rect.top - 30;
    tooltip.style.left = tx + 'px';
    tooltip.style.top = ty + 'px';
}


// ── Side Panel ─────────────────────────────────────────────────────────────
function showPanel(mode) {
    panelEmpty.classList.toggle('hidden', mode !== 'empty');
    panelContent.classList.toggle('hidden', mode !== 'land');
    panelSimple.classList.toggle('hidden', mode !== 'simple');
}

function getTerrainType(t) {
    if (t < 0.2) return 'Dense Forest Valley';
    if (t < 0.45) return 'Green Slopes';
    if (t < 0.7) return 'Mid-Altitude Scrubland';
    if (t < 0.85) return 'Rocky Highlands';
    return 'Snow-Capped Peak';
}

function showLandPanel(cell) {
    showPanel('land');

    // Header badge
    const badge = document.getElementById('cell-type-badge');
    const terrainType = getTerrainType(cell.elevation_score || 0);
    badge.innerHTML = `<span class="badge-dot" style="background:${cell.color}"></span>${terrainType}`;
    badge.style.background = hexToRgba(cell.color, 0.15);
    badge.style.color = lightenColor(cell.color, 0.4);
    document.getElementById('cell-coords').textContent = `Cell (${cell.r}, ${cell.c})`;

    // ── Features first ──
    const contTbody = document.querySelector('#continuous-features tbody');
    contTbody.innerHTML = '';
    const contFeatures = [
        'Elevation', 'Aspect', 'Slope',
        'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points',
    ];
    for (const name of contFeatures) {
        const val = cell.features[name];
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${formatFeatureName(name)}</td><td>${val}</td>`;
        contTbody.appendChild(tr);
    }

    // Wilderness chips
    const wildDiv = document.getElementById('wilderness-chips');
    wildDiv.innerHTML = '';
    for (let i = 1; i <= 4; i++) {
        const key = `Wilderness_Area_${i}`;
        const val = cell.features[key];
        const chip = document.createElement('span');
        chip.className = 'chip' + (val === 1 ? ' active' : '');
        chip.textContent = `Area ${i}`;
        wildDiv.appendChild(chip);
    }

    // Soil chips
    const soilDiv = document.getElementById('soil-chips');
    soilDiv.innerHTML = '';
    for (let i = 1; i <= 40; i++) {
        const key = `Soil_Type_${i}`;
        const val = cell.features[key];
        const chip = document.createElement('span');
        chip.className = 'chip' + (val === 1 ? ' active' : '');
        chip.textContent = `${i}`;
        soilDiv.appendChild(chip);
    }

    // ── Prediction (async, below features) ──
    const predDiv = document.getElementById('predicted-class');
    const probContainer = document.getElementById('probability-bars');
    const treeSection = document.getElementById('tree-image-section');
    const treeImg = document.getElementById('tree-image');
    const treeCap = document.getElementById('tree-image-caption');

    predDiv.innerHTML = `<span style="color:#666">Analyzing 54 features…</span>`;
    probContainer.innerHTML = '';
    treeSection.classList.add('hidden');

    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: cell.feature_vector })
    })
    .then(r => r.json())
    .then(data => {
        if (data.error) {
            predDiv.innerHTML = `<span style="color:red">Error: ${data.error}</span>`;
            return;
        }

        // Show tree image
        const imgFile = TREE_IMAGES[data.cover_name];
        if (imgFile) {
            treeImg.src = `/static/images/${imgFile}`;
            treeImg.alt = data.cover_name;
            treeCap.textContent = data.cover_name;
            treeSection.classList.remove('hidden');
            treeImg.onerror = () => { treeSection.classList.add('hidden'); };
        }

        // Prediction result
        predDiv.innerHTML = `<span class="pred-color" style="background:#4ADE80"></span><span class="pred-name">${data.cover_name}</span>`;

        // Probability bars
        const sorted = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
        probContainer.innerHTML = '';
        for (const [name, prob] of sorted) {
            const row = document.createElement('div');
            row.className = 'prob-row';
            const barCol = name === data.cover_name ? '#4ADE80' : '#666';
            row.innerHTML = `
                <span class="prob-label">${name}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:${Math.max(1, prob * 100)}%;background:${barCol}"></div>
                </div>
                <span class="prob-value">${(prob * 100).toFixed(1)}%</span>
            `;
            probContainer.appendChild(row);
        }
    })
    .catch(() => {
        predDiv.innerHTML = `<span style="color:red">Server error during prediction</span>`;
    });
}

function showWaterPanel(cell) {
    showPanel('simple');
    document.getElementById('simple-icon').style.background = 'rgba(20, 112, 160, 0.2)';
    document.getElementById('simple-icon').textContent = '🌊';
    document.getElementById('simple-label').textContent = 'Water';
    document.getElementById('simple-label').style.color = '#22D3EE';
    document.getElementById('simple-desc').textContent =
        'This cell is near water — not classified by the model. Water bodies are generated procedurally.';
}

function showRoadPanel(cell) {
    showPanel('simple');
    document.getElementById('simple-icon').style.background = 'rgba(107, 107, 107, 0.2)';
    document.getElementById('simple-icon').textContent = '🛤️';
    document.getElementById('simple-label').textContent = 'Road';
    document.getElementById('simple-label').style.color = '#9CA3AF';
    document.getElementById('simple-desc').textContent =
        'This cell is near a road — not classified by the model. Roads are generated as straight lines across terrain.';
}


// ── Helpers ────────────────────────────────────────────────────────────────
function showLoading(visible) {
    loadingOverlay.classList.toggle('visible', visible);
}

function formatFeatureName(name) {
    return name.replace(/_/g, ' ').replace(/To /g, 'to ');
}

function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}

function lightenColor(hex, amount) {
    let r = parseInt(hex.slice(1, 3), 16);
    let g = parseInt(hex.slice(3, 5), 16);
    let b = parseInt(hex.slice(5, 7), 16);
    r = Math.min(255, Math.round(r + (255 - r) * amount));
    g = Math.min(255, Math.round(g + (255 - g) * amount));
    b = Math.min(255, Math.round(b + (255 - b) * amount));
    return `rgb(${r},${g},${b})`;
}

// ── Modal Logic ─────────────────────────────────────────────────────────────
const modal = document.getElementById('about-modal');
document.getElementById('btn-about-model').addEventListener('click', () => {
    modal.classList.remove('hidden');

    fetch('/api/metrics')
        .then(r => r.json())
        .then(data => {
            const mContainer = document.getElementById('modal-metrics');
            if (data.error) {
                mContainer.innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
                return;
            }

            const acc = (data.accuracy * 100).toFixed(1);
            const f1 = (data.macro_f1 * 100).toFixed(1);

            mContainer.innerHTML = `
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-val">${acc}%</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">${f1}%</div>
                        <div class="metric-label">Macro F1</div>
                    </div>
                </div>
            `;
        })
        .catch(() => {
            document.getElementById('modal-metrics').innerHTML = `<p style="color:red;">Failed to load metrics.</p>`;
        });
});

document.getElementById('close-modal-btn').addEventListener('click', () => {
    modal.classList.add('hidden');
});

modal.addEventListener('click', (e) => {
    if (e.target === modal) modal.classList.add('hidden');
});

// ── Resize handler ──────────────────────────────────────────────────────────
window.addEventListener('resize', () => {
    if (!mapData) return;
    computeCellSize();
    resizeCanvas();
    renderGrid();
});


// ── Pie Chart / Calculate Modal ─────────────────────────────────────────────
const chartModal       = document.getElementById('chart-modal');
const pieCanvas        = document.getElementById('pie-chart-canvas');
const pieCtx           = pieCanvas.getContext('2d');
const pieLegend        = document.getElementById('pie-chart-legend');
const chartLoadingText = document.getElementById('chart-loading-text');
const btnCalculate     = document.getElementById('btn-calculate');
const closeChartBtn    = document.getElementById('close-chart-btn');

btnCalculate.addEventListener('click', () => {
    chartModal.classList.remove('hidden');
    openPieChart();
});

closeChartBtn.addEventListener('click', () => {
    chartModal.classList.add('hidden');
});

chartModal.addEventListener('click', (e) => {
    if (e.target === chartModal) chartModal.classList.add('hidden');
});

async function openPieChart() {
    // Clear previous chart
    pieCtx.clearRect(0, 0, pieCanvas.width, pieCanvas.height);
    pieLegend.innerHTML = '';
    chartLoadingText.style.display = 'block';

    // Ensure batch predictions exist
    if (!batchPredictions) {
        await doBatchPredict();
    }
    chartLoadingText.style.display = 'none';

    if (!batchPredictions) {
        pieLegend.innerHTML = '<p style="color:red;">Prediction failed. Is the model loaded?</p>';
        return;
    }

    // Count cover types
    const counts = {};
    for (const name of CLASS_NAMES_LIST) counts[name] = 0;
    let total = 0;
    for (const key of Object.keys(batchPredictions)) {
        const pred = batchPredictions[key];
        if (pred && pred.cover_name) {
            counts[pred.cover_name] = (counts[pred.cover_name] || 0) + 1;
            total++;
        }
    }

    if (total === 0) {
        pieLegend.innerHTML = '<p style="color:#888;">No land cells predicted.</p>';
        return;
    }

    // Build slices sorted by count (desc)
    const slices = CLASS_NAMES_LIST
        .map((name, i) => ({ name, count: counts[name], color: CLASS_COLORS_LIST[i] }))
        .filter(s => s.count > 0)
        .sort((a, b) => b.count - a.count);

    drawAnimatedPieChart(slices, total);
    buildPieLegend(slices, total);
}


function drawAnimatedPieChart(slices, total) {
    const W = pieCanvas.width;
    const H = pieCanvas.height;
    const cx = W / 2;
    const cy = H / 2;
    const R = Math.min(W, H) / 2 - 20;         // outer radius
    const innerR = R * 0.52;                     // donut hole
    const animDuration = 800;                    // ms
    let start = null;

    function frame(ts) {
        if (!start) start = ts;
        const elapsed = ts - start;
        const progress = Math.min(1, elapsed / animDuration);
        // Ease-out cubic
        const t = 1 - Math.pow(1 - progress, 3);
        const sweepEnd = t * Math.PI * 2;

        pieCtx.clearRect(0, 0, W, H);

        let angle = -Math.PI / 2;
        for (const slice of slices) {
            const sliceAngle = (slice.count / total) * Math.PI * 2;
            const drawAngle = Math.min(sliceAngle, Math.max(0, sweepEnd - (angle + Math.PI / 2)));

            if (drawAngle <= 0) { angle += sliceAngle; continue; }

            pieCtx.beginPath();
            pieCtx.moveTo(
                cx + innerR * Math.cos(angle),
                cy + innerR * Math.sin(angle)
            );
            pieCtx.arc(cx, cy, R, angle, angle + drawAngle);
            pieCtx.arc(cx, cy, innerR, angle + drawAngle, angle, true);
            pieCtx.closePath();
            pieCtx.fillStyle = slice.color;
            pieCtx.fill();

            // Slice border
            pieCtx.strokeStyle = 'rgba(10, 13, 20, 0.5)';
            pieCtx.lineWidth = 1.5;
            pieCtx.stroke();

            angle += sliceAngle;
        }

        // Center label
        pieCtx.fillStyle = '#e2e8f0';
        pieCtx.font = '700 22px Inter, sans-serif';
        pieCtx.textAlign = 'center';
        pieCtx.textBaseline = 'middle';
        pieCtx.fillText(total.toLocaleString(), cx, cy - 8);
        pieCtx.font = '400 11px Inter, sans-serif';
        pieCtx.fillStyle = '#94a3b8';
        pieCtx.fillText('Land Cells', cx, cy + 12);

        // Percentage labels (only after animation completes)
        if (progress >= 1) {
            angle = -Math.PI / 2;
            for (const slice of slices) {
                const sliceAngle = (slice.count / total) * Math.PI * 2;
                const pct = ((slice.count / total) * 100).toFixed(1);
                if (parseFloat(pct) >= 4) {
                    const mid = angle + sliceAngle / 2;
                    const labelR = (R + innerR) / 2;
                    const lx = cx + labelR * Math.cos(mid);
                    const ly = cy + labelR * Math.sin(mid);
                    pieCtx.fillStyle = '#fff';
                    pieCtx.font = '600 11px Inter, sans-serif';
                    pieCtx.textAlign = 'center';
                    pieCtx.textBaseline = 'middle';
                    pieCtx.fillText(`${pct}%`, lx, ly);
                }
                angle += sliceAngle;
            }
        }

        if (progress < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}


function buildPieLegend(slices, total) {
    pieLegend.innerHTML = '';
    for (const slice of slices) {
        const pct = ((slice.count / total) * 100).toFixed(1);
        const item = document.createElement('div');
        item.style.cssText = 'display:flex;align-items:center;gap:6px;font-size:0.82rem;color:#cbd5e1;';
        item.innerHTML = `
            <span style="width:12px;height:12px;border-radius:3px;background:${slice.color};flex-shrink:0;"></span>
            <span>${slice.name}</span>
            <span style="color:#64748b;font-weight:600;">${pct}%</span>
        `;
        pieLegend.appendChild(item);
    }
}
