/**
 * index.js — Cover Type Explorer Frontend
 * Renders the grid map, handles cell interaction, legend highlighting,
 * and side-panel feature display.
 */

// ── Constants ──────────────────────────────────────────────────────────────
const CELL_SIZE = 12;
const HIGHLIGHT_BORDER = 2;

// Tree image filenames (to be placed in /static/images/)
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
let mapData = null;          // Full response from /api/generate_map
let cellGrid = [];           // 2D lookup: cellGrid[r][c]
let selectedCell = null;     // Currently selected {r, c}
let highlightedType = null;  // Currently highlighted cover type name
let canvas, ctx;

// ── DOM refs ───────────────────────────────────────────────────────────────
const btnNewMap       = document.getElementById('btn-new-map');
const seedDisplay     = document.getElementById('seed-display');
const canvasEl        = document.getElementById('grid-canvas');
const loadingOverlay  = document.getElementById('loading-overlay');
const tooltip         = document.getElementById('tooltip');
const legendItems     = document.getElementById('legend-items');
const panelEmpty      = document.getElementById('panel-empty');
const panelContent    = document.getElementById('panel-content');
const panelSimple     = document.getElementById('panel-simple');

// ── Init ───────────────────────────────────────────────────────────────────
canvas = canvasEl;
ctx = canvas.getContext('2d');

btnNewMap.addEventListener('click', () => fetchMap());
canvas.addEventListener('click', onCanvasClick);
canvas.addEventListener('mousemove', onCanvasHover);
canvas.addEventListener('mouseleave', () => tooltip.classList.remove('visible'));

// Load first map on page load
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
        resizeCanvas();
        renderGrid();
        buildLegend();
        showPanel('empty');
        selectedCell = null;
        highlightedType = null;
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
function resizeCanvas() {
    const w = mapData.grid_width * CELL_SIZE;
    const h = mapData.grid_height * CELL_SIZE;
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
}

function renderGrid() {
    if (!mapData) return;
    const { grid_width, grid_height } = mapData;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    for (let r = 0; r < grid_height; r++) {
        for (let c = 0; c < grid_width; c++) {
            const cell = cellGrid[r][c];
            if (!cell) continue;

            const x = c * CELL_SIZE;
            const y = r * CELL_SIZE;
            
            // Draw cell
            ctx.fillStyle = cell.color;
            ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);

            // Dim cells not matching highlighted type
            if (highlightedType !== null && cell.type === 'land' && cell.cover_name !== highlightedType) {
                ctx.fillStyle = 'rgba(10, 13, 20, 0.55)';
                ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
            }

            // Glow outline for highlighted type
            if (highlightedType !== null && cell.type === 'land' && cell.cover_name === highlightedType) {
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
                ctx.lineWidth = 1;
                ctx.strokeRect(x + 0.5, y + 0.5, CELL_SIZE - 1, CELL_SIZE - 1);
            }

            // Selected cell outline
            if (selectedCell && selectedCell.r === r && selectedCell.c === c) {
                ctx.strokeStyle = '#FBBF24';
                ctx.lineWidth = 2;
                ctx.strokeRect(x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2);
            }

            // Subtle grid line
            ctx.strokeStyle = 'rgba(255,255,255,0.03)';
            ctx.lineWidth = 0.5;
            ctx.strokeRect(x, y, CELL_SIZE, CELL_SIZE);
        }
    }
}


// ── Legend ──────────────────────────────────────────────────────────────────
function buildLegend() {
    // Legend is now statically defined in HTML
}

function toggleHighlight(coverName) {
    // Highlight feature removed since cells no longer represent tree covers initially
}


// ── Canvas interaction ─────────────────────────────────────────────────────
function getCellAt(event) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const c = Math.floor(x / CELL_SIZE);
    const r = Math.floor(y / CELL_SIZE);
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
    
    if (cell.type === 'water') {
        showWaterPanel(cell);
    } else if (cell.type === 'road') {
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
    if (cell.type === 'land') {
        text = `${cell.cover_name} · (${cell.r}, ${cell.c})`;
    } else if (cell.type === 'water') {
        text = `Water · (${cell.r}, ${cell.c})`;
    } else {
        text = `Road · (${cell.r}, ${cell.c})`;
    }
    
    tooltip.textContent = text;
    tooltip.classList.add('visible');
    
    // Position tooltip
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

function showLandPanel(cell) {
    showPanel('land');
    
    // Header badge
    const badge = document.getElementById('cell-type-badge');
    badge.innerHTML = `<span class="badge-dot" style="background:${cell.color}"></span>Terrain Cell`;
    badge.style.background = hexToRgba(cell.color, 0.15);
    badge.style.color = lightenColor(cell.color, 0.4);
    document.getElementById('cell-coords').textContent = `(${cell.r}, ${cell.c})`;
    
    // Default prediction UI blocks while loading
    const predDiv = document.getElementById('predicted-class');
    const probContainer = document.getElementById('probability-bars');
    const treeSection = document.getElementById('tree-image-section');
    const treeImg = document.getElementById('tree-image');
    const treeCap = document.getElementById('tree-image-caption');
    
    predDiv.innerHTML = `<span style="color:#666">Analyzing 54 features...</span>`;
    probContainer.innerHTML = '';
    treeSection.classList.add('hidden');

    // Fetch prediction from server
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ features: cell.feature_vector })
    })
    .then(r => r.json())
    .then(data => {
        if(data.error) {
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
        
        // Prediction
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
    .catch(err => {
        predDiv.innerHTML = `<span style="color:red">Server error during prediction</span>`;
    });
    
    // Continuous features table
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
}

function showWaterPanel(cell) {
    showPanel('simple');
    document.getElementById('simple-icon').style.background = 'rgba(20, 112, 160, 0.2)';
    document.getElementById('simple-icon').textContent = '🌊';
    document.getElementById('simple-label').textContent = 'Water';
    document.getElementById('simple-label').style.color = '#22D3EE';
    document.getElementById('simple-desc').textContent =
        `Depth: ${cell.depth ?? '—'} · This cell is water — not classified by the model. Water bodies are generated procedurally and excluded from prediction.`;
}

function showRoadPanel(cell) {
    showPanel('simple');
    document.getElementById('simple-icon').style.background = 'rgba(107, 107, 107, 0.2)';
    document.getElementById('simple-icon').textContent = '🛤️';
    document.getElementById('simple-label').textContent = 'Road';
    document.getElementById('simple-label').style.color = '#9CA3AF';
    document.getElementById('simple-desc').textContent =
        'This cell is a road — not classified by the model. Roads are generated as straight lines across the terrain.';
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
        .catch(err => {
            document.getElementById('modal-metrics').innerHTML = `<p style="color:red;">Failed to load metrics API.</p>`;
        });
});

document.getElementById('close-modal-btn').addEventListener('click', () => {
    modal.classList.add('hidden');
});

modal.addEventListener('click', (e) => {
    if (e.target === modal) {
        modal.classList.add('hidden');
    }
});
