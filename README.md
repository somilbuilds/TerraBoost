# 🌲 Cover Type Explorer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost 3.4](https://img.shields.io/badge/XGBoost-3.4.1-FF6F00?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Accuracy](https://img.shields.io/badge/Model_Accuracy-92.1%25-4ADE80?style=flat-square)](https://archive.ics.uci.edu/ml/datasets/Covertype)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

An interactive, high-performance web application that bridges procedural 3D terrain synthesis with real-time machine learning. Built on the classic **UCI Forest Covertype Dataset**, this project generates 100×75 procedural cartographic maps where every land cell is represented as a 54-dimensional feature vector and classified by an **XGBoost** model into one of 7 forest cover types.

---

## 📸 Overview & Features

- 🗺️ **Procedural Map Synthesis**: Multi-octave Perlin-like hash noise generates elevation, slope, aspect, hydrology distance fields, organic lakes, and roads.
- 🌲 **7 Tree Cover Types**: Classified by XGBoost in real time based on 54 cartographic features.
- 📊 **Interactive Pie Chart Distribution**: Modal visualizer displaying the complete tree species breakdown percentage across all ~7,500 grid cells with an animated donut chart.
- ⚡ **Lazy Inference Architecture**: Fast initial map loading with on-demand cell inspection and batch background evaluation for maximum UI responsiveness.
- 🔍 **54-Feature Vector Inspector**: Click any cell to view raw cartographic features (elevation, slope, hillshades, wilderness areas, soil types).

---

## 📊 Machine Learning Model Metrics

The classifier was trained on **581,012 samples** from the UCI Covertype dataset using **XGBoost** with hyperparameter tuning.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **92.10%** |
| **Macro F1-Score** | **91.15%** |
| **Weighted F1-Score** | **92.07%** |
| **Total Test Samples** | 116,203 |

### Per-Class Performance Breakdown

| Class ID | Tree Cover Type | Precision | Recall | F1-Score | Test Samples |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | 🌲 **Spruce/Fir** | `92.41%` | `89.26%` | `0.9081` | 42,368 |
| **2** | 🌲 **Lodgepole Pine** | `91.39%` | `94.26%` | `0.9280` | 56,661 |
| **3** | 🌲 **Ponderosa Pine** | `93.88%` | `94.83%` | `0.9435` | 7,151 |
| **4** | 🌿 **Cottonwood/Willow** | `89.98%` | `86.70%` | `0.8831` | 549 |
| **5** | 🍂 **Aspen** | `92.15%` | `79.09%` | `0.8512` | 1,899 |
| **6** | 🌲 **Douglas-fir** | `91.40%` | `89.63%` | `0.9051` | 3,473 |
| **7** | 🪵 **Krummholz** | `96.74%` | `95.59%` | `0.9616` | 4,102 |

---

## 🌿 Forest Cover Types Explained

| Cover Type | Color Swatch | Elevation Range | Typical Environment |
| :--- | :---: | :--- | :--- |
| **Spruce/Fir** | `#2D5F3E` | High altitude (2,900m - 3,400m) | Cold subalpine zones with high moisture |
| **Lodgepole Pine** | `#5B8C5A` | Mid-to-high (2,500m - 3,200m) | Dense subalpine forests, moderate slopes |
| **Ponderosa Pine** | `#C4A35A` | Lower elevation (1,800m - 2,500m) | Warm, dry, low-elevation montane forests |
| **Cottonwood/Willow** | `#7ABF7E` | Low valley bottoms (< 2,200m) | Riparian zones along rivers and lakes |
| **Aspen** | `#E8C547` | Mid elevation (2,400m - 2,900m) | Rich soils, sunlit slopes and forest clearings |
| **Douglas-fir** | `#3E7A5E` | Low to mid elevation (2,100m - 2,600m) | Shaded north-facing slopes and moist valleys |
| **Krummholz** | `#8B6F47` | Extreme alpine timberline (> 3,300m) | Stunted, wind-bent trees at high exposure |

---

## 🏔️ Terrain Generation Architecture

The application generates synthetic terrain maps using procedural algorithms that closely mirror real-world spatial distributions:

1. **Elevation Map**: Synthesized using 4-octave smooth hash noise scaled across 1,859m to 3,858m.
2. **Hydrology & Roads**: Organic lakes generated using random-walk BFS clusters; single-cell wide roads generated with terrain-aware waypoint interpolation.
3. **54 Cartographic Features per Cell**:
   - **10 Continuous Variables**: Elevation, Aspect, Slope, Distances to Water/Roads/Fire Points, 9am/Noon/3pm Hillshade.
   - **4 Binary Wilderness Areas**: Voronoi partition mapping.
   - **40 Binary Soil Types**: Voronoi multi-zone cluster assignment.
4. **Universal Hillshade Shading**: Solar illumination gradient calculated from surface normals for authentic 3D terrain contrast.

---

## 💡 Key Use Cases

- **Environmental & Forestry Research**: Understand how topographic features (elevation, slope, aspect, soil type) drive species distribution in ecological zones.
- **Machine Learning Explainability**: Interactive visual verification of how non-linear models like XGBoost carve out decision boundaries in high-dimensional feature spaces.
- **Procedural Worldbuilding**: Demonstrates how dataset-accurate feature synthesis can generate plausible synthetic landscapes for simulation & gaming.

---

## 🛠️ Project Structure

```
forest_cover_type_explorer/
├── app.py              # FastAPI server serving REST endpoints & static assets
├── map_generator.py    # Procedural map, noise, terrain, and 54-feature synthesizer
├── train_model.py      # UCI dataset fetcher, XGBoost model trainer & metrics evaluator
├── model.json          # Trained XGBoost model binary (~24 MB)
├── metrics.json        # Saved accuracy, macro F1, classification report & confusion matrix
├── static/
│   ├── index.html      # Glassmorphic single-page web interface
│   ├── index.css       # Custom design system with dark mode & visual hierarchy
│   ├── index.js        # Canvas grid rendering, side panel UI & pie chart visualizer
│   └── images/         # Tree type preview graphics
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Installation
```bash
# Clone repository
git clone https://github.com/somilbuilds/TerraBoost.git
cd TerraBoost

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Web Server
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your web browser.

### 4. (Optional) Retrain Model
```bash
python train_model.py
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
