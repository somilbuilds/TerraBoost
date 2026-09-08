# Cover Type Explorer

An interactive web app that visualizes a trained XGBoost multi-class classifier as a procedural terrain grid. Every cell is colored by a machine learning model's live prediction (Spruce/Fir, Lodgepole Pine, etc.) based on a full 54-feature input vector matching the UCI Covertype dataset.

## The "It's all just numbers" Concept
The terrain visualization is just a rendering of numerical features. Water and roads are generated procedurally as constraints, and the rest of the map attributes (elevation, aspect, slope, hydrology distance, wilderness zones) are synthesized and scaled to match real-world dataset properties. 

When you click on any land cell, you see the exact 54-feature vector generated for that spot along with the XGBoost model's class prediction and probability scores.

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**
   The training script pulls the dataset, trains an XGBoost classifier, and saves it.
   ```bash
   python train_model.py
   ```
   *(Wait roughly ~1 minute. It will print accuracy metrics and save `model.json`)*

3. **Start the Web Server**
   ```bash
   python -m uvicorn app:app --host 0.0.0.0 --port 8000
   ```

4. **Explore**
   - Open your browser to `http://localhost:8000/`.
   - Click "New Map" to generate procedural terrains.
   - Click any colored cell on the map to see its feature vector.
   - Click the legend items at the bottom to highlight exactly which sections the classifier predicted as that cover type.

## Placeholder Images
If you want tree images to appear in the side panel during inspection, place these files inside the `static/images/` directory:
- `spruce_fir.png`
- `lodgepole_pine.png`
- `ponderosa_pine.png`
- `cottonwood_willow.png`
- `aspen.png`
- `douglas_fir.png`
- `krummholz.png`
