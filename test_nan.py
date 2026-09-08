import numpy as np
import xgboost as xgb
from map_generator import generate_map, GRID_W, GRID_H, generate_elevation_field, generate_water, generate_road, compute_distance_field, multi_octave_noise, compute_slope_aspect, compute_hillshade, generate_zones, ELEVATION_MIN

seed = 2
rng = __import__('random').Random(seed)
w, h = GRID_W, GRID_H

elevation = generate_elevation_field(w, h, seed)
water_cells, depth_map = generate_water(w, h, seed)
road_cells = generate_road(w, h, seed)
road_cells -= water_cells

dist_to_water = compute_distance_field(w, h, water_cells) if water_cells else np.full((h, w), fill_value=6000.0, dtype=np.float32)
dist_to_road = compute_distance_field(w, h, road_cells) if road_cells else np.full((h, w), fill_value=6000.0, dtype=np.float32)

wilderness_zones = generate_zones(w, h, 4, seed + 3000)
soil_zone_map = generate_zones(w, h, 15, seed + 4000)
soil_rng = __import__('random').Random(seed + 5000)
zone_to_soil = {i: soil_rng.randint(0, 39) for i in range(15)}
fire_seed = seed + 6000

land_cells = []
for r in range(h):
    for c in range(w):
        if (r, c) not in water_cells and (r, c) not in road_cells:
            elev = float(elevation[r, c])
            slope, aspect = compute_slope_aspect(elevation, r, c, h, w)
            h_dist_hydro = float(dist_to_water[r, c]) * 30.0
            
            min_wd = dist_to_water[r, c] if water_cells else 0.0
            v_dist_hydro = float(elev - ELEVATION_MIN) * (min_wd / max(1, h + w)) * 0.1 if water_cells else 0.0
            
            h_dist_road = float(dist_to_road[r, c]) * 30.0
            h_dist_fire = multi_octave_noise(c, r, fire_seed, octaves=2, scale=20.0) * 6000.0
            hs_9am = compute_hillshade(slope, aspect, azimuth_deg=135, altitude_deg=45)
            hs_noon = compute_hillshade(slope, aspect, azimuth_deg=180, altitude_deg=60)
            hs_3pm = compute_hillshade(slope, aspect, azimuth_deg=225, altitude_deg=45)
            
            wa = [0, 0, 0, 0]
            wa[wilderness_zones[r, c]] = 1
            st = [0] * 40
            soil_idx = zone_to_soil[soil_zone_map[r, c]]
            st[soil_idx] = 1
            
            features = [
                round(elev, 1), round(aspect, 1), round(slope, 1),
                round(h_dist_hydro, 1), round(v_dist_hydro, 1), round(h_dist_road, 1),
                hs_9am, hs_noon, hs_3pm, round(h_dist_fire, 1),
            ] + wa + st
            
            # check for NaN
            fv = np.array(features, dtype=np.float32)
            if np.isnan(fv).any():
                print(f"NaN found at r={r}, c={c}")
                print("features:", features)
                import sys; sys.exit(1)
            land_cells.append(features)

X = np.array(land_cells, dtype=np.float32)
print("X shape:", X.shape)
model = xgb.XGBClassifier()
model.load_model('model.json')
preds = model.predict(X)
print("preds:", preds)
print("preds has nan:", np.isnan(preds).any())
