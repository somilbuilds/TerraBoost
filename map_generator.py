"""
map_generator.py — Procedural map generation for Cover Type Explorer
Generates a 60×60 grid with terrain noise, optional water/road features,
builds 54-feature vectors per cell, and runs batch XGBoost predictions.
"""

import math
import random
import numpy as np
from collections import deque

# ── Constants ────────────────────────────────────────────────────────────────
GRID_W, GRID_H = 60, 60
ELEVATION_MIN, ELEVATION_MAX = 1859, 3858

CLASS_NAMES = {
    0: "Spruce/Fir",
    1: "Lodgepole Pine",
    2: "Ponderosa Pine",
    3: "Cottonwood/Willow",
    4: "Aspen",
    5: "Douglas-fir",
    6: "Krummholz",
}

# 7 distinct colors for cover types (earth/forest palette)
CLASS_COLORS = {
    0: "#2D5F3E",   # Spruce/Fir — deep forest green
    1: "#5B8C5A",   # Lodgepole Pine — medium green
    2: "#C4A35A",   # Ponderosa Pine — warm golden
    3: "#7ABF7E",   # Cottonwood/Willow — light green
    4: "#E8C547",   # Aspen — bright gold/yellow
    5: "#3E7A5E",   # Douglas-fir — teal green
    6: "#8B6F47",   # Krummholz — brown
}

WATER_COLOR_DEEP = "#2B4C6F"
WATER_COLOR_SHALLOW = "#3D6287"
ROAD_COLOR = "#6B6B6B"

FEATURE_NAMES = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
] + [f"Wilderness_Area_{i}" for i in range(1, 5)] + [
    f"Soil_Type_{i}" for i in range(1, 41)
]


# ── Seeded hash noise ───────────────────────────────────────────────────────

def _hash2d(x, y, seed):
    """Simple integer hash for 2D coordinates."""
    h = seed
    h ^= x * 374761393
    h ^= y * 668265263
    h = (h * 1274126177) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 1103515245) & 0xFFFFFFFF
    h ^= h >> 16
    return (h & 0xFFFFFFFF) / 0xFFFFFFFF   # [0, 1]


def _smooth_noise(x, y, seed):
    """Bilinearly interpolated hash noise."""
    ix, iy = int(math.floor(x)), int(math.floor(y))
    fx, fy = x - ix, y - iy
    # Smoothstep
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    n00 = _hash2d(ix, iy, seed)
    n10 = _hash2d(ix + 1, iy, seed)
    n01 = _hash2d(ix, iy + 1, seed)
    n11 = _hash2d(ix + 1, iy + 1, seed)
    nx0 = n00 * (1 - fx) + n10 * fx
    nx1 = n01 * (1 - fx) + n11 * fx
    return nx0 * (1 - fy) + nx1 * fy


def multi_octave_noise(x, y, seed, octaves=4, persistence=0.5, scale=8.0):
    """Multi-octave smoothed hash noise → [0, 1]."""
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_val = 0.0
    for i in range(octaves):
        total += _smooth_noise(x * frequency / scale, y * frequency / scale, seed + i * 31) * amplitude
        max_val += amplitude
        amplitude *= persistence
        frequency *= 2.0
    return total / max_val


# ── Elevation field ──────────────────────────────────────────────────────────

def generate_elevation_field(w, h, seed):
    """Returns a 2D numpy array of elevation values in [ELEVATION_MIN, ELEVATION_MAX]."""
    field = np.zeros((h, w), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            # Increased scale (40.0) for larger, chunkier terrain features
            # Applied a slight curve for wider valleys and sharper peaks
            n = multi_octave_noise(c, r, seed, octaves=4, persistence=0.45, scale=40.0)
            n = n ** 1.3
            field[r, c] = ELEVATION_MIN + n * (ELEVATION_MAX - ELEVATION_MIN)
    return field


# ── Water generation ─────────────────────────────────────────────────────────

def generate_water(w, h, seed):
    """Generates a guaranteed water body (river or lake) for realistic hydrology features."""
    rng = random.Random(seed + 1000)
    water_cells = set()
    depth_map = {}

    # Force a river starting from one edge and meandering across
    edges = [(0, rng.randint(10, w-10)), (h-1, rng.randint(10, w-10)), 
             (rng.randint(10, h-10), 0), (rng.randint(10, h-10), w-1)]
    start = rng.choice(edges)
    
    cr, cc = start
    # Aim for the center, then wander
    steps = rng.randint(80, 150)
    path = [(cr, cc)]
    
    # Base direction towards the center
    dr = 1 if cr == 0 else (-1 if cr == h-1 else rng.choice([-1, 0, 1]))
    dc = 1 if cc == 0 else (-1 if cc == w-1 else rng.choice([-1, 0, 1]))

    for _ in range(steps):
        # Heavy momentum to keep the river flowing, plus random wandering
        if rng.random() < 0.4:
            dr = dr + rng.choice([-1, 0, 1])
            dc = dc + rng.choice([-1, 0, 1])
        dr = max(-1, min(1, dr))
        dc = max(-1, min(1, dc))
        if dr == 0 and dc == 0:
            dr = rng.choice([-1, 1])
            
        cr = max(0, min(h - 1, cr + dr))
        cc = max(0, min(w - 1, cc + dc))
        path.append((cr, cc))

    # Thicken path to create a body (0 means 1 pixel wide, 1 means 3 pixels wide)
    thickness = rng.randint(0, 1)
    for pr, pc in path:
        for dr in range(-thickness, thickness + 1):
            for dc in range(-thickness, thickness + 1):
                if dr * dr + dc * dc <= thickness * thickness:
                    nr, nc = pr + dr, pc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        water_cells.add((nr, nc))

    # Compute depth as distance from edge of water body
    if water_cells:
        max_depth = 0
        for r, c in water_cells:
            # depth = min distance to a non-water neighbor
            min_d = float('inf')
            for dr in range(-thickness, thickness + 1):
                for dc in range(-thickness, thickness + 1):
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in water_cells or not (0 <= nr < h and 0 <= nc < w):
                        d = math.sqrt((r - nr) ** 2 + (c - nc) ** 2)
                        min_d = min(min_d, d)
            if math.isinf(min_d):
                min_d = thickness + 1.0
            depth_map[(r, c)] = min_d
            max_depth = max(max_depth, min_d)
        # Normalize depth 0-1
        if max_depth > 0:
            for k in depth_map:
                depth_map[k] /= max_depth

    return water_cells, depth_map


# ── Road generation ──────────────────────────────────────────────────────────

def generate_road(w, h, seed):
    """With ~45% probability, draws a road line (Bresenham) between two edge points."""
    rng = random.Random(seed + 2000)
    road_cells = set()

    if rng.random() > 0.45:
        return road_cells

    # Pick two random points on different edges
    edges = ['top', 'bottom', 'left', 'right']
    e1, e2 = rng.sample(edges, 2)

    def edge_point(edge):
        if edge == 'top':
            return (0, rng.randint(5, w - 6))
        elif edge == 'bottom':
            return (h - 1, rng.randint(5, w - 6))
        elif edge == 'left':
            return (rng.randint(5, h - 6), 0)
        else:
            return (rng.randint(5, h - 6), w - 1)

    r0, c0 = edge_point(e1)
    r1, c1 = edge_point(e2)

    # Bresenham's line
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        if 0 <= r < h and 0 <= c < w:
            road_cells.add((r, c))
        if r == r1 and c == c1:
            break
        e2_val = 2 * err
        if e2_val > -dc:
            err -= dc
            r += sr
        if e2_val < dr:
            err += dr
            c += sc

    return road_cells


# ── Distance computation (BFS) ──────────────────────────────────────────────

def compute_distance_field(w, h, source_cells):
    """BFS distance from every cell to the nearest source cell. Returns 2D numpy array."""
    dist = np.full((h, w), fill_value=9999.0, dtype=np.float32)
    queue = deque()
    for r, c in source_cells:
        dist[r, c] = 0
        queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] > dist[r, c] + 1:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


# ── Slope, Aspect, Hillshade ────────────────────────────────────────────────

def compute_slope_aspect(elevation, r, c, h, w):
    """Compute slope (degrees) and aspect (degrees, 0-360) from local gradient."""
    # Central differences
    r0 = max(0, r - 1)
    r1 = min(h - 1, r + 1)
    c0 = max(0, c - 1)
    c1 = min(w - 1, c + 1)

    dz_dc = (elevation[r, c1] - elevation[r, c0]) / max(1, c1 - c0)
    dz_dr = (elevation[r1, c] - elevation[r0, c]) / max(1, r1 - r0)

    # Scale: each cell ~ 30m
    cell_size = 30.0
    dz_dc /= cell_size
    dz_dr /= cell_size

    slope_rad = math.atan(math.sqrt(dz_dc ** 2 + dz_dr ** 2))
    slope_deg = math.degrees(slope_rad)

    aspect_rad = math.atan2(-dz_dr, dz_dc)
    aspect_deg = math.degrees(aspect_rad)
    if aspect_deg < 0:
        aspect_deg += 360

    return round(slope_deg, 1), round(aspect_deg, 1)


def compute_hillshade(slope_deg, aspect_deg, azimuth_deg, altitude_deg=45):
    """Compute hillshade value (0-255) for given sun position."""
    slope_rad = math.radians(slope_deg)
    aspect_rad = math.radians(aspect_deg)
    azimuth_rad = math.radians(azimuth_deg)
    altitude_rad = math.radians(altitude_deg)

    hs = (
        math.sin(altitude_rad) * math.cos(slope_rad)
        + math.cos(altitude_rad) * math.sin(slope_rad)
        * math.cos(azimuth_rad - aspect_rad)
    )
    return max(0, min(255, int(hs * 255)))


# ── Wilderness & Soil zones ─────────────────────────────────────────────────

def generate_zones(w, h, n_zones, seed):
    """Generate Voronoi-like zones using random seed points."""
    rng = random.Random(seed)
    centers = [(rng.randint(0, h - 1), rng.randint(0, w - 1)) for _ in range(n_zones)]
    zone_map = np.zeros((h, w), dtype=np.int32)
    for r in range(h):
        for c in range(w):
            min_d = float('inf')
            best = 0
            for i, (cr, cc) in enumerate(centers):
                d = (r - cr) ** 2 + (c - cc) ** 2
                if d < min_d:
                    min_d = d
                    best = i
            zone_map[r, c] = best
    return zone_map


# ── Main map generation ─────────────────────────────────────────────────────

def generate_map(seed, model=None):
    """
    Generate a complete map with feature vectors and predictions.
    
    Args:
        seed: Random seed for reproducible generation.
        model: Loaded XGBoost model (if None, predictions will be skipped).
    
    Returns:
        dict with grid dimensions, seed, and list of cell data.
    """
    rng = random.Random(seed)
    w, h = GRID_W, GRID_H

    # 1. Elevation field
    elevation = generate_elevation_field(w, h, seed)

    # 2. Water
    water_cells, depth_map = generate_water(w, h, seed)

    # 3. Road
    road_cells = generate_road(w, h, seed)

    # Remove road cells that overlap water
    road_cells -= water_cells

    # 4. Distance fields
    if water_cells:
        dist_to_water = compute_distance_field(w, h, water_cells)
    else:
        dist_to_water = np.full((h, w), fill_value=6000.0, dtype=np.float32)

    if road_cells:
        dist_to_road = compute_distance_field(w, h, road_cells)
    else:
        dist_to_road = np.full((h, w), fill_value=6000.0, dtype=np.float32)

    # 5. Wilderness zones (4 zones)
    wilderness_zones = generate_zones(w, h, 4, seed + 3000)

    # 6. Soil type zones (use more Voronoi centers → 15 zones, mapped to 40 types)
    soil_zone_map = generate_zones(w, h, 15, seed + 4000)
    # Map each of the 15 zones to a soil type (1-40)
    soil_rng = random.Random(seed + 5000)
    zone_to_soil = {i: soil_rng.randint(0, 39) for i in range(15)}

    # 7. Fire points noise
    fire_seed = seed + 6000

    # 8. Build feature vectors and colors for land cells
    land_cells = []
    cells = []

    for r in range(h):
        for c in range(w):
            if (r, c) in water_cells:
                depth = depth_map.get((r, c), 0.5)
                t = depth
                dr_c, dg_c, db_c = int(WATER_COLOR_DEEP[1:3], 16), int(WATER_COLOR_DEEP[3:5], 16), int(WATER_COLOR_DEEP[5:7], 16)
                sr_c, sg_c, sb_c = int(WATER_COLOR_SHALLOW[1:3], 16), int(WATER_COLOR_SHALLOW[3:5], 16), int(WATER_COLOR_SHALLOW[5:7], 16)
                cr_v = int(sr_c + t * (dr_c - sr_c))
                cg_v = int(sg_c + t * (dg_c - sg_c))
                cb_v = int(sb_c + t * (db_c - sb_c))
                color = f"#{cr_v:02x}{cg_v:02x}{cb_v:02x}"
                cells.append({
                    "r": r, "c": c, "type": "water",
                    "depth": round(depth, 2), "color": color
                })
            elif (r, c) in road_cells:
                cells.append({
                    "r": r, "c": c, "type": "road", "color": ROAD_COLOR
                })
            else:
                elev = float(elevation[r, c])
                slope, aspect = compute_slope_aspect(elevation, r, c, h, w)

                h_dist_hydro = float(dist_to_water[r, c]) * 30.0
                min_wd = float(dist_to_water[r, c]) if water_cells else 0.0
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

                # Topographic color mapping based on elevation
                t = (elev - ELEVATION_MIN) / (ELEVATION_MAX - ELEVATION_MIN)
                t = max(0, min(1, t))
                
                # Organic, sharp topography colors
                if t < 0.2:
                    # Forest Green
                    r_c = 40 + (t / 0.2) * (15)
                    g_c = 90 + (t / 0.2) * (20)
                    b_c = 50 + (t / 0.2) * (15)
                elif t < 0.45:
                    # Lush Yellow-Green
                    t_sub = (t - 0.2) / 0.25
                    r_c = 55 + t_sub * (65)
                    g_c = 110 + t_sub * (40)
                    b_c = 65 - t_sub * (25)
                elif t < 0.7:
                    # Tan / Brown
                    t_sub = (t - 0.45) / 0.25
                    r_c = 120 + t_sub * (40)
                    g_c = 150 - t_sub * (40)
                    b_c = 40 + t_sub * (10)
                elif t < 0.85:
                    # Dark Rock
                    t_sub = (t - 0.7) / 0.15
                    r_c = 160 - t_sub * (50)
                    g_c = 110 - t_sub * (10)
                    b_c = 50 + t_sub * (50)
                else:
                    # Snow peaks
                    t_sub = (t - 0.85) / 0.15
                    r_c = 110 + t_sub * (130)
                    g_c = 100 + t_sub * (140)
                    b_c = 100 + t_sub * (140)
                
                # Apply hillshade for highly defined 3D effect (sharp peaks/valleys)
                shade_factor = hs_noon / 255.0
                shade_mapped = 0.3 + (shade_factor * 1.1)  # Higher contrast
                
                fr = min(255, max(0, int(r_c * shade_mapped)))
                fg = min(255, max(0, int(g_c * shade_mapped)))
                fb = min(255, max(0, int(b_c * shade_mapped)))
                hex_color = f"#{fr:02x}{fg:02x}{fb:02x}"

                cell_data = {
                    "r": r, "c": c, "type": "land",
                    "features": {name: val for name, val in zip(FEATURE_NAMES, features)},
                    "feature_vector": features,
                    "color": hex_color,
                    "elevation_score": t
                }
                cells.append(cell_data)

    return {
        "grid_width": w,
        "grid_height": h,
        "seed": seed,
        "cells": cells,
    }
