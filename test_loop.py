import numpy as np
import xgboost as xgb
from map_generator import generate_map

try:
    model = xgb.XGBClassifier()
    model.load_model('model.json')
    
    for seed in range(500):
        m = generate_map(seed, model)
        # Just generating should crash if there's a NaN
except Exception as e:
    import traceback
    traceback.print_exc()
    print("FAILED ON SEED:", seed)
else:
    print("ALL SEEDS PASSED")
