"""
convert_model_to_onnx.py
------------------------
One-time script: converts best_model.pkl -> model.onnx
Run this ONCE locally, then commit model.onnx to git.

Usage:
    python -m pip install onnxmltools skl2onnx onnxruntime
    python convert_model_to_onnx.py
"""
import os
import sys
import json
import joblib
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# -- Paths ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "app", "models")

model_path         = os.path.join(MODELS_DIR, "best_model.pkl")
feature_names_path = os.path.join(MODELS_DIR, "feature_names.json")
output_onnx        = os.path.join(MODELS_DIR, "model.onnx")

print(f"Loading model from: {model_path}")
model = joblib.load(model_path)

with open(feature_names_path) as f:
    meta = json.load(f)
feature_names = meta["feature_names"]
n_features = len(feature_names)
print(f"Features  : {n_features}")
print(f"Model type: {type(model).__name__}")

# -- Strip named features from the booster (onnxmltools requires 'f0..fn') -
booster = model.get_booster() if hasattr(model, "get_booster") else model
original_feature_names = booster.feature_names
original_feature_types = booster.feature_types
booster.feature_names = None
booster.feature_types = None
print(f"Stripped feature names from booster for ONNX conversion")

# -- Convert via onnxmltools -----------------------------------------------
try:
    from onnxmltools import convert_xgboost
    from onnxmltools.utils import save_model as save_onnx
    from skl2onnx.common.data_types import FloatTensorType

    print("Converting with onnxmltools...")
    onnx_model = convert_xgboost(
        booster,
        initial_types=[("input", FloatTensorType([None, n_features]))]
    )
    save_onnx(onnx_model, output_onnx)
    print(f"Saved: {output_onnx}")

except Exception as e:
    print(f"onnxmltools failed: {e}")
    # Restore feature names before exiting
    booster.feature_names = original_feature_names
    booster.feature_types = original_feature_types
    sys.exit(1)

# Restore for safety
booster.feature_names = original_feature_names
booster.feature_types = original_feature_types

# -- Verify output matches original model -----------------------------------
print("\nVerifying ONNX output...")
import onnxruntime as ort

# Use a realistic dummy input (not all zeros, as model may behave oddly)
dummy = np.ones((1, n_features), dtype=np.float32) * 0.5

orig_pred  = float(model.predict(dummy)[0])
orig_price = float(np.expm1(orig_pred))

sess       = ort.InferenceSession(output_onnx)
input_name = sess.get_inputs()[0].name
onnx_out   = sess.run(None, {input_name: dummy})
onnx_pred  = float(onnx_out[0].flatten()[0])
onnx_price = float(np.expm1(onnx_pred))

print(f"  Original : log={orig_pred:.6f}  price=${orig_price:.2f}")
print(f"  ONNX     : log={onnx_pred:.6f}  price=${onnx_price:.2f}")
diff = abs(orig_price - onnx_price)
status = "PASS" if diff < 0.01 else "WARN"
print(f"  [{status}] Difference: ${diff:.4f}")

size_mb = os.path.getsize(output_onnx) / 1024 / 1024
print(f"\nmodel.onnx written ({size_mb:.2f} MB)")
print("Next: commit app/models/model.onnx to git")
