import json, re
import joblib, numpy as np

with open('app/models/model.json') as f:
    data = json.load(f)

trees = data['learner']['gradient_booster']['model']['trees']

# base_score is stored as a string like '[5.0140452E0]'
bs_raw = data['learner']['learner_model_param']['base_score']
base_score = float(re.findall(r'[\d.E+\-]+', str(bs_raw))[0])
print(f'base_score: {base_score}')
print(f'Trees: {len(trees)}')

# Load model and get expected prediction
model = joblib.load('app/models/best_model.pkl')
dummy = np.ones((1, 52), dtype=np.float32) * 0.5

# ── Method A: base_score + sum of leaf values ─────────────────────────────
def walk_trees(features, trees, base_score, add_base=True):
    total = base_score if add_base else 0.0
    for tree in trees:
        lc = tree['left_children']
        rc = tree['right_children']
        si = tree['split_indices']
        sc = tree['split_conditions']
        bw = tree['base_weights']
        dl = tree['default_left']
        node = 0
        while lc[node] != -1:  # -1 = leaf
            fv = float(features[si[node]])
            if fv < sc[node]:
                node = lc[node]
            else:
                node = rc[node]
        total += bw[node]
    return total

expected = float(model.predict(dummy)[0])
print(f'\nExpected (model.predict): {expected:.6f}  price=${np.expm1(expected):.2f}')

pred_with_base    = walk_trees(dummy[0], trees, base_score, add_base=True)
pred_without_base = walk_trees(dummy[0], trees, base_score, add_base=False)
print(f'Walk WITH base_score:    {pred_with_base:.6f}  price=${np.expm1(pred_with_base):.2f}  diff={abs(pred_with_base-expected):.6f}')
print(f'Walk WITHOUT base_score: {pred_without_base:.6f}  price=${np.expm1(pred_without_base):.2f}  diff={abs(pred_without_base-expected):.6f}')

# ── Test with a few more inputs ────────────────────────────────────────────
print('\n--- Multiple test cases ---')
for seed in [42, 7, 99]:
    np.random.seed(seed)
    x = np.random.rand(1, 52).astype(np.float32)
    exp = float(model.predict(x)[0])
    got = walk_trees(x[0], trees, base_score, add_base=True)
    print(f'seed={seed}: expected={exp:.4f}  got={got:.4f}  diff={abs(exp-got):.6f}  {"PASS" if abs(exp-got)<0.001 else "FAIL"}')
