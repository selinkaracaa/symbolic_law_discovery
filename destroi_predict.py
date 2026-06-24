"""
DeSTrOI Predictor — standalone inference script.

Takes raw (x1, x2, y) data points and returns operator probability scores
using the pretrained DeSTrOI weights.

Usage:
    from destroi_predict import destroi_predict
    scores, present, absent = destroi_predict(x1_array, x2_array, y_array)
    # scores: dict {operator_name: probability}
"""

import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DESTORI_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Symbolic-Prediction-master")
ENCODER_PATH  = os.path.join(DESTORI_DIR, "weights", "superr_encoder_2.h5")
DECODER_TMPL  = os.path.join(DESTORI_DIR, "weights", "decoder_2_{}.h5")

sys.path.insert(0, DESTORI_DIR)

import numpy as np

OPERATOR_LIST      = ['add', 'mul', 'inv', 'sqrt', 'log', 'sin']
LOW_RESOLUTION     = (50, 50)
VAR_RANGE_LOW      = -10
VAR_RANGE_HIGH     = 10
INTERPOLATION_ITER = 20


def _points_to_image(x1s, x2s, ys):
    """Convert scattered data points into a low-res 50×50 grid image (vectorized)."""
    rx, ry = LOW_RESOLUTION
    low, high = VAR_RANGE_LOW, VAR_RANGE_HIGH
    # vectorized bin assignment
    ix = ((x1s - low) / (high - low) * rx).astype(int)
    iy = ((x2s - low) / (high - low) * ry).astype(int)
    mask = (ix >= 0) & (ix < rx) & (iy >= 0) & (iy < ry)
    ix, iy, ys_m = ix[mask], iy[mask], ys[mask]
    img   = np.zeros((rx, ry))
    count = np.zeros((rx, ry))
    np.add.at(img,   (ix, iy), ys_m)
    np.add.at(count, (ix, iy), 1)
    norm = np.divide(img, count, out=np.zeros_like(img), where=count != 0)
    # fill empty pixels using scipy nearest-neighbor interpolation
    from scipy.ndimage import distance_transform_edt
    empty = count == 0
    if empty.any():
        _, idx = distance_transform_edt(empty, return_indices=True)
        norm[empty] = norm[idx[0][empty], idx[1][empty]]
    return norm


def destroi_predict(x1s, x2s, ys, threshold=0.5, verbose=True):
    """
    Run DeSTrOI on data points and return operator probability scores.

    Parameters
    ----------
    x1s, x2s : array-like, shape (N,)  — input variable values
    ys        : array-like, shape (N,)  — output values
    threshold : float — operators with score >= threshold are flagged as present
    verbose   : bool  — print scores

    Returns
    -------
    scores  : dict {operator: float probability}
    present : list of operators predicted as present
    absent  : list of operators predicted as absent
    """
    from common.channel import add_channel, remove_channel
    from model.super_resolution import EDSR
    from model.CNN import CNN

    x1s = np.array(x1s, dtype=float)
    x2s = np.array(x2s, dtype=float)
    ys  = np.array(ys,  dtype=float)

    # Step 1 — build naive low-res image
    lr_img   = _points_to_image(x1s, x2s, ys)        # (50, 50)
    lr_batch = add_channel(lr_img[np.newaxis])         # (1, 50, 50, 1)

    # Step 2 — super-resolution encoder → high-res image
    encoder = EDSR()
    encoder.load_weights(ENCODER_PATH)
    hr_batch = encoder.predict(lr_batch, verbose=0)   # (1, 200, 200, 1)
    hr_img   = remove_channel(hr_batch)               # (1, 200, 200)

    # Step 3 — CNN decoder → operator scores (one decoder per operator)
    decoder = CNN()
    scores  = {}
    for i, op in enumerate(OPERATOR_LIST):
        decoder.load_weights(DECODER_TMPL.format(i))
        inp  = add_channel(hr_img)                    # (1, 200, 200, 1)
        prob = float(np.array(decoder.predict_on_batch(inp)).flatten()[0])
        scores[op] = prob

    present = [op for op, p in scores.items() if p >= threshold]
    absent  = [op for op, p in scores.items() if p <  threshold]

    if verbose:
        print("\n  DeSTrOI operator predictions:")
        for op, p in scores.items():
            bar  = "█" * int(p * 20)
            flag = "✓ PRESENT" if p >= threshold else "✗ absent"
            print(f"    {op:6s}  {p:.3f}  {bar:<20}  {flag}")
        print(f"\n  Present : {present}")
        print(f"  Absent  : {absent}")

    return scores, present, absent


if __name__ == "__main__":
    print("\n" + "─"*52)
    print("  DeSTrOI test: sin(x1) * sqrt(|x2|)")
    print("─"*52)
    rng = np.random.default_rng(42)
    x1  = rng.uniform(-10, 10, 2000)
    x2  = rng.uniform(0,   10, 2000)
    y   = np.sin(x1) * np.sqrt(x2) + rng.normal(0, 0.01, 2000)
    destroi_predict(x1, x2, y)
