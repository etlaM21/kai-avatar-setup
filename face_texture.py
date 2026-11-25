"""face_texture.py - paint a person's eyes, brows and mouth from a frontal photo into
the MetaHuman face texture.

Usage:
  python face_texture.py photo.jpg landmark_uv.json T_Head_BC.png out_dir
         [--margin 0.025] [--feather 0.012] [--no-brows] [--match 0.0]

Outputs (in out_dir):
  face_BC.png      base texture with the photo regions blended in (same size as base)
  face_mask.png    the blend mask in UV space (for checking / for UE)
  photo_lms.png    the photo with detected landmarks (for checking capture quality)
"""
import os, sys, json, argparse
import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from scipy.spatial import Delaunay

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
FLC = vision.FaceLandmarksConnections

def pts(conns): return sorted({i for c in conns for i in (c.start, c.end)})
LEFT_EYE, RIGHT_EYE = pts(FLC.FACE_LANDMARKS_LEFT_EYE), pts(FLC.FACE_LANDMARKS_RIGHT_EYE)
LEFT_BROW, RIGHT_BROW = pts(FLC.FACE_LANDMARKS_LEFT_EYEBROW), pts(FLC.FACE_LANDMARKS_RIGHT_EYEBROW)
LIPS = pts(FLC.FACE_LANDMARKS_LIPS)

# Ordered rings around the openings. A frontal photo only shows a line here, but in the
# MetaHuman UV the eyelid rims and the inner lips are a whole band -> keep the base texture.
RING_EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RING_EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
RING_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
             415, 310, 311, 312, 13, 82, 81, 80, 191]
INNER_LIPS = RING_LIPS
OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              409, 270, 269, 267, 0, 37, 39, 40, 185]


def detect(img):
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        res = landmarker.detect(mp_image)
    if not res.face_landmarks:
        raise SystemExit("No face found in photo")
    h, w = img.shape[:2]
    return np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0][:468]],
                    np.float32)


def region_mask(uv_px, groups, size, margin, feather):
    """Union of dilated convex hulls in UV space, softened at the edge."""
    m = np.zeros((size, size), np.uint8)
    for g in groups:
        cv2.fillConvexPoly(m, cv2.convexHull(uv_px[g].astype(np.int32)), 255)
    r = max(1, int(margin * size))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    s = max(1, int(feather * size)) | 1
    return cv2.GaussianBlur(m, (0, 0), s / 2).astype(np.float32) / 255.0


def openings_mask(uv_px, size, grow, feather, whole_lips=False):
    m = np.zeros((size, size), np.uint8)
    for ring in (RING_EYE_R, RING_EYE_L, OUTER_LIPS if whole_lips else RING_LIPS):
        cv2.fillPoly(m, [uv_px[ring].round().astype(np.int32)], 255)
    r = max(1, int(grow * size))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    s = max(1, int(feather * size))
    return cv2.GaussianBlur(m, (0, 0), s / 2).astype(np.float32) / 255.0


def warp_photo_to_uv(photo, src_px, uv_px, size):
    """Piecewise-affine warp of the photo into UV space (Delaunay on UV positions)."""
    tri = Delaunay(uv_px)
    out = np.zeros((size, size, 3), np.float32)
    cover = np.zeros((size, size), np.float32)
    inner = set(INNER_LIPS) | set(LEFT_EYE) | set(RIGHT_EYE)
    skipped = 0
    cross2d = lambda a, b: a[0] * b[1] - a[1] * b[0]
    area = lambda p: abs(cross2d(p[1] - p[0], p[2] - p[0])) / 2 + 1e-6
    ratios = np.array([area(uv_px[t]) / area(src_px[t]) for t in tri.simplices])
    max_ratio = 6 * np.median(ratios)            # strongly stretched slivers -> keep base
    for t, r in zip(tri.simplices, ratios):
        if r > max_ratio:
            skipped += 1; continue
        if all(i in inner for i in t):          # would bridge eye or mouth opening
            skipped += 1; continue
        d, s = uv_px[t], src_px[t]
        o_uv = np.sign(cross2d(d[1] - d[0], d[2] - d[0]))
        o_ph = np.sign(cross2d(s[1] - s[0], s[2] - s[0]))
        if o_uv != o_ph or o_uv == 0:           # folded triangle -> skip
            skipped += 1; continue
        x, y, w, h = cv2.boundingRect(d.astype(np.float32))
        x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w + 1, size), min(y + h + 1, size)
        if x1 <= x0 or y1 <= y0: continue
        M = cv2.getAffineTransform(s.astype(np.float32), (d - [x0, y0]).astype(np.float32))
        patch = cv2.warpAffine(photo, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        tm = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(tm, (d - [x0, y0]).round().astype(np.int32), 1, cv2.LINE_8)
        sel = (tm > 0) & (cover[y0:y1, x0:x1] == 0)
        out[y0:y1, x0:x1][sel] = patch[sel]
        cover[y0:y1, x0:x1][sel] = 1
    return out, cover, skipped


def match_colour(src, ref, weight, strength):
    """Lab mean/std transfer of src towards ref inside weight; strength 0..1."""
    if strength <= 0: return src
    sl = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    rl = cv2.cvtColor(ref.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    w = weight > 0.5
    for c in range(3):
        ms, ss = sl[..., c][w].mean(), sl[..., c][w].std() + 1e-6
        mr, sr = rl[..., c][w].mean(), rl[..., c][w].std()
        k = np.clip(sr / ss, 0.7, 1.4)
        sl[..., c] = (1 - strength) * sl[..., c] + strength * ((sl[..., c] - ms) * k + mr)
    return cv2.cvtColor(np.clip(sl, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo"); ap.add_argument("table"); ap.add_argument("base"); ap.add_argument("out")
    ap.add_argument("--margin", type=float, default=0.025, help="mask growth, fraction of texture")
    ap.add_argument("--feather", type=float, default=0.012, help="mask softness, fraction of texture")
    ap.add_argument("--no-brows", action="store_true")
    ap.add_argument("--match", type=float, default=0.0,
                    help="0 = keep the person's colours, 1 = fully match base skin tone")
    ap.add_argument("--open-grow", type=float, default=0.012,
                    help="extra growth of the eye/mouth opening area kept from the base")
    ap.add_argument("--lips", choices=["photo", "base"], default="base",
                    help="'base' keeps MetaHuman's lips, recoloured to the photo's lip colour")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    base = cv2.imread(a.base, cv2.IMREAD_UNCHANGED)
    alpha = base[..., 3:] if base.shape[2] == 4 else None
    base_bgr = base[..., :3].astype(np.float32)
    size = base.shape[0]
    assert base.shape[1] == size, "expects a square texture"

    table = json.load(open(a.table))["landmarks"]
    uv_px = np.array([table[str(i)]["uv"] for i in range(468)], np.float32) * size

    photo = cv2.imread(a.photo)
    src_px = detect(photo)
    dbg = photo.copy()
    for p in src_px: cv2.circle(dbg, tuple(int(c) for c in p), 1, (0, 255, 0), -1)
    cv2.imwrite(os.path.join(a.out, "photo_lms.png"), dbg)

    groups = [LEFT_EYE + ([] if a.no_brows else LEFT_BROW),
              RIGHT_EYE + ([] if a.no_brows else RIGHT_BROW), LIPS]
    mask = region_mask(uv_px, groups, size, a.margin, a.feather)

    warped, cover, skipped = warp_photo_to_uv(photo.astype(np.float32), src_px, uv_px, size)
    # where no triangle was warped (skipped slivers, outside the mesh) use the base texture,
    # and fade softly into those spots so no dark seams appear
    warped = np.where(cover[..., None] > 0, warped, base_bgr)
    mask *= cv2.GaussianBlur(cv2.erode(cover, np.ones((5, 5), np.uint8)), (0, 0), 6)
    # Lid rims and the rolled-in part of the lips are foreshortened in a frontal photo and
    # would be stretched. There we keep the base texture, shifted to the photo's colour.
    band = openings_mask(uv_px, size, a.open_grow, 0.006, whole_lips=(a.lips == "base"))
    region = mask.copy()
    mask = region * (1.0 - band)
    lab = lambda img: cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8),
                                   cv2.COLOR_BGR2LAB).astype(np.float32)
    ring = (cv2.dilate((band > 0.5).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0) \
           & (band < 0.5) & (mask > 0.5)
    if ring.sum() > 50:
        delta = lab(warped)[ring].mean(0) - lab(base_bgr)[ring].mean(0)
        band_fill = cv2.cvtColor(np.clip(lab(base_bgr) + delta, 0, 255).astype(np.uint8),
                                 cv2.COLOR_LAB2BGR).astype(np.float32)
    else:
        band_fill = base_bgr
    band_w = region * band
    warped = match_colour(warped, base_bgr, mask, a.match)

    out = (base_bgr * (1 - mask - band_w)[..., None] + warped * mask[..., None]
           + band_fill * band_w[..., None])
    out = np.clip(out, 0, 255).astype(np.uint8)
    if alpha is not None: out = np.dstack([out, alpha])
    cv2.imwrite(os.path.join(a.out, "face_BC.png"), out)
    cv2.imwrite(os.path.join(a.out, "face_mask.png"), (mask * 255).astype(np.uint8))
    print(f"done: {size}px texture, triangles skipped: {skipped}, "
          f"mask covers {100 * (mask > 0.5).mean():.1f}% of texture")


if __name__ == "__main__":
    main()
