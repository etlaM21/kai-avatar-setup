"""balaclava_texture.py - print a person's whole head (face, hair, ears, neck) from one
frontal photo onto the balaclava texture, so the MetaHuman seems to "wear" that face.

The balaclava was made from the face mesh, so it shares the MetaHuman face UVs.

How it works
  1. Render B (EXR, UV as colour) says, for every pixel of the calibration camera, which
     texel of the head it shows. Inverting it gives: texel -> calibration-render pixel.
  2. A thin-plate-spline fitted on the 468 landmarks (calibration render -> photo) maps
     calibration-render pixels to photo pixels. Inside the face it follows the landmarks
     exactly; outside (hair, ears, neck) it continues smoothly.
  3. Texel -> render pixel -> photo pixel, sample the photo. Texels the front camera never
     sees (back of the head) or photo background are filled from their surroundings.

Usage:
  python balaclava_texture.py photo.jpg landmark_uv.json render_B.exr out_dir [--size 2048]
Outputs:
  balaclava_BC.png     the texture for the balaclava (base colour)
  visible_mask.png     where the photo was used directly (white) vs filled (black)
  photo_lms.png        detected landmarks, to check the capture
"""
import os, sys, argparse
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import json
import cv2, numpy as np, mediapipe as mp
from scipy.interpolate import RBFInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "face_landmarker.task")
SEG_MODEL_PATH = os.path.join(HERE, "selfie_segmenter.tflite")


# ---------------------------------------------------------------- detection ----------
def detect_landmarks(img):
    """468 face landmarks in photo pixels. Uses the Tasks API if face_landmarker.task is
    next to this script, otherwise the legacy face_mesh solution (mediapipe <= 0.10.14)."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if os.path.exists(MODEL_PATH):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE, num_faces=1)
        with vision.FaceLandmarker.create_from_options(opts) as lm:
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks: raise SystemExit("No face found in photo")
        pts = res.face_landmarks[0]
    else:
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True,
                                             max_num_faces=1) as fm:
            res = fm.process(rgb)
        if not res.multi_face_landmarks: raise SystemExit("No face found in photo")
        pts = res.multi_face_landmarks[0].landmark
    return np.array([[p.x * w, p.y * h] for p in pts[:468]], np.float64)


def person_mask(img):
    """Foreground (person) probability 0..1, or None if no segmenter is available.
    Tasks API: put selfie_segmenter.tflite next to this script."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if os.path.exists(SEG_MODEL_PATH):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=SEG_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE, output_confidence_masks=True)
        with vision.ImageSegmenter.create_from_options(opts) as seg:
            res = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        m = res.confidence_masks[-1].numpy_view()        # last mask = person
        return np.squeeze(m).astype(np.float32)
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "selfie_segmentation"):
        with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0) as seg:
            return seg.process(rgb).segmentation_mask.astype(np.float32)
    return None


# ---------------------------------------------------------------- helpers ------------
def read_uv_exr(path):
    """Read render B as float32 BGRA (B, G=V, R=U, A=mask).
    Tries OpenCV first, then the OpenEXR package (pip install OpenEXR), which also
    handles Blender's multilayer EXRs (channel names like 'ViewLayer.Combined.R')."""
    if not os.path.isfile(path):
        raise SystemExit(f"render_B not found: {os.path.abspath(path)}")
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except cv2.error:
        img = None                                   # OpenCV built/configured without EXR
    if img is not None and img.ndim == 3 and img.shape[2] == 4:
        return img.astype(np.float32)
    try:
        import OpenEXR
    except ImportError:
        raise SystemExit("OpenCV could not read the EXR with an alpha channel. "
                         "Install the OpenEXR package:  pip install OpenEXR")
    with OpenEXR.File(path, separate_channels=True) as f:
        for part in f.parts:
            ch = part.channels
            def find(c):
                for name in ch:
                    if name == c or name.endswith("." + c):
                        return np.asarray(ch[name].pixels, np.float32)
                return None
            r, g, a = find("R"), find("G"), find("A")
            if r is not None and g is not None:
                if a is None:
                    raise SystemExit("render_B has no alpha channel. In Blender: Film -> "
                                     "Transparent, and save as RGBA OpenEXR.")
                b = find("B")
                return np.dstack([b if b is not None else np.zeros_like(r), g, r, a])
        names = [n for p in f.parts for n in p.channels]
    raise SystemExit(f"render_B: no R/G channels found, channels are: {names}")


def pull_push_fill(img, weight, levels=12):
    """Fill pixels with weight 0 from their surroundings (smooth, any hole size)."""
    w = weight.astype(np.float32)[..., None]
    pyr = [(img * w, w)]
    for _ in range(levels):
        c, ww = pyr[-1]
        if min(c.shape[:2]) <= 2: break
        c2 = cv2.resize(c, (c.shape[1] // 2, c.shape[0] // 2), interpolation=cv2.INTER_AREA)
        w2 = cv2.resize(ww, (ww.shape[1] // 2, ww.shape[0] // 2), interpolation=cv2.INTER_AREA)
        if c2.ndim == 2: c2 = c2[..., None]
        if w2.ndim == 2: w2 = w2[..., None]
        pyr.append((c2, w2))
    col, ww = pyr[-1]
    est = col / np.maximum(ww, 1e-6)
    for c, wl in reversed(pyr[:-1]):
        up = cv2.resize(est, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_LINEAR)
        if up.ndim == 2: up = up[..., None]
        known = c / np.maximum(wl, 1e-6)
        a = np.clip(wl, 0, 1)
        est = known * a + up * (1 - a)
    return est


def hair_fill(src, known, depth, margin):
    """Fill everything outside the visible area (back of the head, neck back) by mirroring
    the visible band next to it: each hidden texel copies the texel at the same distance on
    the other side of the nearest visible edge (folded back and forth within `depth`).
    Keeps the real hair texture at its real scale instead of smearing an average colour.
    src: colour with interior holes already filled; known: 0/1 visible texels."""
    from scipy.ndimage import distance_transform_edt
    S = src.shape[0]
    hidden = known < 0.5
    d, (iy, ix) = distance_transform_edt(hidden, return_indices=True)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    dy, dx = iy - yy, ix - xx                        # towards the nearest visible texel
    n = np.maximum(d, 1e-6)
    t = np.mod(d, 2 * depth)
    refl = np.where(t < depth, t, 2 * depth - t)      # triangle wave: fold back and forth
    step = margin + refl
    map_x = (ix + dx / n * step).astype(np.float32)
    map_y = (iy + dy / n * step).astype(np.float32)
    fill = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return fill


FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
             400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
             54, 103, 67, 109]


def hair_patches(photo, lms, fg, uv_scale, block, n=400, seed=7):
    """Sample hair blocks from the photo: person pixels outside the face oval and above
    the chin (top and sides of the head). Scaled to texel size. None if too little hair."""
    img = cv2.resize(photo, None, fx=uv_scale, fy=uv_scale,
                     interpolation=cv2.INTER_AREA if uv_scale < 1 else cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    face = np.zeros(photo.shape[:2], np.uint8)
    cv2.fillPoly(face, [lms[FACE_OVAL].round().astype(np.int32)], 1)
    face = cv2.dilate(face, np.ones((15, 15), np.uint8))
    hair = (face == 0)
    hair[int(lms[152][1]):] = False                        # nothing below the chin
    if fg is not None: hair &= fg > 0.7
    hair = cv2.resize(hair.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    # a block is usable if it lies (almost) completely on hair
    cover = cv2.blur(hair.astype(np.float32), (block, block))
    ys, xs = np.nonzero(cover > 0.97)
    ok = (ys >= block // 2) & (ys < h - block // 2) & (xs >= block // 2) & (xs < w - block // 2)
    ys, xs = ys[ok], xs[ok]
    if len(ys) < 50:
        return None
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ys), min(3 * n, len(ys)), replace=False)
    r = block // 2
    cands = np.stack([img[y - r:y - r + block, x - r:x - r + block] for y, x in
                      zip(ys[pick], xs[pick])]).astype(np.float32)
    # drop blocks with a lot of skin showing through (ears, forehead under the fringe):
    # compare against the skin colour inside the face oval
    face_in = cv2.erode(face, np.ones((41, 41), np.uint8)) > 0
    lab = lambda a: cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    skin = np.median(lab(photo)[face_in], 0)
    c_lab = lab(cands.reshape(-1, block, 3)).reshape(cands.shape)
    skin_frac = (np.linalg.norm(c_lab - skin, axis=-1) < 14).mean((1, 2))
    keep = np.nonzero(skin_frac < 0.15)[0]
    if len(keep) < 50:
        keep = np.argsort(skin_frac)[:max(50, len(skin_frac) // 2)]
    return cands[keep[:n]]


def quilt(cands, size, overlap=None, seed=7):
    """Image quilting (Efros & Freeman 2001, without the seam cut): cover a size x size
    canvas with blocks that match their neighbours in the overlaps."""
    rng = np.random.default_rng(seed)
    block = cands.shape[1]
    overlap = overlap or block // 4
    step = block - overlap
    n = size // step + 2
    out = np.zeros((n * step + overlap, n * step + overlap, 3), np.float32)
    ramp = np.linspace(0, 1, overlap, dtype=np.float32)
    for ty in range(n):
        for tx in range(n):
            y, x = ty * step, tx * step
            cost = np.zeros(len(cands), np.float32)
            if tx: cost += ((cands[:, :, :overlap] - out[y:y + block, x:x + overlap]) ** 2).sum((1, 2, 3))
            if ty: cost += ((cands[:, :overlap] - out[y:y + overlap, x:x + block]) ** 2).sum((1, 2, 3))
            pick = cands[rng.choice(np.argsort(cost)[:15])]
            a = np.ones((block, block), np.float32)
            if tx: a[:, :overlap] *= ramp[None, :]
            if ty: a[:overlap, :] *= ramp[:, None]
            region = out[y:y + block, x:x + block]
            out[y:y + block, x:x + block] = region * (1 - a[..., None]) + pick * a[..., None]
    return out[:size, :size]


def texel_to_render(uvimg, size):
    """Invert render B: for each texel, the calibration-render pixel that shows it."""
    H, W = uvimg.shape[:2]
    valid = uvimg[..., 3] > 0.5
    # skip pixels right at silhouettes and at the eye/mouth openings: there the photo
    # pixel is often background or eyeball, not the surface the texel belongs to
    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ys, xs = np.nonzero(valid)
    u, v = uvimg[ys, xs, 2], uvimg[ys, xs, 1]
    tx = np.clip((u * size).astype(int), 0, size - 1)
    ty = np.clip(((1.0 - v) * size).astype(int), 0, size - 1)   # image convention
    sx = np.zeros((size, size), np.float64); sy = np.zeros_like(sx); n = np.zeros_like(sx)
    np.add.at(sx, (ty, tx), xs); np.add.at(sy, (ty, tx), ys); np.add.at(n, (ty, tx), 1)
    seen = n > 0
    rx = np.where(seen, sx / np.maximum(n, 1), 0).astype(np.float32)
    ry = np.where(seen, sy / np.maximum(n, 1), 0).astype(np.float32)
    # close small gaps (where one render pixel covers several texels), but only locally,
    # so coordinates never get averaged across far-apart UV islands
    w = seen.astype(np.float32)
    for _ in range(4):
        k = (5, 5)
        bw = cv2.blur(w, k)
        brx, bry = cv2.blur(rx * w, k), cv2.blur(ry * w, k)
        grow = (w == 0) & (bw > 0.2)
        rx[grow] = brx[grow] / bw[grow]; ry[grow] = bry[grow] / bw[grow]
        w[grow] = 1
    return rx, ry, w > 0


# ---------------------------------------------------------------- main ---------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo"); ap.add_argument("table"); ap.add_argument("render_b")
    ap.add_argument("out")
    ap.add_argument("--size", type=int, default=2048, help="output texture size")
    ap.add_argument("--smooth", type=float, default=2.0,
                    help="thin-plate-spline smoothing (higher = less landmark jitter)")
    ap.add_argument("--no-seg", action="store_true", help="don't remove photo background")
    ap.add_argument("--hair-depth", type=float, default=0.04,
                    help="thickness of the visible band that is mirrored onto the back of "
                         "the head, as a fraction of the texture (smaller = less face skin)")
    ap.add_argument("--hair-margin", type=float, default=0.008,
                    help="how far inside the visible edge the mirroring starts")
    ap.add_argument("--no-hair-synth", action="store_true",
                    help="don't synthesise hair for the back of the head, only mirror")
    ap.add_argument("--smear", action="store_true",
                    help="old behaviour: fill the back of the head with smoothed colour")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = a.size

    table = json.load(open(a.table))["landmarks"]
    render_lms = np.array([table[str(i)]["px_render"] for i in range(468)], np.float64)

    photo = cv2.imread(a.photo)
    ph, pw = photo.shape[:2]
    photo_lms = detect_landmarks(photo)
    dbg = photo.copy()
    for p in photo_lms: cv2.circle(dbg, (int(p[0]), int(p[1])), 1, (0, 255, 0), -1)
    cv2.imwrite(os.path.join(a.out, "photo_lms.png"), dbg)

    uvimg = read_uv_exr(a.render_b)
    RH, RW = uvimg.shape[:2]

    # 1. texel -> calibration render pixel
    rx, ry, seen = texel_to_render(uvimg, S)

    # 2. calibration render pixel -> photo pixel (thin-plate spline on the landmarks),
    #    evaluated on a coarse grid and upsampled
    tps = RBFInterpolator(render_lms, photo_lms, kernel="thin_plate_spline",
                          smoothing=a.smooth)
    step = 8
    gy, gx = np.mgrid[0:RH + step:step, 0:RW + step:step].astype(np.float64)
    grid = tps(np.stack([gx.ravel(), gy.ravel()], 1)).astype(np.float32)
    map_x = grid[:, 0].reshape(gx.shape); map_y = grid[:, 1].reshape(gx.shape)

    # 3. texel -> photo pixel
    px = cv2.remap(map_x, rx / step, ry / step, cv2.INTER_LINEAR)
    py = cv2.remap(map_y, rx / step, ry / step, cv2.INTER_LINEAR)
    inside = (px >= 0) & (px < pw - 1) & (py >= 0) & (py < ph - 1)
    colour = cv2.remap(photo, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    weight = (seen & inside).astype(np.float32)
    fg_photo = None
    if not a.no_seg:
        fg = person_mask(photo)
        fg_photo = fg
        if fg is None:
            print("note: no segmenter available - photo background is not removed")
        else:
            fg_t = cv2.remap(fg, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            weight *= np.clip((fg_t - 0.3) / 0.4, 0, 1)
    # 4. fill what the front camera / photo can't see (back of head, background).
    #    The fill only uses texels that are reliably known; the photo is then faded in
    #    with a soft edge that stays inside the known area.
    known = (weight > 0.5).astype(np.uint8)
    # drop isolated specks (grazing-angle texels); they would act as false edges
    n_cc, lab, st, _ = cv2.connectedComponentsWithStats(known, connectivity=8)
    big = np.zeros(n_cc, bool); big[1:] = st[1:, cv2.CC_STAT_AREA] > 0.001 * S * S
    known = big[lab].astype(np.float32)
    filled = pull_push_fill(colour.astype(np.float32), known)
    if not a.smear:
        # holes enclosed by the visible area (e.g. under the chin) stay smoothly filled;
        # everything else hidden gets the mirrored band from the nearest visible edge
        # mirror only from the large regions (head+hair, neck/shoulders), not from the
        # small ear islands - mirrored ears look like rings
        large = np.zeros(n_cc, bool); large[1:] = st[1:, cv2.CC_STAT_AREA] > 0.02 * S * S
        from scipy.ndimage import binary_fill_holes
        inner = binary_fill_holes(large[lab]).astype(np.float32)   # enclosed holes = interior
        interior = colour.astype(np.float32) * known[..., None] + filled * (1 - known[..., None])
        # Mirror only from the hair band above the brows and from the neck/shoulders below
        # the chin - never from cheeks or temples, which would put skin on the back of the
        # head. Hidden texels between brow and chin height take whichever is nearer.
        brow_y = np.mean([table[str(i)]["uv"][1] for i in (105, 334, 9)]) * S
        chin_y = table["152"]["uv"][1] * S
        rows = np.arange(S, dtype=np.float32)[:, None]
        src_mask = inner * ((rows < brow_y - a.hair_margin * S) | (rows > chin_y)).astype(np.float32)
        mirrored = hair_fill(interior, src_mask, depth=a.hair_depth * S, margin=a.hair_margin * S)
        back = mirrored
        # back of the head above the chin: synthesise hair from a sample of the photo
        uv_l, uv_r = np.array(table["127"]["uv"]) * S, np.array(table["356"]["uv"]) * S
        uv_scale = np.linalg.norm(uv_r - uv_l) / np.linalg.norm(photo_lms[356] - photo_lms[127])
        block = max(32, int(0.05 * S))
        cands = None if a.no_hair_synth else hair_patches(photo, photo_lms, fg_photo,
                                                            uv_scale, block)
        if cands is not None:
            synth = quilt(cands, S)
            # match synthesised hair to the brightness of the visible hair band
            band = (src_mask > 0.5) & (rows < brow_y)
            if band.sum() > 500:                           # brightness only, keep the hue
                gain = interior[band].mean() / max(float(synth.mean()), 1.0)
                synth *= np.clip(gain, 0.6, 1.6)
            up = np.clip((chin_y + 0.03 * S - rows) / (0.08 * S), 0, 1)  # 1 above, 0 below chin
            back = synth * up[..., None] + mirrored * (1 - up[..., None])
            cv2.imwrite(os.path.join(a.out, "hair_sample.png"),
                        np.clip(np.hstack(list(cands[:12])), 0, 255).astype(np.uint8))
        elif not a.no_hair_synth:
            print("note: not enough visible hair in the photo - back of head is mirrored")
        out_w = cv2.GaussianBlur((1 - inner) * (1 - known), (0, 0), 0.004 * S)
        filled = filled * (1 - out_w[..., None]) + back * out_w[..., None]
    soft = cv2.GaussianBlur(cv2.erode(known, np.ones((7, 7), np.uint8)), (0, 0), 1.0)
    weight = soft * weight
    out = colour.astype(np.float32) * weight[..., None] + filled * (1 - weight[..., None])
    out = np.clip(out, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(a.out, "balaclava_BC.png"), out)
    cv2.imwrite(os.path.join(a.out, "visible_mask.png"), (weight * 255).astype(np.uint8))
    print(f"done: {S}px texture, photo used on {100 * (weight > 0.5).mean():.1f}% of texels")


if __name__ == "__main__":
    main()
