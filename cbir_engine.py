"""
cbir_engine.py - mesin pencarian CBIR topeng Bali (tanpa ketergantungan pada notebook).

Memuat model yang disimpan notebook (folder `model_cbir/`), mengekstrak fitur dari gambar unggahan
dengan pipa yang SAMA seperti Skenario 2/2b di notebook, lalu mengurutkan galeri dengan fusi cosine berbobot
dan (opsional) relevance feedback Rocchio.

Pemakaian ringkas:
    from cbir_engine import CBIRModel, decode_image
    model = CBIRModel.load("model_cbir")                    # sekali saja (di Streamlit: st.cache_resource)
    sess  = model.start(decode_image(bytes_upload))         # ekstraksi fitur kueri
    hasil = model.rank(sess, weights=None, top_k=10)        # bobot None = bobot hasil validasi
    sess  = model.feedback(sess, relevant_ids=[...])        # umpan balik, lalu rank() lagi
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

# ---- Parameter ekstraksi (harus sama dengan notebook; disimpan juga di meta.json dan diperiksa saat load) ----
RANDOM_SEED = 42
IMAGE_SIZE = (128, 128)
SEG_MAX_SIDE = 160
HSV_BINS = (16, 4, 4)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
LBP_P, LBP_R = 8, 1
LBP_GRID = (2, 2)
LBP_BINS_PER_CELL = 59
BLOCKS = ["hsv", "hog", "hu", "lbp"]
ROCCHIO_ALPHA, ROCCHIO_BETA, ROCCHIO_GAMMA = 1.0, 0.75, 0.15


# ------------------------------------------------------------------ pra-pemrosesan (Skenario 2b: tanpa kotak anotasi)
def decode_image(data: bytes) -> np.ndarray:
    """Byte berkas gambar (hasil unggahan) -> array BGR."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Berkas bukan gambar yang dapat dibaca.")
    return img


def _shrink(bgr, max_side=SEG_MAX_SIDE):
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return bgr
    return cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def _largest_contour_mask(mask):
    mask = (mask > 0).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full(mask.shape, 255, np.uint8)
    largest = max(contours, key=cv2.contourArea)
    ratio = cv2.contourArea(largest) / max(1, mask.shape[0] * mask.shape[1])
    if ratio < 0.02 or ratio > 0.98:
        return np.full(mask.shape, 255, np.uint8)
    clean = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
    return clean


def _segment_foreground(bgr):
    h, w = bgr.shape[:2]
    if h < 10 or w < 10:
        return np.full((h, w), 255, np.uint8)
    try:
        gc_mask = np.zeros((h, w), np.uint8)
        rect = (max(1, int(0.03 * w)), max(1, int(0.03 * h)), max(2, int(0.94 * w)), max(2, int(0.94 * h)))
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.setRNGSeed(RANDOM_SEED)
        cv2.grabCut(bgr, gc_mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        binary = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        clean = _largest_contour_mask(binary)
        if 0.03 <= np.mean(clean > 0) <= 0.97:
            return clean
    except Exception:
        pass
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = max(1, min(h, w) // 30)

    def border_ratio(m):
        b = np.concatenate([m[:bw].ravel(), m[-bw:].ravel(), m[:, :bw].ravel(), m[:, -bw:].ravel()])
        return float(np.mean(b > 0))

    return _largest_contour_mask(min([th, cv2.bitwise_not(th)], key=border_ratio))


def _crop_to_mask_bbox(image, mask, padding=0.08):
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return image, mask
    h, w = image.shape[:2]
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    px, py = int((x1 - x0) * padding), int((y1 - y0) * padding)
    x0, x1, y0, y1 = max(0, x0 - px), min(w, x1 + px), max(0, y0 - py), min(h, y1 + py)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return image, mask
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def _normalize_v_channel(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def preprocess(bgr: np.ndarray):
    """Pra-pemrosesan penuh tanpa kotak anotasi: segmentasi GrabCut, crop, resize 128x128, CLAHE pada kanal V."""
    small = _shrink(bgr.copy())
    mask = _segment_foreground(small)
    cropped, mask = _crop_to_mask_bbox(small, mask)
    resized = cv2.resize(cropped, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    shape_mask = (cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8) * 255
    if np.mean(shape_mask > 0) < 0.02 or np.mean(shape_mask > 0) > 0.98:
        shape_mask = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0]), 255, np.uint8)
    resized = _normalize_v_channel(resized)
    return resized, shape_mask.copy(), shape_mask


def extract_features(bgr: np.ndarray) -> dict:
    """Fitur mentah HSV (256), HOG (1764), Hu (7), LBP (236) dari satu gambar BGR."""
    img, color_mask, shape_mask = preprocess(bgr)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fgray = gray.copy()
    fgray[shape_mask == 0] = 255
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv], [0, 1, 2], color_mask, list(HSV_BINS), [0, 180, 0, 256, 0, 256]).flatten().astype(np.float32)
    hog_vec = hog(fgray, orientations=HOG_ORIENTATIONS, pixels_per_cell=HOG_PIXELS_PER_CELL,
                  cells_per_block=HOG_CELLS_PER_BLOCK, block_norm="L2-Hys", feature_vector=True).astype(np.float32)
    hu = cv2.HuMoments(cv2.moments((shape_mask > 0).astype(np.uint8) * 255)).flatten()
    hu_log = (-np.sign(hu) * np.log10(np.abs(hu) + 1e-12)).astype(np.float32)
    lbp_img = local_binary_pattern(fgray, P=LBP_P, R=LBP_R, method="nri_uniform")
    parts, (H, W) = [], lbp_img.shape
    for r in range(LBP_GRID[0]):
        for c in range(LBP_GRID[1]):
            cell = np.asarray(lbp_img[r * H // LBP_GRID[0]:(r + 1) * H // LBP_GRID[0],
                                      c * W // LBP_GRID[1]:(c + 1) * W // LBP_GRID[1]], dtype=np.int32)
            hist = np.bincount(cell.ravel(), minlength=LBP_BINS_PER_CELL)[:LBP_BINS_PER_CELL].astype(np.float32)
            parts.append(hist / max(float(hist.sum()), 1.0))
    return {"hsv": hsv_hist, "hog": hog_vec, "hu": hu_log, "lbp": np.concatenate(parts).astype(np.float32)}


# ------------------------------------------------------------------ skor dan umpan balik
def _l2_rows(m, eps=1e-12):
    m = np.asarray(m, np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), eps)


def _zscore(s):
    s = np.asarray(s, np.float32)
    return (s - s.mean(axis=-1, keepdims=True)) / np.maximum(s.std(axis=-1, keepdims=True), 1e-8)


@dataclass
class Session:
    """Keadaan satu pencarian (dibuat model.start). Aman disimpan di st.session_state."""
    qb: dict
    excluded: np.ndarray
    shown: list = field(default_factory=list)
    round: int = 0


class CBIRModel:
    def __init__(self, model_dir, meta, gallery, scenarios, thumbs_dir):
        self.model_dir, self.meta, self.gallery, self._sc, self.thumbs_dir = Path(model_dir), meta, gallery, scenarios, thumbs_dir
        self.scenario = None
        self.use_scenario(meta["default_scenario"])

    # ---- memuat ----
    @classmethod
    def load(cls, model_dir="model_cbir", scenario=None):
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        feat = meta["features"]
        mine = dict(IMAGE_SIZE=list(IMAGE_SIZE), SEG_MAX_SIDE=SEG_MAX_SIDE, HSV_BINS=list(HSV_BINS), HOG_ORIENTATIONS=HOG_ORIENTATIONS,
                    HOG_PIXELS_PER_CELL=list(HOG_PIXELS_PER_CELL), HOG_CELLS_PER_BLOCK=list(HOG_CELLS_PER_BLOCK),
                    LBP_P=LBP_P, LBP_R=LBP_R, LBP_GRID=list(LBP_GRID), LBP_BINS_PER_CELL=LBP_BINS_PER_CELL)
        diff = {k: (feat.get(k), v) for k, v in mine.items() if feat.get(k) != v}
        if diff:
            raise ValueError(f"Parameter fitur model tidak sama dengan cbir_engine.py (model, engine): {diff}")
        import pandas as pd
        gallery = pd.read_csv(model_dir / "gallery.csv")
        scenarios = {}
        for tag in meta["scenarios"]:
            z = np.load(model_dir / f"index_{tag}.npz")
            scenarios[tag] = {k: z[k] for k in z.files}
        m = cls(model_dir, meta, gallery, scenarios, model_dir / "thumbs")
        if scenario:
            m.use_scenario(scenario)
        return m

    def use_scenario(self, tag):
        if tag not in self.meta["scenarios"]:
            raise KeyError(f"Skenario '{tag}' tidak ada. Tersedia: {list(self.meta['scenarios'])}")
        self.scenario, self.info, self.z = tag, self.meta["scenarios"][tag], self._sc[tag]
        self.gn = {b: self.z[f"g_{b}"] for b in BLOCKS}

    @property
    def labels(self):
        return self.gallery["label"].tolist()

    def default_weights(self, model="w2"):
        """Bobot hasil validasi. 'w1' = Model 1, 'w2' = Model 2 (usulan), 'w2m' = Model 2b (semua blok aktif)."""
        return dict(self.info["weights"][model])

    def weights_from_ui(self, w_warna, w_bentuk, w_tekstur, model="w2"):
        """Slider pengguna (warna, bentuk, tekstur) -> bobot blok; bobot bentuk dibagi HOG/Hu menurut proporsi hasil validasi."""
        w = self.default_weights(model)
        shape_tot = w["hog"] + w["hu"]
        hog_share = (w["hog"] / shape_tot) if shape_tot > 0 else 1.0
        tot = float(w_warna + w_bentuk + w_tekstur)
        if tot <= 0:
            return w
        a, b, c = w_warna / tot, w_bentuk / tot, w_tekstur / tot
        return {"hsv": a, "hog": b * hog_share, "hu": b * (1 - hog_share), "lbp": c}

    # ---- pencarian ----
    def _normalize_query(self, raw):
        out = {}
        for b in BLOCKS:
            v = np.asarray(raw[b], np.float32).reshape(1, -1)
            if b == "hu":
                v = (v - self.z["hu_mean"]) / self.z["hu_std"]
            elif f"c_{b}" in self.z:
                v = v - self.z[f"c_{b}"]
            out[b] = _l2_rows(v)[0]
        return out

    def start(self, bgr: np.ndarray) -> Session:
        return Session(qb=self._normalize_query(extract_features(bgr)), excluded=np.zeros(len(self.gallery), bool))

    def start_from_features(self, raw: dict) -> Session:
        return Session(qb=self._normalize_query(raw), excluded=np.zeros(len(self.gallery), bool))

    def scores(self, qb, weights):
        zs = self.info["zscore"]
        s = np.zeros(len(self.gallery), np.float32)
        for b, w in weights.items():
            if w:
                sb = self.gn[b] @ qb[b]
                s += float(w) * (_zscore(sb) if zs else sb)
        return s

    def rank(self, sess: Session, weights=None, top_k=10):
        """Daftar top_k hasil (citra yang sudah dinilai pada putaran sebelumnya dikeluarkan = residual ranking)."""
        weights = weights or self.default_weights()
        sc = np.where(sess.excluded, -np.inf, self.scores(sess.qb, weights))
        top = np.argsort(-sc, kind="stable")[:top_k]
        sess.shown = top.tolist()
        rows = self.gallery.iloc[top]
        return [{"id": int(i), "label": r.label, "score": float(sc[i]), "path": r.path,
                 "thumb": str(self.thumbs_dir / r.thumb) if isinstance(r.thumb, str) else None}
                for i, r in zip(top, rows.itertuples())]

    def feedback(self, sess: Session, relevant_ids, weights=None) -> Session:
        """Rocchio (Persamaan 8 proposal): hasil yang ditandai mirip = positif, sisanya yang ditampilkan = negatif."""
        pos = np.array([i for i in sess.shown if i in set(relevant_ids)], dtype=int)
        neg = np.array([i for i in sess.shown if i not in set(relevant_ids)], dtype=int)
        qb = {}
        for b in BLOCKS:
            q = ROCCHIO_ALPHA * np.asarray(sess.qb[b], np.float32)
            if len(pos):
                q = q + ROCCHIO_BETA * self.gn[b][pos].mean(axis=0)
            if len(neg):
                q = q - ROCCHIO_GAMMA * self.gn[b][neg].mean(axis=0)
            n = float(np.linalg.norm(q))
            qb[b] = (q / n if n > 1e-12 else q).astype(np.float32)
        sess.qb = qb
        sess.excluded[sess.shown] = True
        sess.round += 1
        return sess
