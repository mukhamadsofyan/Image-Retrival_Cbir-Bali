"""Aplikasi pencarian topeng Bali (CBIR). Jalankan dari folder TuBes:  streamlit run app_streamlit.py
Logika pencarian ada di cbir_engine.py; file ini berisi tampilan, penjelasan, dan visualisasi."""
import hashlib
import inspect
import os
import random
from base64 import b64encode
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from skimage import exposure
from skimage.feature import hog, local_binary_pattern

from cbir_engine import BLOCKS, CBIRModel, decode_image, preprocess

BASE = Path(__file__).parent
TOP_K = 10
GOLD, IVORY, SAGE, TEAL, LAV, LINE = "#D2A549", "#ECE6D6", "#7C8A83", "#4E9F93", "#A99BD1", "#2B3A34"
GROUP_COLOR = {"Warna": GOLD, "Bentuk": TEAL, "Tekstur": LAV}
GROUPS = {"Warna": ["hsv"], "Bentuk": ["hog", "hu"], "Tekstur": ["lbp"]}
BLOCK_NAME = {"hsv": "Warna (HSV)", "hog": "Garis dan kontur (HOG)", "hu": "Siluet (Hu Moments)", "lbp": "Pola permukaan (LBP)"}
CLASS_COLORS = ["#D2A549", "#4E9F93", "#A99BD1", "#C0573F", "#7FB069", "#5B8DB8", "#E3B7A0", "#B7C4BC",
                "#E6D36F", "#8C6BB1", "#D77FA1", "#6FC2C0", "#9C9A6B"]
SCEN_NAMES = {"s2b_praproses_otomatis": "Potong otomatis (seperti unggahan pengguna)",
              "s2_praproses_penuh": "Potong menurut anotasi"}
DESC = {
    "Topeng Tua": "Tokoh lelaki tua dalam pertunjukan topeng. Wajahnya keriput dan sering digambarkan ompong, geraknya pelan dan penuh kebijaksanaan.",
    "Topeng Penasar": "Abdi atau punakawan yang menerjemahkan dan membawa jalan cerita. Wajahnya sederhana dan ekspresif.",
    "Topeng Sidakarya": "Topeng penutup yang dipercaya menyempurnakan sebuah upacara. Biasanya berwajah putih dengan mata kecil dan gigi besar.",
    "Topeng Dalem Arsa Wijaya": "Tokoh raja yang berwibawa, berwajah halus dan tenang.",
    "Topeng Telek": "Topeng tarian Telek yang halus dan anggun, sering dipasangkan dengan Jauk.",
    "Topeng Jauk Keras": "Topeng tari Jauk berkarakter gagah dan kasar, dengan mata melotot, kumis tebal, dan taring.",
    "Topeng Jauk Manis": "Topeng Jauk dengan wajah lebih halus dan manis, geraknya lebih lembut.",
    "Topeng Barong Ket": "Barong yang paling dikenal, berwujud makhluk mitologi mirip singa dan dipandang sebagai pelindung.",
    "Topeng Barong Macan": "Barong berwujud harimau.",
    "Topeng Barong Bangkung": "Barong berwujud babi hutan.",
}

st.set_page_config(page_title="Temu Kembali Topeng Bali", page_icon="🎭", layout="wide", initial_sidebar_state="expanded")


def _full(fn):
    return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}


FULL_BTN, FULL_PLOT = _full(st.button), _full(st.plotly_chart)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Young+Serif&family=Figtree:wght@400;500;600&display=swap');
:root{--bg:#101815;--panel:#18231F;--panel2:#1F2D28;--line:#2B3A34;--ink:#ECE6D6;--mute:#8FA097;--gold:#D2A549;--gold2:#E8C87A;--teal:#4E9F93;--lav:#A99BD1}
html,body,[class*="css"],.stApp{font-family:'Figtree',system-ui,sans-serif}
.stApp{background:radial-gradient(1100px 520px at 88% -8%,#1D2F28 0%,var(--bg) 62%)}
#MainMenu,footer{visibility:hidden}
header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:2rem;max-width:1360px}
h1,h2,h3{font-family:'Young Serif',Georgia,serif!important;font-weight:400!important;letter-spacing:-.01em}
.hero{display:grid;grid-template-columns:1.05fr 1fr;gap:2.2rem;align-items:center;margin-bottom:1.2rem}
.hero h1{font-size:clamp(2.1rem,4.4vw,3.5rem);line-height:1.06;margin:0 0 .9rem}
.hero p{color:var(--mute);font-size:1.04rem;line-height:1.65;max-width:56ch;margin:0 0 1rem}
.hero .facts{display:flex;gap:1.6rem;flex-wrap:wrap;margin-top:1.2rem}
.hero .facts b{display:block;font-family:'Young Serif',serif;font-weight:400;font-size:1.7rem;color:var(--gold2)}
.hero .facts span{color:var(--mute);font-size:.82rem}
.mosaic{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;padding:8px;border-radius:14px;
  background:conic-gradient(#0A0F0D 25%,#7C8A83 0 50%,#0A0F0D 0 75%,#ECE6D6 0) 0 0/22px 22px}
.mosaic img{width:100%;aspect-ratio:3/4;object-fit:cover;border-radius:6px;display:block;background:#0B110F}
.poleng{height:12px;border-radius:2px;margin:.4rem 0 1.4rem;background:conic-gradient(#0A0F0D 25%,#7C8A83 0 50%,#0A0F0D 0 75%,#ECE6D6 0) 0 0/24px 12px}
.sect{font-family:'Young Serif',serif;font-size:1.5rem;margin:2.2rem 0 .3rem;color:var(--ink)}
.sub{color:var(--mute);font-size:.95rem;line-height:1.6;margin:0 0 1rem;max-width:90ch}
.how{border-left:3px solid var(--gold);background:var(--panel);border-radius:0 10px 10px 0;padding:.8rem 1rem;color:#C9C4B4;font-size:.92rem;line-height:1.6;margin:.4rem 0 1rem}
.how b{color:var(--gold2)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1rem;height:100%}
.panel h4{font-family:'Young Serif',serif;font-weight:400;font-size:1.05rem;margin:0 0 .6rem;color:var(--gold2)}
.pic{width:100%;aspect-ratio:1/1;object-fit:contain;background:#0B110F;border-radius:6px;display:block}
figure{margin:0}
figcaption,.cap{color:var(--mute);font-size:.8rem;line-height:1.45;margin-top:.4rem}
figcaption b{color:var(--ink);font-weight:600;display:block;font-size:.86rem}
.five{display:grid;grid-template-columns:repeat(5,1fr);gap:.8rem}
.pal{display:flex;height:34px;border-radius:6px;overflow:hidden;margin:.5rem 0 .4rem}
.pal i{display:block;height:100%}
.mk{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.mk.top{border-color:var(--gold);box-shadow:0 0 0 1px var(--gold) inset}
.mk-img{position:relative;aspect-ratio:1/1;background:#0B110F}
.mk-img img{width:100%;height:100%;object-fit:contain;display:block}
.mk-rank{position:absolute;left:8px;top:8px;background:var(--bg);color:var(--gold2);border:1px solid var(--line);border-radius:999px;min-width:26px;height:26px;display:flex;align-items:center;justify-content:center;font-size:.8rem;font-weight:600}
.mk.top .mk-rank{background:var(--gold);color:#17120A;border-color:var(--gold)}
.mk-body{padding:.65rem .75rem .75rem}
.mk-name{font-family:'Young Serif',serif;font-size:1rem;line-height:1.2;min-height:2.4em}
.mk-bar{height:4px;background:var(--line);border-radius:4px;margin:.5rem 0 .35rem;overflow:hidden}
.mk-bar i{display:block;height:100%;background:var(--gold)}
.mk-score{color:var(--mute);font-size:.78rem;display:flex;justify-content:space-between;align-items:center;gap:.4rem}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}
.bar{display:flex;height:10px;border-radius:5px;overflow:hidden;margin:.3rem 0 .5rem;background:var(--line)}
.legend{display:flex;justify-content:space-between;color:var(--mute);font-size:.78rem}
.pill{display:inline-block;border:1px solid var(--gold);color:var(--gold2);border-radius:999px;padding:.15rem .7rem;font-size:.8rem;margin-left:.6rem;vertical-align:middle;font-family:'Figtree',sans-serif}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:.9rem;margin:.6rem 0 1rem}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1rem 1.1rem}
.kpi b{display:block;font-family:'Young Serif',serif;font-weight:400;font-size:2.1rem;color:var(--gold2);line-height:1.1}
.kpi span{color:var(--ink);font-size:.9rem;display:block;margin-top:.3rem}
.kpi small{color:var(--mute);font-size:.78rem;line-height:1.4;display:block;margin-top:.2rem}
.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:.7rem}
.step{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1rem;position:relative}
.step:not(:last-child):after{content:"";position:absolute;right:-.62rem;top:50%;width:.55rem;height:.55rem;border-top:2px solid var(--gold);border-right:2px solid var(--gold);transform:rotate(45deg) translateY(-50%)}
.step b{font-family:'Young Serif',serif;font-weight:400;display:block;color:var(--gold2);margin-bottom:.4rem}
.step span{color:#C9C4B4;font-size:.86rem;line-height:1.55}
table.t{width:100%;border-collapse:collapse;font-size:.9rem}
table.t th{text-align:left;color:var(--gold2);font-weight:600;border-bottom:1px solid var(--line);padding:.55rem .6rem}
table.t td{padding:.55rem .6rem;border-bottom:1px solid var(--line);color:#C9C4B4;vertical-align:top}
.tag{border-radius:999px;padding:.1rem .6rem;font-size:.78rem;font-weight:600;white-space:nowrap}
.ok{background:#1E3A2C;color:#8FD3A4}.no{background:#3B3220;color:#E8C87A}
.chips{display:flex;flex-wrap:wrap;gap:.45rem;margin:.4rem 0 1rem}
.chip{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:999px;padding:.3rem .8rem;font-size:.85rem}
.stTabs [data-baseweb="tab-list"]{gap:.3rem;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{height:3rem;padding:0 1.1rem;font-weight:600}
.stTabs [aria-selected="true"]{color:var(--gold2)}
div[data-testid="stCheckbox"] label p{font-size:.85rem}
div[data-testid="stFileUploader"] section{background:var(--panel);border:1.5px dashed #4A5C54;border-radius:12px;padding:1.4rem}
div[data-testid="stFileUploader"] section:hover{border-color:var(--gold)}
.stButton>button{border-radius:8px;font-weight:600}
section[data-testid="stSidebar"]{background:#0D1411;border-right:1px solid var(--line)}
@media (max-width:900px){.hero,.kpis,.flow,.five{grid-template-columns:1fr 1fr}.step:after{display:none!important}}
@media (max-width:560px){.hero,.kpis,.flow,.five{grid-template-columns:1fr}}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ model dan pembantu
@st.cache_resource(show_spinner="Memuat model ...")
def load_model(model_dir):
    return CBIRModel.load(model_dir)


def find_model_dir():
    env = os.environ.get("CBIR_MODEL_DIR")
    for p in ([Path(env)] if env else []) + [BASE / "notebook" / "model_cbir", BASE / "model_cbir"]:
        if (p / "meta.json").exists():
            return p
    return None


model_dir = find_model_dir()
if model_dir is None:
    st.error("Folder model tidak ditemukan. Jalankan notebook sampai sel 15b, lalu pastikan folder `model_cbir` ada di "
             "`notebook/` atau di samping file ini (atau isi variabel lingkungan CBIR_MODEL_DIR).")
    st.stop()
model = load_model(str(model_dir))


def idr(n):
    return f"{n:,}".replace(",", ".")


def short(label):
    return label.replace("Topeng ", "")


def uri_from_bgr(img, side=480, q=88):
    h, w = img.shape[:2]
    s = side / max(h, w)
    if s != 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + b64encode(buf.tobytes()).decode()


@lru_cache(maxsize=1024)
def thumb_uri(path):
    return "data:image/jpeg;base64," + b64encode(Path(path).read_bytes()).decode()


def thumb_of(i):
    t = model.gallery.iloc[i]["thumb"]
    p = model.thumbs_dir / t if isinstance(t, str) and t else None
    return thumb_uri(str(p)) if p is not None and p.exists() else ""


def zscore(s):
    return (s - s.mean()) / max(float(s.std()), 1e-8)


def style(fig, h=340):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=28, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Figtree, sans-serif", color=IVORY, size=13),
                      legend=dict(orientation="h", y=-0.18, bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(bgcolor="#18231F", font_color=IVORY))
    fig.update_xaxes(gridcolor=LINE, zerolinecolor="#4A5C54")
    fig.update_yaxes(gridcolor=LINE, zerolinecolor="#4A5C54")
    return fig


def show(fig, h=340):
    st.plotly_chart(style(fig, h), **FULL_PLOT)


def section(title, text=None):
    st.markdown(f'<div class="sect">{title}</div>' + (f'<div class="sub">{text}</div>' if text else ""), unsafe_allow_html=True)


def how(text):
    st.markdown(f'<div class="how">{text}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ pembacaan foto kueri
def explain_query(bgr):
    """Visual 'apa yang dilihat sistem': area topeng, siluet, garis HOG, pola LBP, dan palet warna."""
    res, cmask, smask = preprocess(bgr)
    keep = (smask[..., None] > 0)
    dim = np.where(keep, res, (res * 0.25).astype(np.uint8)).astype(np.uint8)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    fg = gray.copy()
    fg[smask == 0] = 255
    _, hv = hog(fg, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys", visualize=True)
    hv = exposure.rescale_intensity(hv, in_range=(0, float(np.percentile(hv, 99.5)) + 1e-6), out_range=(0, 1))
    hog_img = (hv[..., None] * np.array([73, 165, 210])).astype(np.uint8)
    lbp = local_binary_pattern(fg, 8, 1, "nri_uniform")
    lbp_img = cv2.cvtColor((lbp / max(float(lbp.max()), 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    pix = res[cmask > 0].reshape(-1, 3).astype(np.float32)
    if len(pix) < 50:
        pix = res.reshape(-1, 3).astype(np.float32)
    pix = pix[:: max(1, len(pix) // 2500)]
    k = int(min(5, len(pix)))
    cv2.setRNGSeed(42)
    _, lab, cen = cv2.kmeans(pix, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    cnt = np.bincount(lab.ravel(), minlength=k)
    palette = [("#%02x%02x%02x" % tuple(int(v) for v in cen[i][::-1]), float(cnt[i] / cnt.sum())) for i in np.argsort(-cnt)]
    return dict(crop=uri_from_bgr(dim, 256), sil=uri_from_bgr(cv2.cvtColor(smask, cv2.COLOR_GRAY2BGR), 256),
                hog=uri_from_bgr(hog_img, 256), lbp=uri_from_bgr(lbp_img, 256), palette=palette,
                crop_bgr=dim)


def start_search(raw, scen):
    bgr = decode_image(raw)
    st.session_state.update(sess=model.start(bgr), scen=scen, q_uri=uri_from_bgr(bgr), expl=explain_query(bgr), hist=[])


def send_feedback():
    s = st.session_state["sess"]
    pos = [i for i in s.shown if st.session_state.get(f"rel_{s.round}_{i}")]
    st.session_state["hist"].append({"Putaran": s.round, "Mirip": len(pos), "Tidak mirip": len(s.shown) - len(pos)})
    model.feedback(s, pos)


def reset_search():
    for k in ("sess", "scen", "q_uri", "expl", "hist", "seen"):
        st.session_state.pop(k, None)


def new_photo():
    reset_search()
    st.session_state["up_n"] = st.session_state.get("up_n", 0) + 1


@st.cache_data(show_spinner="Menyusun peta kemiripan ...")
def pca_map(_m, scen):
    """Proyeksi 2D semua fitur galeri (PCA acak) agar kedekatan antar topeng bisa dilihat."""
    X = np.hstack([_m.gn[b] for b in BLOCKS]).astype(np.float32)
    mu = X.mean(axis=0)
    Xc = X - mu
    rng = np.random.default_rng(42)
    Y = Xc @ rng.standard_normal((Xc.shape[1], 8)).astype(np.float32)
    for _ in range(3):
        Y = Xc @ (Xc.T @ Y)
    Q, _ = np.linalg.qr(Y)
    _, S, Vt = np.linalg.svd(Q.T @ Xc, full_matrices=False)
    comps = Vt[:2]
    var = float((S[:2] ** 2).sum() / (Xc ** 2).sum())
    return Xc @ comps.T, comps, mu, var


# ------------------------------------------------------------------ sidebar
tags = list(model.meta["scenarios"])
with st.sidebar:
    st.markdown("### Pengaturan")
    with st.expander("Indeks galeri"):
        scen = st.selectbox("Cara galeri diproses", tags, index=tags.index(model.meta["default_scenario"]),
                            format_func=lambda t: SCEN_NAMES.get(t, t), label_visibility="collapsed",
                            help="Foto unggahan pengguna selalu dipotong otomatis. Pilihan pertama paling sesuai dengan kondisi itu.")
    model.use_scenario(scen)
    w0 = model.default_weights("w2")
    defaults = {"wc": round(w0["hsv"], 2), "ws": round(w0["hog"] + w0["hu"], 2), "wt": round(w0["lbp"], 2)}
    if st.session_state.get("w_scen") != scen:
        st.session_state.update(defaults, w_scen=scen)
    st.markdown("**Seberapa penting tiap fitur?**")
    wc = st.slider("Warna", 0.0, 1.0, step=0.05, key="wc", help="Sebaran warna pada area topeng.")
    ws = st.slider("Bentuk", 0.0, 1.0, step=0.05, key="ws", help="Arah garis, kontur wajah, dan siluet.")
    wt = st.slider("Tekstur", 0.0, 1.0, step=0.05, key="wt", help="Pola permukaan halus seperti ukiran.")
    tot = (wc + ws + wt) or 1.0
    pc, ps, pt = (round(100 * x / tot) for x in (wc, ws, wt))
    st.markdown(
        f'<div class="bar"><i style="width:{pc}%;background:{GOLD}"></i><i style="width:{ps}%;background:{TEAL}"></i>'
        f'<i style="width:{pt}%;background:{LAV}"></i></div>'
        f'<div class="legend"><span>Warna {pc}%</span><span>Bentuk {ps}%</span><span>Tekstur {pt}%</span></div>', unsafe_allow_html=True)
    st.button("Kembalikan ke bobot validasi", on_click=lambda: st.session_state.update(defaults), **FULL_BTN)
    st.caption("Bobot awal dipilih otomatis lewat data validasi di notebook." +
               (" Tekstur (LBP) bernilai 0 karena tidak menambah ketepatan." if w0["lbp"] == 0 else ""))
    with st.expander("Tentang model"):
        vm = model.info.get("val_mAP_strict", {})
        st.markdown(f"Koleksi **{idr(model.meta['n_gallery'])}** citra dari **{len(model.meta['classes'])}** jenis topeng.\n\n"
                    f"mAP validasi Model 2: **{100 * vm.get('model2', 0):.1f}%**.\n\n"
                    "Skor adalah jumlah berbobot kemiripan cosine yang sudah dibakukan per kueri, bukan persentase.")

# ------------------------------------------------------------------ kepala halaman
@st.cache_data(show_spinner=False)
def mosaic_html(_m, key):
    rows = _m.gallery[_m.gallery["thumb"].map(lambda t: isinstance(t, str) and bool(t))]
    ids = random.Random(11).sample(range(len(rows)), 8)
    return "".join(f'<img src="{thumb_uri(str(_m.thumbs_dir / rows.iloc[i]["thumb"]))}">' for i in ids)


try:
    mosaic = mosaic_html(model, str(model_dir))
except Exception:
    mosaic = ""
st.markdown(
    '<div class="hero"><div><h1>Temukan topeng Bali yang paling mirip</h1>'
    '<p>Unggah satu foto topeng. Sistem membaca warna, bentuk, dan tekstur topeng itu, lalu mencari citra yang paling dekat '
    'di seluruh koleksi. Hasilnya bisa disempurnakan dengan menandai mana yang memang mirip.</p>'
    f'<div class="facts"><div><b>{idr(model.meta["n_gallery"])}</b><span>citra koleksi</span></div>'
    f'<div><b>{len(model.meta["classes"])}</b><span>jenis topeng</span></div><div><b>3</b><span>jenis fitur visual</span></div></div></div>'
    f'<div class="mosaic">{mosaic}</div></div><div class="poleng"></div>', unsafe_allow_html=True)

tab_search, tab_how, tab_perf, tab_coll = st.tabs(["Cari topeng", "Cara kerja", "Performa model", "Koleksi"])


# ------------------------------------------------------------------ tab: cari
def tile(name, uri, rank=None, score=None, top=False, bar=None, note=None):
    badge = f'<span class="mk-rank">{rank}</span>' if rank else ""
    foot = ""
    if score is not None:
        foot = f'<div class="mk-bar"><i style="width:{bar:.0f}%"></i></div><div class="mk-score"><span>skor {score:.2f}</span><span>{note or ""}</span></div>'
    return (f'<div class="mk{" top" if top else ""}"><div class="mk-img"><img src="{uri}">{badge}</div>'
            f'<div class="mk-body"><div class="mk-name">{name}</div>{foot}</div></div>')


def page_search():
    section("Unggah foto", "Gunakan foto satu topeng dengan latar yang tidak terlalu ramai. Format JPG, PNG, BMP, atau WebP.")
    up = st.file_uploader("Unggah foto topeng", type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="collapsed",
                          key=f"up_{st.session_state.get('up_n', 0)}")
    if up is None:
        reset_search()
        how("<b>Belum ada foto.</b> Setelah foto diunggah, bagian ini akan menampilkan apa yang dibaca sistem, sepuluh topeng "
            "paling mirip beserta alasannya, perkiraan jenis topeng, dan peta kedekatan di seluruh koleksi.")
        return
    raw = up.getvalue()
    key = hashlib.md5(raw).hexdigest() + scen
    if st.session_state.get("seen") != key:
        with st.spinner("Membaca foto dan menghitung fitur ..."):
            try:
                start_search(raw, scen)
            except ValueError as e:
                st.error(str(e))
                return
        st.session_state["seen"] = key

    sess, ex = st.session_state["sess"], st.session_state["expl"]
    weights = model.weights_from_ui(wc, ws, wt)
    results = model.rank(sess, weights, TOP_K)

    # ---- yang dibaca sistem
    section("Yang dibaca sistem", "Foto diperkecil, topeng dipisahkan dari latar, lalu empat jenis fitur dihitung dari area topeng saja.")
    items = [(st.session_state["q_uri"], "Foto Anda", "Gambar asli yang diunggah."),
             (ex["crop"], "Area topeng", "Hasil segmentasi dan pemotongan. Latar digelapkan."),
             (ex["sil"], "Siluet", "Dipakai untuk Hu Moments, yaitu ukuran bentuk luar."),
             (ex["hog"], "Arah garis (HOG)", "Setiap sel menunjukkan arah tepi dominan. Garis terang berarti tepi kuat."),
             (ex["lbp"], "Pola permukaan (LBP)", "Setiap piksel diberi kode pola lokal. Ukiran halus tampak sebagai pola berulang.")]
    st.markdown('<div class="five">' + "".join(f'<figure><img class="pic" src="{u}"><figcaption><b>{t}</b>{d}</figcaption></figure>'
                                                for u, t, d in items) + "</div>", unsafe_allow_html=True)
    pal = "".join(f'<i style="width:{100 * p:.1f}%;background:{c}" title="{c}"></i>' for c, p in ex["palette"])
    st.markdown(f'<div class="panel" style="margin-top:1rem;height:auto"><h4>Warna dominan</h4><div class="pal">{pal}</div>'
                '<div class="cap">Lima kelompok warna terbesar pada area topeng. Lebar tiap blok sesuai luas warnanya. '
                'Fitur warna sebenarnya berupa histogram HSV 16×4×4, bukan lima warna ini; palet ini hanya ringkasan visualnya.</div></div>',
                unsafe_allow_html=True)

    # ---- hasil
    section(f'Sepuluh paling mirip<span class="pill">putaran umpan balik {sess.round}</span>',
            "Urutan berdasarkan skor gabungan. Centang topeng yang memang mirip dengan foto Anda. Yang tidak dicentang dianggap tidak mirip.")
    zs = {b: zscore(model.gn[b] @ sess.qb[b]) if model.info["zscore"] else model.gn[b] @ sess.qb[b] for b in BLOCKS}
    contrib = {i["id"]: {g: sum(weights.get(b, 0) * float(zs[b][i["id"]]) for b in bl) for g, bl in GROUPS.items()} for i in results}
    top_score = max(results[0]["score"], 1e-6)
    picked = 0
    for start in range(0, len(results), 5):
        cols = st.columns(5, gap="small")
        for col, (n, r) in zip(cols, list(enumerate(results))[start:start + 5]):
            lead = max(contrib[r["id"]], key=contrib[r["id"]].get)
            note = f'<span class="dot" style="background:{GROUP_COLOR[lead]}"></span>{lead}'
            with col:
                uri = thumb_uri(r["thumb"]) if r["thumb"] and Path(r["thumb"]).exists() else ""
                st.markdown(tile(short(r["label"]), uri, n + 1, r["score"], n == 0, max(4, min(100, 100 * r["score"] / top_score)), note),
                            unsafe_allow_html=True)
                picked += st.checkbox("Mirip", key=f"rel_{sess.round}_{r['id']}")
        st.write("")
    how("<b>Cara membaca kartu.</b> Bar emas menunjukkan skor dibanding peringkat 1. Titik berwarna di kanan bawah menandai fitur "
        "yang paling menyumbang skor: <span class='dot' style='background:%s'></span>warna, <span class='dot' style='background:%s'></span>bentuk, "
        "<span class='dot' style='background:%s'></span>tekstur. Setelah menandai, tekan <b>Sempurnakan hasil</b>. Topeng yang sudah "
        "dinilai tidak ditampilkan lagi pada putaran berikutnya." % (GOLD, TEAL, LAV))
    b1, b2, b3 = st.columns([2.2, 1.2, 1.2], gap="small")
    b1.markdown(f'<div style="padding-top:.45rem;color:var(--mute)">{picked} dipilih mirip, {len(results) - picked} tidak mirip</div>',
                unsafe_allow_html=True)
    b2.button("Sempurnakan hasil", type="primary", on_click=send_feedback, **FULL_BTN)
    b3.button("Cari foto lain", on_click=new_photo, **FULL_BTN)

    # ---- perkiraan jenis
    section("Perkiraan jenis topeng", "Dihitung dari jenis yang muncul di sepuluh hasil. Ini perkiraan berdasarkan kemiripan, bukan klasifikasi pasti.")
    cnt = Counter(r["label"] for r in results).most_common()
    c1, c2 = st.columns([1, 1.1], gap="large")
    with c1:
        fig = go.Figure(go.Pie(labels=[short(k) for k, _ in cnt], values=[v for _, v in cnt], hole=0.62, sort=False,
                               marker=dict(colors=CLASS_COLORS, line=dict(color="#101815", width=2)),
                               textinfo="value", hovertemplate="%{label}: %{value} dari 10<extra></extra>"))
        fig.update_layout(annotations=[dict(text=f"<b>{short(cnt[0][0])}</b><br>{cnt[0][1]} dari {len(results)}", showarrow=False,
                                            font=dict(size=15, color=IVORY))], showlegend=True)
        show(fig, 330)
    with c2:
        name, k = cnt[0]
        st.markdown(f'<div class="panel"><h4>{short(name)}</h4><div class="sub" style="margin:0">{DESC.get(name, "Jenis topeng pada koleksi ini.")}</div>'
                    f'<div class="cap" style="margin-top:.8rem">Muncul <b style="color:var(--ink)">{k} dari {len(results)}</b> hasil teratas. '
                    f'{"Mayoritas jelas, sehingga perkiraan ini cukup kuat." if k >= 6 else "Hasil bercampur: beberapa jenis berdekatan, jadi perkiraan ini kurang pasti."} '
                    'Ringkasan karakter adalah gambaran umum dan sebaiknya dicek ke rujukan budaya.</div></div>', unsafe_allow_html=True)

    # ---- kenapa
    section("Kenapa hasilnya begini?", "Skor tiap topeng adalah jumlah kontribusi warna, bentuk, dan tekstur. Bar ke kiri berarti fitur itu di bawah rata-rata koleksi.")
    labels = [f"#{n + 1} {short(r['label'])}" for n, r in enumerate(results)]
    fig = go.Figure()
    for g in GROUPS:
        fig.add_bar(y=labels, x=[contrib[r["id"]][g] for r in results], name=g, orientation="h", marker_color=GROUP_COLOR[g],
                    hovertemplate="%{y}<br>" + g + ": %{x:.2f}<extra></extra>")
    fig.update_layout(barmode="relative", yaxis=dict(autorange="reversed"), xaxis_title="Kontribusi ke skor")
    show(fig, 400)
    pick = st.selectbox("Bandingkan satu hasil secara rinci", range(len(results)), format_func=lambda n: labels[n])
    r = results[pick]
    d1, d2, d3 = st.columns([1, 1, 1.6], gap="medium")
    d1.markdown(f'<div class="panel"><h4>Topeng Anda</h4><img class="pic" src="{uri_from_bgr(ex["crop_bgr"], 300)}"></div>', unsafe_allow_html=True)
    d2.markdown(f'<div class="panel"><h4>#{pick + 1} {short(r["label"])}</h4><img class="pic" src="{thumb_uri(r["thumb"]) if r["thumb"] else ""}"></div>',
                unsafe_allow_html=True)
    with d3:
        vals = [float(zs[b][r["id"]]) for b in BLOCKS]
        fig = go.Figure(go.Bar(y=[BLOCK_NAME[b] for b in BLOCKS], x=vals, orientation="h",
                               marker_color=[TEAL if v >= 0 else "#C0573F" for v in vals], text=[f"{v:+.1f}" for v in vals],
                               textposition="outside", hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
        fig.update_layout(title=dict(text="Kemiripan per fitur", font=dict(size=14)), yaxis=dict(autorange="reversed"),
                          xaxis_title="Simpangan baku di atas rata-rata koleksi")
        show(fig, 300)
    how("<b>Membaca grafik.</b> Angka +2 berarti topeng itu jauh lebih mirip dengan foto Anda pada fitur tersebut dibanding topeng "
        "koleksi pada umumnya. Fitur yang bobotnya 0 tetap ditampilkan di sini, tetapi tidak ikut menentukan peringkat.")

    # ---- peta
    section("Peta kedekatan koleksi", "Setiap titik adalah satu citra koleksi. Titik yang berdekatan memiliki fitur yang mirip. Klik nama jenis di legenda untuk menyembunyikannya.")
    coords, comps, mu, var = pca_map(model, scen)
    qxy = (np.hstack([sess.qb[b] for b in BLOCKS]) - mu) @ comps.T
    labs = np.array(model.labels)
    fig = go.Figure()
    for k, c in enumerate(sorted(set(labs))):
        m = labs == c
        fig.add_trace(go.Scattergl(x=coords[m, 0], y=coords[m, 1], mode="markers", name=short(c), opacity=0.55,
                                   marker=dict(size=5, color=CLASS_COLORS[k % len(CLASS_COLORS)]),
                                   hovertemplate=short(c) + "<extra></extra>"))
    ids = [r["id"] for r in results]
    fig.add_trace(go.Scatter(x=coords[ids, 0], y=coords[ids, 1], mode="markers+text", name="10 hasil",
                             text=[str(n + 1) for n in range(len(ids))], textposition="top center", textfont=dict(color=IVORY),
                             marker=dict(size=15, symbol="circle-open", line=dict(width=2.5, color=IVORY))))
    fig.add_trace(go.Scatter(x=[qxy[0]], y=[qxy[1]], mode="markers", name="Foto Anda",
                             marker=dict(size=22, symbol="star", color=GOLD, line=dict(width=2, color="#101815"))))
    fig.update_layout(xaxis=dict(showticklabels=False, title=f"Dimensi 1"), yaxis=dict(showticklabels=False, title="Dimensi 2"))
    show(fig, 520)
    how(f"<b>Catatan.</b> Peta ini meringkas ribuan dimensi fitur menjadi dua, sehingga hanya {100 * var:.0f}% variasi data yang tergambar. "
        "Dua titik yang tampak berdekatan belum tentu benar-benar dekat, tetapi kelompok yang jelas terpisah biasanya memang berbeda.")

    # ---- riwayat
    if st.session_state["hist"]:
        section("Dampak umpan balik", "Jumlah hasil yang Anda tandai mirip pada tiap putaran. Jika umpan balik bekerja, angkanya cenderung naik.")
        h = pd.DataFrame(st.session_state["hist"])
        fig = go.Figure(go.Bar(x=[f"Putaran {p}" for p in h["Putaran"]], y=h["Mirip"], marker_color=GOLD, text=h["Mirip"], textposition="outside"))
        fig.update_layout(yaxis=dict(range=[0, TOP_K + 1], title="Dari 10 hasil"))
        show(fig, 280)


# ------------------------------------------------------------------ tab: cara kerja
def page_how():
    section("Dari foto ke hasil", "Lima tahap yang sama dijalankan untuk setiap foto yang diunggah.")
    st.markdown('<div class="flow">'
                '<div class="step"><b>Segmentasi</b><span>GrabCut memisahkan topeng dari latar. Area topeng dipotong, diubah ke 128×128, dan kecerahannya diratakan dengan CLAHE.</span></div>'
                '<div class="step"><b>Ekstraksi fitur</b><span>Empat vektor dihitung: warna HSV, arah garis HOG, siluet Hu Moments, dan pola permukaan LBP.</span></div>'
                '<div class="step"><b>Normalisasi</b><span>Tiap vektor dinormalisasi dengan statistik koleksi supaya skalanya sebanding.</span></div>'
                '<div class="step"><b>Fusi berbobot</b><span>Kemiripan cosine tiap fitur dibakukan lalu dijumlahkan menurut bobot. Hasilnya diurutkan.</span></div>'
                '<div class="step"><b>Umpan balik</b><span>Topeng yang ditandai mirip menggeser kueri mendekat, yang tidak mirip menjauh (Rocchio).</span></div></div>',
                unsafe_allow_html=True)
    section("Empat fitur yang dipakai")
    st.markdown(
        '<table class="t"><tr><th>Fitur</th><th>Menangkap</th><th>Dimensi</th><th>Cara hitung</th></tr>'
        '<tr><td>Warna (HSV)</td><td>Sebaran warna dan kecerahan di area topeng</td><td>256</td><td>Histogram 16×4×4 pada kanal H, S, V</td></tr>'
        '<tr><td>Garis dan kontur (HOG)</td><td>Arah tepi: alis, hidung, mulut, hiasan</td><td>1764</td><td>9 arah, sel 16×16, blok 2×2</td></tr>'
        '<tr><td>Siluet (Hu Moments)</td><td>Bentuk luar topeng, tidak terpengaruh posisi dan ukuran</td><td>7</td><td>Tujuh momen invarian dari siluet, skala log</td></tr>'
        '<tr><td>Tekstur (LBP)</td><td>Pola permukaan halus seperti ukiran</td><td>236</td><td>Uniform LBP (P=8, R=1) pada grid 2×2 sel</td></tr></table>', unsafe_allow_html=True)
    section("Rumus yang dipakai")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Skor kemiripan.** Tiap fitur *b* dibandingkan dengan cosine, dibakukan per kueri, lalu dijumlahkan menurut bobot *w*.")
        st.latex(r"S(q,g)=\sum_{b\in\{HSV,HOG,Hu,LBP\}} w_b \cdot z\!\left(\cos(q_b,g_b)\right)")
    with c2:
        ro = model.meta.get("rocchio", {"alpha": 1.0, "beta": 0.75, "gamma": 0.15})
        st.markdown(f"**Umpan balik Rocchio.** Kueri baru digeser ke rata-rata hasil yang mirip (P) dan menjauh dari yang tidak mirip (N). "
                    f"Nilai: α = {ro['alpha']}, β = {ro['beta']}, γ = {ro['gamma']}.")
        st.latex(r"q'=\alpha\,q+\beta\,\overline{P}-\gamma\,\overline{N}")
    section("Istilah yang muncul")
    how("<b>Skor.</b> Bukan persentase. Semakin tinggi semakin mirip, dan nilainya hanya bermakna untuk membandingkan hasil pada pencarian yang sama.<br>"
        "<b>Bobot.</b> Menentukan seberapa besar tiap fitur memengaruhi skor. Bobot awal dipilih dari data validasi di notebook.<br>"
        "<b>Residual ranking.</b> Topeng yang sudah Anda nilai dikeluarkan dari putaran berikutnya, supaya yang muncul selalu citra baru.<br>"
        "<b>Segmentasi otomatis.</b> Foto Anda tidak punya kotak anotasi, jadi topeng dicari sendiri oleh GrabCut. Pada foto berlatar ramai, tahap ini bisa meleset dan hasil pencarian ikut menurun.")


# ------------------------------------------------------------------ tab: performa
def page_perf():
    section("Seberapa baik modelnya?", "Angka berikut diambil dari hasil evaluasi di notebook pada data uji. Label hanya dipakai sebagai kunci jawaban, tidak ada model yang dilatih.")
    st.markdown('<div class="kpis">'
                '<div class="kpi"><b>77,4%</b><span>P@10 Model 2</span><small>Skenario 2. Baseline: 73,0%.</small></div>'
                '<div class="kpi"><b>35,3%</b><span>mAP ketat</span><small>Skenario 2. Baseline: 30,4%.</small></div>'
                '<div class="kpi"><b>+9,8</b><span>poin mAP dari umpan balik</span><small>Putaran 3 dibanding kontrol, tanpa kotak anotasi. CI95 +8,8 sampai +10,9.</small></div>'
                '<div class="kpi"><b>72,4%</b><span>P@10 tanpa kotak anotasi</span><small>Kondisi pengguna sebenarnya (Skenario 2b).</small></div></div>', unsafe_allow_html=True)
    how("<b>P@10</b> adalah rata-rata persentase topeng sejenis di 10 hasil teratas. <b>mAP</b> menilai seberapa awal semua topeng sejenis muncul di urutan. "
        "mAP lebih rendah dari P@10 karena tiap jenis punya sekitar 270 citra, sehingga mustahil semuanya masuk 10 teratas.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Model usulan lawan baseline pada tiga skenario**")
        sk = ["Skenario 1\nhanya resize", "Skenario 2\npra-proses penuh", "Skenario 2b\ntanpa anotasi"]
        fig = go.Figure()
        fig.add_bar(x=sk, y=[24.37, 30.38, 28.37], name="Baseline [16]", marker_color=SAGE, text=[24.4, 30.4, 28.4], textposition="outside")
        fig.add_bar(x=sk, y=[25.90, 35.28, 32.11], name="Model 2", marker_color=GOLD, text=[25.9, 35.3, 32.1], textposition="outside")
        fig.update_layout(barmode="group", yaxis_title="mAP ketat (%)", yaxis_range=[0, 42])
        show(fig, 340)
    with c2:
        st.markdown("**Kontribusi tiap fitur** (Skenario 2b, mAP ketat)")
        names = ["Acak", "Hu saja", "LBP saja", "HSV saja", "HSV+HOG+Hu+LBP\n(bobot sama)", "HSV+HOG+Hu\n(bobot sama)", "HSV+HOG\n(bobot sama)", "HOG saja", "Model 1 dan 2\n(bobot validasi)"]
        vals = [7.9, 10.04, 13.49, 18.03, 25.05, 26.05, 29.37, 30.00, 32.11]
        fig = go.Figure(go.Bar(y=names, x=vals, orientation="h", text=[f"{v:.1f}" for v in vals], textposition="outside",
                               marker_color=[SAGE] * 8 + [GOLD]))
        fig.update_layout(xaxis_range=[0, 38], xaxis_title="mAP ketat (%)")
        show(fig, 340)

    section("Per jenis topeng", "P@10 pada Skenario 2. Jenis dengan wajah yang khas, seperti Wijil, Tua, dan Sidakarya, paling mudah ditemukan.")
    cls = ["Wijil", "Tua", "Jauk Manis", "Sidakarya", "Barong Macan", "Barong Ket", "Jauk Keras", "Penasar", "Telek", "Barong Bangkung", "Bujuh", "Dalem Arsa Wijaya", "Keras"]
    m2 = [93.20, 91.15, 93.85, 87.04, 78.21, 74.00, 73.20, 72.26, 71.15, 70.69, 69.26, 67.41, 66.54]
    bl = [96.40, 93.85, 75.00, 90.74, 81.43, 68.80, 74.80, 74.84, 53.08, 51.72, 65.56, 65.93, 58.85]
    order = np.argsort(m2)[::-1]
    fig = go.Figure()
    fig.add_bar(x=[cls[i] for i in order], y=[bl[i] for i in order], name="Baseline [16]", marker_color=SAGE)
    fig.add_bar(x=[cls[i] for i in order], y=[m2[i] for i in order], name="Model 2", marker_color=GOLD)
    fig.update_layout(barmode="group", yaxis_title="P@10 (%)", yaxis_range=[0, 105])
    show(fig, 380)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("Efek umpan balik", "Tiga putaran Rocchio dibanding kontrol tanpa umpan balik pada galeri sisa yang sama besar.")
        r = ["Putaran 0", "Putaran 1", "Putaran 2", "Putaran 3"]
        fig = go.Figure()
        fig.add_scatter(x=r, y=[35.28, 37.09, 38.71, 39.58], name="Dengan umpan balik", mode="lines+markers", line=dict(color=GOLD, width=3))
        fig.add_scatter(x=r, y=[35.28, 32.84, 31.26, 29.99], name="Kontrol", mode="lines+markers", line=dict(color=SAGE, width=3, dash="dot"))
        fig.update_layout(yaxis_title="mAP ketat (%)")
        show(fig, 320)
    with c2:
        section("Ketahanan terhadap gangguan", "mAP ketat Skenario 2b setelah foto kueri diubah.")
        pt = ["Asli", "Rotasi ±15°", "Kecerahan ±30%", "Skala 0,8×", "Potong tepi 20%"]
        fig = go.Figure()
        fig.add_bar(x=pt, y=[30.15, 23.26, 29.78, 16.72, 21.69], name="Baseline [16]", marker_color=SAGE)
        fig.add_bar(x=pt, y=[35.22, 26.53, 34.66, 18.81, 25.28], name="Model 1 dan 2", marker_color=GOLD)
        fig.update_layout(barmode="group", yaxis_title="mAP ketat (%)")
        show(fig, 320)

    section("Pemeriksaan klaim", "Hanya klaim berstatus didukung yang aman dinyatakan sebagai peningkatan (Skenario 2b, interval kepercayaan 95%).")
    rows = [("Model 1 lebih baik dari baseline (bobot sama)", "+3,74 (+2,74 sampai +4,67)", True),
            ("Model 1 lebih baik dari baseline dengan bobot dituning", "+2,44 (+1,16 sampai +3,60)", True),
            ("Warna menambah dibanding HOG saja", "+2,11 (+1,52 sampai +2,72)", True),
            ("Umpan balik lebih baik dari kontrol", "+9,82 (+8,84 sampai +10,85)", True),
            ("Tekstur (LBP) menambah ketepatan", "+0,00. Bobot LBP = 0 pada Model 2", False)]
    st.markdown('<table class="t"><tr><th>Klaim</th><th>Selisih mAP (poin)</th><th>Status</th></tr>' + "".join(
        f'<tr><td>{a}</td><td>{b}</td><td><span class="tag {"ok" if ok else "no"}">{"Didukung" if ok else "Tidak terbukti"}</span></td></tr>'
        for a, b, ok in rows) + "</table>", unsafe_allow_html=True)
    how("<b>Yang perlu diingat.</b> Keunggulan Model 1 hampir hilang saat foto diperkecil ke skala 0,8×. Kinerja turun pada foto berlatar ramai atau topeng "
        "yang terpotong, dan jenis yang mirip seperti ketiga Barong sulit dibedakan hanya dari fitur visual.")


# ------------------------------------------------------------------ tab: koleksi
def page_coll():
    df = model.gallery
    section("Isi koleksi", f"{idr(len(df))} citra dari {df['label'].nunique()} jenis topeng.")
    if "split" in df:
        ct = pd.crosstab(df["label"], df["split"])
        ct = ct.loc[ct.sum(axis=1).sort_values().index]
        fig = go.Figure()
        for k, s in enumerate(ct.columns):
            fig.add_bar(y=[short(i) for i in ct.index], x=ct[s], name=str(s), orientation="h", marker_color=[GOLD, TEAL, LAV, SAGE][k % 4])
        fig.update_layout(barmode="stack", xaxis_title="Jumlah citra")
    else:
        vc = df["label"].value_counts().sort_values()
        fig = go.Figure(go.Bar(y=[short(i) for i in vc.index], x=vc.values, orientation="h", marker_color=GOLD))
    show(fig, 470)
    section("Jelajahi jenis topeng", "Pilih satu jenis untuk melihat contoh citranya.")
    cl = sorted(df["label"].unique())
    sel = st.selectbox("Jenis", cl, format_func=short, label_visibility="collapsed")
    ids = df.index[df["label"] == sel].tolist()
    seed = st.number_input("Acak contoh", 0, 999, 0, help="Ganti angka untuk melihat contoh lain.")
    sample = random.Random(int(seed)).sample(ids, min(10, len(ids)))
    if sel in DESC:
        how(DESC[sel])
    for s0 in range(0, len(sample), 5):
        for col, i in zip(st.columns(5, gap="small"), sample[s0:s0 + 5]):
            col.markdown(tile(short(sel), thumb_of(i)), unsafe_allow_html=True)
        st.write("")


with tab_search:
    page_search()
with tab_how:
    page_how()
with tab_perf:
    page_perf()
with tab_coll:
    page_coll()