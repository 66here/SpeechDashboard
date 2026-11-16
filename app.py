"""Suara & Emosi — dashboard Speech Emotion Recognition (IndoWaveSentiment).

Jalankan dari folder proyek:  streamlit run app.py
Model, label, preprocessing, dan hasil evaluasi dipakai apa adanya dari model/, src/, dan assets/.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import app_utils as U

warnings.filterwarnings("ignore")  # peringatan versi internal tidak perlu ditampilkan ke pengguna

HERE = Path(__file__).resolve().parent


def _dir(name: str) -> Path:
    """Folder proyek (assets/, model/, src/); jika repo diunggah datar tanpa folder, pakai folder aplikasi."""
    return HERE / name if (HERE / name).is_dir() else HERE


ASSETS, SRC, MODEL_DIR = _dir("assets"), _dir("src"), _dir("model")
SAMPLES = ASSETS / "samples" if (ASSETS / "samples").is_dir() else ASSETS


def list_samples() -> list:
    return sorted(p for p in SAMPLES.glob("*.wav") if p.stem in U.ORDER)


st.set_page_config(page_title="Suara & Emosi", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")

# ============================================================ tema
BG, SIDEBAR, PRIMARY, PRIMARY_D = "#FAF5EC", "#F1E7D6", "#3B4BB8", "#2C3A94"
TERRA, AMBER, INK, MUTED = "#D9593D", "#D98E04", "#1F2433", "#5F6479"
FONT = "Plus Jakarta Sans, sans-serif"
CREAM_SCALE = [[0, "#FFFFFF"], [0.35, "#DADFF5"], [0.7, "#7C89D6"], [1, PRIMARY_D]]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background-color:#FAF5EC; background-image: radial-gradient(rgba(31,36,51,.07) 1px, transparent 1px); background-size: 22px 22px; color:#1F2433; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1240px; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
h1,h2,h3,h4 { font-family:'Fraunces', serif !important; color:#1F2433; letter-spacing:-0.01em; }
h2 { font-size: 1.7rem !important; }
[data-testid="stSidebar"] { background:#F1E7D6; border-right:1px solid rgba(31,36,51,.08); }
[data-testid="stSidebar"] .stButton>button { width:100%; justify-content:flex-start; text-align:left; border-radius:999px; padding:.55rem 1rem; font-weight:600; border:1px solid transparent; background:transparent; color:#1F2433; box-shadow:none; }
[data-testid="stSidebar"] .stButton>button:hover { background:rgba(59,75,184,.10); border-color:rgba(59,75,184,.25); color:#2C3A94; }
[data-testid="stSidebar"] .stButton>button[kind="primary"] { background:#3B4BB8; color:#fff; border-color:#3B4BB8; box-shadow:0 6px 16px rgba(59,75,184,.25); }
.stButton>button, .stDownloadButton>button { border-radius:999px; font-weight:600; padding:.5rem 1.4rem; transition:all .2s ease; }
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"] { background:#3B4BB8; border:1px solid #3B4BB8; color:#fff; }
.stButton>button[kind="primary"]:hover { background:#2C3A94; transform:translateY(-1px); }
.brand { font-family:'Fraunces',serif; font-size:1.65rem; font-weight:700; color:#1F2433; line-height:1.1; }
.brand-tag { font-style:italic; color:#5F6479; font-size:.88rem; margin:.15rem 0 1rem 0; }
.side-note { color:#5F6479; font-size:.78rem; line-height:1.5; }
.hero { background: linear-gradient(135deg,#FFF9EC 0%,#F6E8CF 55%,#EFDDC0 100%); border:1px solid rgba(31,36,51,.08); border-radius:24px; padding:2.4rem 2.6rem; display:flex; gap:2rem; align-items:center; justify-content:space-between; box-shadow:0 10px 30px rgba(31,36,51,.06); margin-bottom:1.4rem; overflow:hidden; }
.hero .kicker { text-transform:uppercase; letter-spacing:.14em; font-size:.74rem; font-weight:700; color:#D9593D; }
.hero h1 { font-size:3.3rem; line-height:1.05; margin:.35rem 0 .6rem 0; }
.hero .sub { font-size:1.12rem; color:#2C3A94; font-weight:500; margin-bottom:.7rem; max-width:34rem; }
.hero .desc { color:#5F6479; max-width:34rem; line-height:1.65; font-size:.95rem; }
.hero svg { flex:0 0 auto; }
.page-title { font-size:2.3rem; margin-bottom:.1rem; }
.page-sub { color:#5F6479; margin-bottom:1.3rem; max-width:50rem; line-height:1.6; }
.kicker2 { text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:700; color:#D9593D; margin-top:1.8rem; }
.sec-title { font-family:'Fraunces',serif; font-size:1.55rem; font-weight:600; margin:.1rem 0 .3rem 0; }
.sec-desc { color:#5F6479; font-size:.93rem; margin-bottom:.8rem; line-height:1.6; max-width:52rem; }
.mcard { background:#fff; border:1px solid rgba(31,36,51,.08); border-radius:18px; padding:1.1rem 1.3rem; box-shadow:0 4px 16px rgba(31,36,51,.05); transition:transform .2s ease, box-shadow .2s ease; height:100%; }
.mcard:hover { transform:translateY(-3px); box-shadow:0 10px 26px rgba(31,36,51,.10); }
.mcard .lab { color:#5F6479; font-size:.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.08em; }
.mcard .val { font-family:'Fraunces',serif; font-size:2.2rem; font-weight:700; line-height:1.15; margin-top:.2rem; }
.mcard .note { color:#5F6479; font-size:.82rem; margin-top:.15rem; }
.ecard { background:#fff; border:1px solid rgba(31,36,51,.08); border-top:5px solid var(--c); border-radius:18px; padding:1.1rem 1rem; text-align:center; box-shadow:0 4px 16px rgba(31,36,51,.05); transition:transform .2s ease, box-shadow .2s ease; height:100%; }
.ecard:hover { transform:translateY(-4px); box-shadow:0 12px 26px rgba(31,36,51,.11); }
.ecard .em { font-size:2.6rem; } .ecard .nm { font-family:'Fraunces',serif; font-weight:600; font-size:1.2rem; color:var(--c); }
.ecard .lb { font-size:.72rem; color:#5F6479; text-transform:uppercase; letter-spacing:.1em; margin-bottom:.4rem; }
.ecard .ds { font-size:.84rem; color:#5F6479; line-height:1.5; }
.predcard { border-radius:22px; padding:1.6rem 1.4rem; color:#fff; text-align:center; box-shadow:0 14px 34px rgba(31,36,51,.18); }
.predcard .em { font-size:4.6rem; line-height:1.1; } .predcard .nm { font-family:'Fraunces',serif; font-size:2.1rem; font-weight:700; }
.predcard .cf { font-size:1.05rem; opacity:.95; margin-top:.2rem; } .predcard .cfv { font-size:2.4rem; font-weight:700; line-height:1.1; }
.pill { display:inline-block; padding:.28rem .9rem; border-radius:999px; font-weight:700; font-size:.85rem; color:#fff; margin-top:.8rem; background:rgba(255,255,255,.22); border:1px solid rgba(255,255,255,.45); }
.badge { display:inline-block; background:#D9593D; color:#fff; border-radius:999px; padding:.22rem .8rem; font-size:.74rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
.insight { background:rgba(59,75,184,.07); border-left:4px solid #3B4BB8; border-radius:10px; padding:.7rem 1rem; color:#1F2433; font-size:.9rem; line-height:1.6; margin:.5rem 0 1rem 0; }
.warnbox { background:rgba(217,89,61,.08); border-left:4px solid #D9593D; border-radius:10px; padding:.8rem 1rem; color:#1F2433; font-size:.9rem; line-height:1.6; margin:.6rem 0; }
.flow { display:flex; flex-wrap:wrap; gap:.5rem; align-items:stretch; margin:.5rem 0 1rem 0; }
.flow .step { flex:1 1 120px; background:#fff; border:1px solid rgba(31,36,51,.08); border-radius:16px; padding:.8rem .7rem; text-align:center; box-shadow:0 4px 14px rgba(31,36,51,.05); }
.flow .step .n { width:26px; height:26px; line-height:26px; border-radius:50%; background:#3B4BB8; color:#fff; font-weight:700; font-size:.8rem; margin:0 auto .35rem auto; }
.flow .step .t { font-weight:700; font-size:.88rem; } .flow .step .d { color:#5F6479; font-size:.76rem; margin-top:.15rem; }
.flow .arrow { align-self:center; color:#D9593D; font-weight:700; font-size:1.2rem; }
[data-testid="stPlotlyChart"] { background:#fff; border:1px solid rgba(31,36,51,.08); border-radius:18px; padding:.4rem .6rem; box-shadow:0 4px 16px rgba(31,36,51,.05); }
[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; border:1px solid rgba(31,36,51,.08); }
[data-testid="stExpander"] { background:#fff; border-radius:16px; border:1px solid rgba(31,36,51,.08); }
.stTabs [data-baseweb="tab-list"] { gap:.4rem; }
.stTabs [data-baseweb="tab"] { border-radius:999px; padding:.4rem 1.1rem; background:rgba(31,36,51,.05); }
.stTabs [aria-selected="true"] { background:#3B4BB8 !important; color:#fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
@media (max-width: 900px) { .hero { flex-direction:column; } .hero h1 { font-size:2.4rem; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============================================================ util tampilan
_VER = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2])
_WIDTH_API = _VER >= (1, 49)  # width="stretch" menggantikan use_container_width


def html(s: str) -> None:
    """Render HTML; hapus indentasi & baris kosong agar tidak dibaca sebagai blok kode markdown."""
    st.markdown("\n".join(l.strip() for l in s.strip().splitlines() if l.strip()), unsafe_allow_html=True)


def bold(t: str) -> str:
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", t)


def insight(t: str) -> None:
    html(f"<div class='insight'>💡 {bold(t)}</div>")


def warnbox(t: str) -> None:
    html(f"<div class='warnbox'>⚠️ {bold(t)}</div>")


def section(kicker: str, title: str, desc: str = "") -> None:
    html(f"<div class='kicker2'>{kicker}</div><div class='sec-title'>{title}</div>" + (f"<div class='sec-desc'>{bold(desc)}</div>" if desc else ""))


def page_header(title: str, sub: str) -> None:
    html(f"<div class='page-title sec-title'>{title}</div><div class='page-sub'>{bold(sub)}</div>")


def metric_card(label: str, value: str, note: str = "", color: str = INK) -> str:
    return f"<div class='mcard'><div class='lab'>{label}</div><div class='val' style='color:{color}'>{value}</div><div class='note'>{note}</div></div>"


def show(fig, key: str) -> None:
    cfg = {"displayModeBar": False}
    if _WIDTH_API:
        st.plotly_chart(fig, width="stretch", key=key, config=cfg)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key, config=cfg)


def table(df, key: str, **kw) -> None:
    if _WIDTH_API:
        st.dataframe(df, width="stretch", hide_index=True, key=key, **kw)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True, key=key, **kw)


def style(fig: go.Figure, h: int = 340, title: str | None = None, **kw) -> go.Figure:
    fig.update_layout(
        height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=13), margin=dict(l=10, r=14, t=52 if title else 20, b=10),
        title=dict(text=title, font=dict(family="Fraunces, serif", size=18), x=0.01) if title else None,
        hoverlabel=dict(font_family=FONT, bgcolor="#fff", font_color=INK), **kw)
    fig.update_xaxes(gridcolor="rgba(95,100,121,.14)", zeroline=False, linecolor="rgba(95,100,121,.25)")
    fig.update_yaxes(gridcolor="rgba(95,100,121,.14)", zeroline=False, linecolor="rgba(95,100,121,.25)")
    return fig


def need(df, name: str) -> bool:
    if df is None:
        st.info(f"Aset `{name}` tidak tersedia, sehingga bagian ini tidak dapat ditampilkan.")
        return False
    return True


def go_page(p: str) -> None:
    st.session_state["page"] = p


# ============================================================ pemuatan data & model (cache)
@st.cache_data(show_spinner=False)
def load_data() -> dict:
    return U.load_assets(ASSETS)


@st.cache_resource(show_spinner=False)
def get_model():
    """Muat model final sekali per proses, lewat src/infer.py (kontrak yang sama dengan notebook)."""
    try:
        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))
        import infer
        pre = infer.load(str(MODEL_DIR), str(SRC))
        return pre, infer, None
    except Exception as e:  # noqa: BLE001
        return None, None, f"{type(e).__name__}: {e}"


class AudioError(Exception):
    pass


def run_prediction(raw: bytes, name: str) -> dict:
    """Simpan ke file sementara -> validasi -> infer.predict (preprocessing identik dengan training) -> hapus file."""
    pre, infer, err = get_model()
    if pre is None:
        raise AudioError(f"Model belum dapat dimuat. Pastikan `model/model_final.joblib` dan folder `src/` ada. (Detail: {err})")
    if len(raw) < 100 or raw[:4] != b"RIFF":
        raise AudioError("Format audio tidak didukung. Gunakan file WAV (PCM) yang valid.")
    import librosa
    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        try:
            y = librosa.load(path, sr=U.SR, mono=True)[0]
        except Exception as e:  # noqa: BLE001
            raise AudioError("File audio tidak dapat dibaca. File mungkin rusak atau bukan WAV yang valid.") from e
        if len(y) < int(0.3 * U.SR):
            raise AudioError("Rekaman terlalu pendek (kurang dari 0,3 detik). Coba rekaman 2–4 detik.")
        if float(np.max(np.abs(y))) < 1e-4:
            raise AudioError("Rekaman nyaris senyap. Periksa mikrofon atau tingkat volume, lalu coba lagi.")
        try:
            labels, p = infer.predict(pre, path)
        except Exception as e:  # noqa: BLE001
            raise AudioError(f"Prediksi gagal diproses ({type(e).__name__}). Coba file audio lain.") from e
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    p = np.asarray(p, dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    return dict(name=name, labels=list(labels), p=p, y=y, hash=hashlib.md5(raw).hexdigest(), time=time.strftime("%H:%M:%S"))


@st.cache_data(show_spinner=False, max_entries=8)
def cached_signal(sig_hash: str, _y: np.ndarray) -> dict:
    return U.signal_views(_y)


# ============================================================ grafik
def fig_emotion_dist(M: pd.DataFrame) -> go.Figure:
    c = M["emotion"].value_counts().reindex(U.ORDER).fillna(0).astype(int)
    tot = int(c.sum())
    fig = go.Figure(go.Bar(
        x=[U.emo_label(e) for e in c.index], y=c.values, marker_color=[U.EMO[e][2] for e in c.index],
        text=[f"{n} · {n / tot:.0%}" for n in c.values], textposition="outside",
        hovertemplate="%{x}<br>%{y} rekaman<extra></extra>"))
    style(fig, 330, "Distribusi kelas emosi")
    fig.update_yaxes(title="Jumlah rekaman", range=[0, c.max() * 1.25])
    return fig


def fig_donut(M, col, names, colors, title) -> go.Figure:
    c = M[col].map(lambda v: names.get(v, str(v))).value_counts()
    fig = go.Figure(go.Pie(labels=c.index, values=c.values, hole=0.58, marker=dict(colors=[colors.get(k, PRIMARY) for k in c.index]),
                           textinfo="label+value", hovertemplate="%{label}: %{value} rekaman (%{percent})<extra></extra>"))
    return style(fig, 270, title, showlegend=False)


def fig_count_bar(M, col, title, color, xtitle, extra_hover=None) -> go.Figure:
    c = M[col].value_counts().sort_index()
    fig = go.Figure(go.Bar(x=[str(i) for i in c.index], y=c.values, marker_color=color, text=c.values, textposition="outside",
                           hovertemplate=f"{xtitle} %{{x}}<br>%{{y}} rekaman<extra></extra>"))
    style(fig, 270, title)
    fig.update_xaxes(title=xtitle, type="category")
    fig.update_yaxes(title="Rekaman", range=[0, c.max() * 1.3])
    return fig


def fig_probs(labels, p) -> go.Figure:
    d = pd.DataFrame({"emo": labels, "p": p}).sort_values("p", ascending=False)
    fig = go.Figure(go.Bar(
        x=d.p, y=[U.emo_label(e) for e in d.emo], orientation="h", marker_color=[U.EMO[e][2] for e in d.emo],
        text=[f"{v:.1%}" for v in d.p], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.2%}<extra></extra>"))
    style(fig, 300, "Probabilitas lima emosi")
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%", title="Probabilitas keluaran model")
    fig.update_yaxes(autorange="reversed")
    return fig


def fig_sent(labels, p) -> go.Figure:
    s = U.sentiment_probs(labels, p)
    fig = go.Figure([go.Bar(x=[v], y=["Sentimen"], orientation="h", name=k.capitalize(), marker_color=U.SENT_COLOR[k],
                            text=f"{k.capitalize()} {v:.0%}" if v >= 0.07 else "", textposition="inside",
                            insidetextfont=dict(color="#fff"), hovertemplate=f"{k.capitalize()}: {v:.2%}<extra></extra>")
                     for k, v in s.items()])
    style(fig, 150, "Probabilitas sentimen (agregasi)", barmode="stack", showlegend=False)
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    return fig


def fig_heatmap(tab: pd.DataFrame, sel_key) -> go.Figure:
    models = [m for m in ["SVM", "RF", "XGBoost", "MLP", "1D-CNN", "BiLSTM", "CNN2D", "MobileNetV2"] if m in set(tab["model"])]
    methods = [m for m in ["mfcc", "mel", "prosodic"] if m in set(tab["metode"])]
    z = tab.pivot_table(index="model", columns="metode", values="f1", aggfunc="mean").reindex(index=models, columns=methods)
    sd = tab.pivot_table(index="model", columns="metode", values="sd", aggfunc="mean").reindex(index=models, columns=methods)
    txt = [[("–" if np.isnan(v) else f"{v:.3f}") for v in row] for row in z.values]
    hov = [[("Tidak ada kombinasi ini" if np.isnan(v) else f"macro-F1 {v:.3f} ± {s:.3f}") for v, s in zip(r1, r2)] for r1, r2 in zip(z.values, sd.values)]
    fig = go.Figure(go.Heatmap(z=z.values, x=[U.METHOD_NAMES[m] for m in methods], y=models, text=txt, texttemplate="%{text}",
                               customdata=hov, hovertemplate="%{y} · %{x}<br>%{customdata}<extra></extra>",
                               colorscale=CREAM_SCALE, zmin=0, zmax=0.8, xgap=3, ygap=3, colorbar=dict(title="macro-F1"), hoverongaps=False))
    if sel_key and sel_key[2] in models and sel_key[0] in methods:
        i, j = models.index(sel_key[2]), methods.index(sel_key[0])
        fig.add_shape(type="rect", x0=j - .5, x1=j + .5, y0=i - .5, y1=i + .5, line=dict(color=TERRA, width=3))
    style(fig, 120 + 48 * len(models), "Rata-rata macro-F1 per model × metode fitur")
    fig.update_yaxes(autorange="reversed", title="Model")
    fig.update_xaxes(title="Metode ekstraksi fitur", side="bottom")
    return fig


def fig_method_avg(tab: pd.DataFrame) -> go.Figure:
    g = tab.groupby("metode").agg(mean=("f1", "mean"), best=("f1", "max"), n=("f1", "size")).reindex([m for m in ["mfcc", "mel", "prosodic"] if m in set(tab["metode"])])
    fig = go.Figure()
    fig.add_bar(x=[U.METHOD_NAMES[m] for m in g.index], y=g["mean"], marker_color=[U.METHOD_COLOR[m] for m in g.index], name="Rata-rata",
                text=[f"{v:.3f}" for v in g["mean"]], textposition="outside", customdata=g["n"],
                hovertemplate="%{x}<br>rata-rata %{y:.3f} dari %{customdata} model<extra></extra>")
    fig.add_scatter(x=[U.METHOD_NAMES[m] for m in g.index], y=g["best"], mode="markers", name="Terbaik", marker=dict(symbol="diamond", size=13, color=INK),
                    hovertemplate="%{x}<br>terbaik %{y:.3f}<extra></extra>")
    style(fig, 330, "Rata-rata & terbaik per metode fitur", legend=dict(orientation="h", y=-0.15))
    fig.update_yaxes(title="macro-F1", range=[0, max(0.8, float(g["best"].max()) * 1.15)])
    return fig


def fig_ranking(t: pd.DataFrame, sel_key) -> go.Figure:
    t = t.sort_values("f1", ascending=False).reset_index(drop=True)
    t["nama"] = t["metode"].map(U.METHOD_NAMES) + " · " + t["model"]
    is_sel = [(r.metode, r.lapis, r.model) == sel_key for r in t.itertuples()]
    fig = go.Figure()
    for m in ["mfcc", "mel", "prosodic"]:
        s = t[t["metode"] == m]
        if s.empty:
            continue
        sel_mask = [is_sel[i] for i in s.index]
        fig.add_bar(y=s["nama"], x=s["f1"], orientation="h", name=U.METHOD_NAMES[m], marker=dict(color=U.METHOD_COLOR[m], line=dict(color=[INK if k else "rgba(0,0,0,0)" for k in sel_mask], width=[3 if k else 0 for k in sel_mask])),
                    error_x=dict(type="data", array=s["sd"], color="rgba(31,36,51,.55)", thickness=1.4, width=4),
                    text=[f"{v:.3f}" + ("  ★ terpilih" if k else "") for v, k in zip(s["f1"], sel_mask)], textposition="outside", cliponaxis=False,
                    customdata=np.stack([s["lapis"], s["sd"], s["acc"], s["ms"]], axis=-1),
                    hovertemplate="%{y}<br>Lapisan: %{customdata[0]}<br>macro-F1: %{x:.4f} ± %{customdata[1]:.4f}<br>"
                                  "Accuracy: %{customdata[2]:.4f}<br>Inferensi: %{customdata[3]:.3f} ms<extra></extra>")
    style(fig, 120 + 30 * len(t), "Peringkat kombinasi berdasarkan macro-F1 (± std antar fold)", legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center"))
    fig.update_yaxes(categoryorder="array", categoryarray=list(t["nama"][::-1]), title="")
    fig.update_xaxes(title="Rata-rata macro-F1 (5 fold)", range=[0, max(0.85, float((t["f1"] + t["sd"]).max()) * 1.25)])
    return fig


def fig_ablation_pair(a: pd.DataFrame) -> go.Figure:
    a = a.copy()
    a["nama"] = a["metode"].map(U.METHOD_NAMES) + " · " + a["model"] + " · " + a["varian"].map(U.VARIAN_NAMES)
    fig = go.Figure()
    fig.add_bar(x=a["nama"], y=a["f1_base"], name="Base", marker_color="#B9BDCB", text=[f"{v:.3f}" for v in a["f1_base"]], textposition="outside")
    fig.add_bar(x=a["nama"], y=a["f1_perlakuan"], name="Dengan perlakuan", marker_color=PRIMARY, text=[f"{v:.3f}" for v in a["f1_perlakuan"]], textposition="outside")
    style(fig, 380, "macro-F1 sebelum dan sesudah perlakuan", barmode="group", legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
    fig.update_yaxes(title="Rata-rata macro-F1", range=[0, max(0.8, float(max(a["f1_base"].max(), a["f1_perlakuan"].max())) * 1.2)])
    fig.update_xaxes(tickangle=-20)
    return fig


def fig_ablation_delta(a: pd.DataFrame) -> go.Figure:
    a = a.sort_values("delta").reset_index(drop=True)
    a["nama"] = a["metode"].map(U.METHOD_NAMES) + " · " + a["model"] + " · " + a["varian"].map(U.VARIAN_NAMES)
    fig = go.Figure(go.Bar(
        y=a["nama"], x=a["delta"], orientation="h", marker_color=[U.SENT_COLOR["positif"] if d > 0 else U.SENT_COLOR["negatif"] for d in a["delta"]],
        text=[f"{d:+.3f} · {u}/{n} fold naik" for d, u, n in zip(a["delta"], a["fold_naik"], a["n_fold"])], textposition="outside", cliponaxis=False,
        name="Rata-rata ΔF1", hovertemplate="%{y}<br>Rata-rata ΔF1: %{x:+.4f}<extra></extra>"))
    xs, ys = [], []
    for r in a.itertuples():
        xs += list(r.delta_per_fold)
        ys += [r.nama] * len(r.delta_per_fold)
    fig.add_scatter(x=xs, y=ys, mode="markers", name="ΔF1 per fold", marker=dict(color=INK, size=7, opacity=.75),
                    hovertemplate="%{y}<br>ΔF1 fold: %{x:+.4f}<extra></extra>")
    lim = max(0.05, float(np.max(np.abs(xs))) * 1.5) if xs else 0.1
    style(fig, 130 + 62 * len(a), "ΔF1 = F1(perlakuan) − F1(base), pada fold yang sama", legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_xaxes(title="Δ macro-F1", range=[-lim, lim * 1.6])
    return fig


def fig_confusion(cm: np.ndarray, labels, display, norm: bool, title: str) -> go.Figure:
    z = U.row_normalize(cm) if norm else cm
    txt = [[(f"{z[i, j]:.0%}<br>({cm[i, j]})" if norm else f"{cm[i, j]}") for j in range(len(labels))] for i in range(len(labels))]
    fig = go.Figure(go.Heatmap(
        z=z, x=display, y=display, text=txt, texttemplate="%{text}", colorscale=CREAM_SCALE, xgap=3, ygap=3,
        zmin=0, zmax=1 if norm else None, colorbar=dict(title="Proporsi baris" if norm else "Jumlah"),
        customdata=cm, hovertemplate="Aktual: %{y}<br>Prediksi: %{x}<br>Jumlah: %{customdata}<extra></extra>"))
    style(fig, 420, title)
    fig.update_xaxes(title="Predicted Emotion", side="bottom")
    fig.update_yaxes(title="Actual Emotion", autorange="reversed")
    return fig


def fig_per_class(pm: pd.DataFrame, display) -> go.Figure:
    fig = go.Figure()
    for col, name, color in (("precision", "Precision", PRIMARY), ("recall", "Recall", TERRA), ("f1", "F1-score", AMBER)):
        fig.add_bar(x=display, y=pm[col], name=name, marker_color=color, text=[f"{v:.2f}" for v in pm[col]], textposition="outside",
                    hovertemplate="%{x}<br>" + name + ": %{y:.3f}<extra></extra>")
    style(fig, 420, "Precision, recall, dan F1 per kelas (out-of-fold)", barmode="group", legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
    fig.update_yaxes(range=[0, 1.1], title="Skor")
    return fig


def fig_confusion_pairs(tc: pd.DataFrame, is_sent: bool) -> go.Figure:
    nm = (lambda e: e.capitalize()) if is_sent else U.emo_name
    names = [f"{nm(a)} → {nm(b)}" for a, b in zip(tc["aktual"], tc["prediksi"])]
    fig = go.Figure(go.Bar(y=names, x=tc["jumlah"], orientation="h", marker_color=TERRA, text=[f"{n}× ({p:.0%} dari kelas aktual)" for n, p in zip(tc["jumlah"], tc["persen_baris"])],
                           textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x} rekaman<extra></extra>"))
    style(fig, 120 + 46 * len(tc), "Pasangan yang paling sering tertukar (Aktual → Prediksi)")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Jumlah rekaman", range=[0, float(tc["jumlah"].max()) * 1.5])
    return fig


def fig_subgroup(s: pd.DataFrame, key: str, title: str, overall: float, color: str, labels=None) -> go.Figure:
    x = [labels.get(v, str(v)) for v in s[key]] if labels else [str(v) for v in s[key]]
    fig = go.Figure(go.Bar(x=x, y=s["akurasi"], marker_color=color, text=[f"{a:.0%}" for a in s["akurasi"]], textposition="outside",
                           customdata=np.stack([s["benar"], s["n"]], axis=-1),
                           hovertemplate="%{x}<br>Akurasi %{y:.1%} (%{customdata[0]}/%{customdata[1]} benar)<extra></extra>"))
    fig.add_hline(y=overall, line_dash="dash", line_color=INK, annotation_text=f"Keseluruhan {overall:.1%}", annotation_position="top right")
    style(fig, 340, title)
    fig.update_xaxes(type="category", title=key.capitalize() if not labels else "")
    fig.update_yaxes(range=[0, 1.12], tickformat=".0%", title="Akurasi (OOF)")
    return fig


def fig_wave(v) -> go.Figure:
    step = max(1, len(v["y"]) // 6000)
    fig = go.Figure(go.Scatter(x=v["t_wave"][::step], y=v["y"][::step], mode="lines", line=dict(width=1, color=PRIMARY), hovertemplate="%{x:.2f} dtk<br>amplitudo %{y:.3f}<extra></extra>"))
    style(fig, 280, "Waveform")
    fig.update_xaxes(title="Waktu (detik)")
    fig.update_yaxes(title="Amplitudo")
    return fig


def fig_mel(v) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=v["mel"], x=v["t_mel"], y=v["mel_freqs"], colorscale=[[0, "#1F2433"], [.35, PRIMARY], [.65, TERRA], [.85, "#F4C95D"], [1, "#FAF5EC"]],
                               colorbar=dict(title="dB"), hovertemplate="%{x:.2f} dtk · %{y:.0f} Hz<br>%{z:.1f} dB<extra></extra>"))
    style(fig, 340, "Log-Mel spectrogram")
    fig.update_xaxes(title="Waktu (detik)")
    fig.update_yaxes(title="Frekuensi (Hz, skala Mel)")
    return fig


def fig_pitch(v) -> go.Figure:
    fig = go.Figure(go.Scatter(x=v["t"], y=v["f0"], mode="lines+markers", connectgaps=False, line=dict(color=TERRA, width=2), marker=dict(size=4),
                               hovertemplate="%{x:.2f} dtk<br>F0 %{y:.0f} Hz<extra></extra>"))
    style(fig, 300, "Kontur pitch (F0)")
    fig.update_xaxes(title="Waktu (detik)")
    fig.update_yaxes(title="F0 (Hz)")
    return fig


def fig_rms(v) -> go.Figure:
    fig = go.Figure(go.Scatter(x=v["t"], y=v["rms"], mode="lines", fill="tozeroy", line=dict(color=AMBER, width=2), fillcolor="rgba(217,142,4,.18)",
                               hovertemplate="%{x:.2f} dtk<br>RMS %{y:.4f}<extra></extra>"))
    style(fig, 300, "Energi (RMS)")
    fig.update_xaxes(title="Waktu (detik)")
    fig.update_yaxes(title="RMS")
    return fig


# ============================================================ komponen halaman
def wave_svg() -> str:
    rng = np.random.default_rng(11)
    n = 46
    xs = np.arange(n)
    env = 0.25 + 0.75 * np.abs(np.sin(xs / 5.2)) * (0.55 + 0.45 * np.sin(xs / 2.1 + 1) ** 2)
    hs = np.clip(env * (0.7 + 0.3 * rng.random(n)), 0.08, 1) * 80
    cols = [PRIMARY, TERRA, AMBER]
    bars = "".join(f"<rect x='{10 + i * 8}' y='{100 - h / 2:.1f}' width='4.5' height='{h:.1f}' rx='2.2' fill='{cols[i % 3]}' opacity='{0.55 + 0.45 * (h / 80):.2f}'/>" for i, h in enumerate(hs))
    return f"<svg width='390' height='200' viewBox='0 0 390 200' xmlns='http://www.w3.org/2000/svg'><circle cx='195' cy='100' r='92' fill='rgba(255,255,255,.55)'/>{bars}</svg>"


def sidebar(D) -> None:
    pages = ["Overview", "Emotion Prediction", "Model Performance", "Error Analysis", "Dataset & Methodology"]
    icons = {"Overview": "🏠", "Emotion Prediction": "🎤", "Model Performance": "📊", "Error Analysis": "🔍", "Dataset & Methodology": "📚"}
    with st.sidebar:
        html("<div class='brand'>🎙️ Suara &amp; Emosi</div><div class='brand-tag'>Listen. Understand. Explore.</div>")
        cur = st.session_state["page"]
        for p in pages:
            st.button(f"{icons[p]}  {p}", key=f"nav_{p}", on_click=go_page, args=(p,), type="primary" if p == cur else "secondary")
        st.markdown("---")
        sel = D["sel"]
        if sel:
            html(f"<div class='side-note'><b>Model aktif</b><br>{U.METHOD_NAMES.get(sel['metode'], sel['metode'])} + {sel['model']}<br>macro-F1 {sel['f1']:.4f} ± {sel['f1_std']:.4f}</div>")
        html("<div class='side-note' style='margin-top:.8rem'>Klasifikasi emosi dari intonasi suara · IndoWaveSentiment · validasi silang 5 fold berbasis aktor.</div>")


# ============================================================ HALAMAN 1 — OVERVIEW
def page_overview(D) -> None:
    M, sel, L, R = D["dataset_meta"], D["sel"], D["leaderboard"], D["results_all"]
    html(f"""<div class='hero'><div>
    <div class='kicker'>Speech Emotion Recognition · IndoWaveSentiment</div>
    <h1>Listen Beyond Words.</h1>
    <div class='sub'>Explore how vocal patterns reveal emotional expressions through machine learning.</div>
    <div class='desc'>Penelitian ini mengklasifikasikan lima emosi (netral, senang, terkejut, jijik, kecewa) dari intonasi suara berbahasa Indonesia. Tiga metode ekstraksi fitur dan beberapa keluarga model dibandingkan dengan validasi silang 5 fold berbasis aktor.</div>
    </div>{wave_svg()}</div>""")
    st.button("Coba prediksi emosi  →", key="cta_pred", type="primary", on_click=go_page, args=("Emotion Prediction",))

    n_rec = len(M) if M is not None else None
    n_act = int(M["actor"].nunique()) if M is not None else None
    n_cls = int(M["emotion"].nunique()) if M is not None else None
    sel_txt = f"{U.METHOD_NAMES.get(sel['metode'], sel['metode'])} + {sel['model']}" if sel else "Tidak tersedia"
    c = st.columns(4)
    cards = [("Total Recordings", f"{n_rec}" if n_rec else "–", "file audio WAV", PRIMARY), ("Number of Actors", f"{n_act}" if n_act else "–", "5 laki-laki · 5 perempuan", TERRA),
             ("Emotion Classes", f"{n_cls}" if n_cls else "–", "netral, senang, terkejut, jijik, kecewa", AMBER), ("Selected Model", sel_txt, "berdasarkan macro-F1 tertinggi", INK)]
    for col, (a, b, n, k) in zip(c, cards):
        col.markdown(metric_card(a, b, n, k).replace("2.2rem", "1.7rem") if a == "Selected Model" else metric_card(a, b, n, k), unsafe_allow_html=True)

    # Emotion distribution
    section("Dataset", "Emotion Distribution", "Setiap emosi memiliki jumlah rekaman yang sama sehingga dataset seimbang antarkelas.")
    if need(M, "dataset_meta.csv"):
        c1, c2 = st.columns([3, 2])
        with c1:
            show(fig_emotion_dist(M), "ov_emo")
        with c2:
            cnt = M["emotion"].value_counts()
            show_df = pd.DataFrame({"Emosi": [U.emo_label(e) for e in U.ORDER], "Jumlah": [int(cnt.get(e, 0)) for e in U.ORDER],
                                    "Persentase": [100 * cnt.get(e, 0) / len(M) for e in U.ORDER]})
            table(show_df, "ov_emo_tbl", column_config={"Persentase": st.column_config.ProgressColumn("Persentase", format="%.1f%%", min_value=0, max_value=100)})
            if cnt.nunique() == 1:
                insight(f"Dataset **seimbang**: tiap emosi {int(cnt.iloc[0])} rekaman, sehingga accuracy dan macro-F1 tidak bias oleh kelas mayoritas.")

        section("Dataset", "Dataset Composition", "Komposisi rekaman berdasarkan gender, intensitas, aktor, dan fold validasi silang.")
        a, b = st.columns(2)
        with a:
            show(fig_donut(M, "gender", U.GENDER_NAMES, {"Laki-laki": PRIMARY, "Perempuan": AMBER}, "Gender"), "ov_gender")
        with b:
            show(fig_donut(M, "intensity", {"normal": "Normal", "strong": "Strong"}, {"Normal": PRIMARY, "Strong": TERRA}, "Intensitas"), "ov_int")
        a, b = st.columns(2)
        with a:
            show(fig_count_bar(M, "actor", "Rekaman per aktor", PRIMARY, "Aktor"), "ov_actor")
        with b:
            show(fig_count_bar(M, "fold", "Rekaman per fold", TERRA, "Fold"), "ov_fold")
        insight("Komposisi merata: 30 rekaman per aktor (5 emosi × 2 intensitas × 3 pengulangan), dan 60 rekaman per fold karena setiap fold berisi dua aktor.")

    # Model at a glance
    section("Model", "Model at a Glance")
    if sel:
        lb = U.leaderboard_table(L, R)
        c1, c2 = st.columns([1.1, 1.4])
        with c1:
            html(f"""<div class='mcard'><span class='badge'>Selected Model</span>
            <div class='val' style='color:{PRIMARY};margin-top:.5rem'>{U.METHOD_NAMES.get(sel['metode'], sel['metode'])} + {sel['model']}</div>
            <div class='note'>Lapisan: {U.LAYER_NAMES.get(sel['lapis'], sel['lapis'])} · Varian: Base (tanpa denoising)</div>
            <div style='display:flex;gap:1.4rem;margin-top:1rem'>
            <div><div class='lab'>Macro-F1</div><div class='val' style='font-size:1.7rem'>{sel['f1']:.4f}</div></div>
            <div><div class='lab'>Accuracy</div><div class='val' style='font-size:1.7rem'>{sel['acc']:.4f}</div></div>
            <div><div class='lab'>Std antar fold</div><div class='val' style='font-size:1.7rem'>{sel['f1_std']:.4f}</div></div></div></div>""")
        with c2:
            if lb is not None:
                top = lb.head(6).copy()
                top["nama"] = top["metode"].map(U.METHOD_NAMES) + " · " + top["model"]
                fig = go.Figure(go.Bar(y=top["nama"], x=top["f1"], orientation="h", marker_color=[U.METHOD_COLOR[m] for m in top["metode"]],
                                       error_x=dict(type="data", array=top["sd"], color="rgba(31,36,51,.5)"), text=[f"{v:.3f}" for v in top["f1"]], textposition="outside", cliponaxis=False))
                style(fig, 290, "Enam kombinasi teratas (macro-F1 ± std)")
                fig.update_yaxes(autorange="reversed")
                fig.update_xaxes(range=[0, 0.95], title="macro-F1")
                show(fig, "ov_top")
        if lb is not None and len(lb) > 1:
            first, second = lb.iloc[0], lb.iloc[1]
            gap = float(first["f1"] - second["f1"])
            why = (f"Dari **{len(lb)}** kombinasi metode × model, **{U.METHOD_NAMES.get(first['metode'])} + {first['model']}** memiliki macro-F1 rata-rata tertinggi "
                   f"(**{first['f1']:.4f}**), unggul **{gap:.4f}** dari peringkat kedua ({U.METHOD_NAMES.get(second['metode'])} + {second['model']}, {second['f1']:.4f}). ")
            why += ("Namun selisih ini lebih kecil daripada std antar fold, jadi keunggulannya perlu dibaca hati-hati. " if gap < float(first["sd"]) else "")
            why += "Inferensinya juga sangat cepat (rata-rata sekitar 0,1 ms) sehingga cocok untuk dashboard interaktif."
            insight(why)

    # Emotion gallery
    section("Gallery", "Emotion Gallery", "Lima kelas emosi dengan warna identitas yang dipakai konsisten di seluruh dashboard.")
    cols = st.columns(5)
    for col, e in zip(cols, U.ORDER):
        em, nm, clr, ds = U.EMO[e]
        col.markdown(f"<div class='ecard' style='--c:{clr}'><div class='em'>{em}</div><div class='nm'>{nm}</div><div class='lb'>{e} · sentimen {U.SENT[e]}</div><div class='ds'>{ds}</div></div>", unsafe_allow_html=True)
    st.caption("Deskripsi bersifat ringkasan umum untuk memudahkan pembaca, bukan hasil pengukuran dataset.")


# ============================================================ HALAMAN 2 — PREDICTION
def render_result(res: dict) -> None:
    labels, p = res["labels"], res["p"]
    top = labels[int(np.argmax(p))]
    em, nm, clr, _ = U.EMO[top]
    sent = U.SENT[top]
    c1, c2 = st.columns([1, 1.6])
    with c1:
        html(f"""<div class='predcard' style='background:linear-gradient(145deg,{clr},{clr}CC)'>
        <div class='em'>{em}</div><div class='nm'>{nm}</div>
        <div class='cf'>Confidence</div><div class='cfv'>{p.max():.1%}</div>
        <div class='pill'>Sentimen: {sent.capitalize()}</div></div>""")
    with c2:
        show(fig_probs(labels, p), "pred_probs")
        st.caption(f"Total probabilitas lima kelas: {p.sum():.1%}")
    show(fig_sent(labels, p), "pred_sent")
    st.caption("Positif = Senang + Terkejut · Netral = Netral · Negatif = Jijik + Kecewa. Klasifikasi utama tetap 5 emosi; sentimen hanyalah agregasi.")

    level, txt = U.confidence_text(labels, p)
    section("Interpretasi", f"Confidence {level}")
    insight(txt + " Confidence adalah probabilitas keluaran model, **bukan** peluang kebenaran yang sudah terkalibrasi.")
    order = np.argsort(p)[::-1]
    if {labels[order[0]], labels[order[1]]} == {"happy", "surprise"}:
        st.info("Senang dan Terkejut paling sering tertukar pada evaluasi model ini (lihat halaman Error Analysis).")
    if level == "Rendah":
        st.warning("Model kurang yakin. Coba ucapkan kalimat dengan intonasi lebih tegas, atau rekam di tempat yang lebih tenang.")

    # unduh
    row = {"waktu": res["time"], "file": res["name"], "prediksi": top, "confidence": round(float(p.max()), 4), "sentimen": sent}
    row.update({f"p_{l}": round(float(v), 4) for l, v in zip(labels, p)})
    st.download_button("Unduh hasil prediksi (CSV)", pd.DataFrame([row]).to_csv(index=False).encode("utf-8"), file_name="prediksi_emosi.csv", mime="text/csv", key=f"dl_{res['hash']}")

    section("Analisis sinyal", "Audio Signal Analysis", "Eksplorasi karakteristik suara pada rekamanmu.")
    warnbox("Log-Mel dan pitch di sini hanyalah alat **eksplorasi sinyal**. Model final tetap hanya menggunakan fitur **MFCC (+ delta, delta-delta)**, bukan fitur Log-Mel atau prosodik.")
    try:
        with st.spinner("Menghitung pitch dan spektrogram…"):
            v = cached_signal(res["hash"], res["y"])
    except Exception as e:  # noqa: BLE001
        st.warning(f"Analisis sinyal tidak dapat ditampilkan untuk rekaman ini ({type(e).__name__}).")
        return
    m = st.columns(4)
    fmt = lambda x, u: "tidak tersedia" if np.isnan(x) else f"{x:.0f} {u}"  # noqa: E731
    for col, (a, b) in zip(m, [("Durasi", f"{v['dur']:.2f} dtk"), ("Pitch rata-rata", fmt(v["f0_mean"], "Hz")), ("Rentang pitch", fmt(v["f0_range"], "Hz")), ("Bagian bersuara", f"{v['voiced']:.0%}")]):
        col.markdown(metric_card(a, b, "", PRIMARY).replace("2.2rem", "1.5rem"), unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["Waveform", "Log-Mel Spectrogram", "Pitch (F0)", "Energi (RMS)"])
    with t1:
        show(fig_wave(v), "sig_wave")
        st.caption("Amplitudo terhadap waktu: menunjukkan kapan suara muncul, jeda, dan seberapa keras tiap bagian ucapan.")
    with t2:
        show(fig_mel(v), "sig_mel")
        st.caption("Distribusi energi pada frekuensi (skala Mel) dari waktu ke waktu. Warna terang = energi lebih besar; pita horizontal terang menandai harmonik suara.")
    with t3:
        show(fig_pitch(v), "sig_pitch")
        st.caption("Estimasi F0 (pYIN) pada bagian bersuara; celah berarti tidak bersuara atau tidak terdeteksi. Naik-turunnya kontur adalah inti intonasi.")
    with t4:
        show(fig_rms(v), "sig_rms")
        st.caption(f"Energi sesaat (RMS), rata-rata {v['rms_mean']:.4f}. Lonjakan menandai penekanan suku kata; energi yang melemah sering menyertai ucapan datar.")


def page_prediction(D) -> None:
    sel = D["sel"]
    page_header("Emotion Prediction", "Unggah atau rekam suara, lalu lihat emosi yang diprediksi model **MFCC + SVM** beserta distribusi probabilitasnya.")
    html("""<div class='insight'>🎯 <b>Petunjuk:</b> ucapkan kalimat selama sekitar <b>2–4 detik</b>. Model dilatih pada <b>satu kalimat yang sama</b> ("Kualitas HP ini cukup bagus"),
    jadi hasil untuk kalimat lain dapat kurang akurat.</div>""")
    src = st.radio("Sumber audio", ["Unggah file WAV", "Rekam suara langsung", "Contoh dari dataset"], horizontal=True, key="audio_src")
    raw, name = None, None
    if src == "Unggah file WAV":
        f = st.file_uploader("Pilih file .wav", type=["wav"], key="up_wav")
        if f is not None:
            raw, name = f.getvalue(), f.name
    elif src == "Rekam suara langsung":
        if hasattr(st, "audio_input"):
            f = st.audio_input("Tekan untuk merekam", key="rec_wav")
            if f is not None:
                raw, name = f.getvalue(), "rekaman_langsung.wav"
        else:
            st.info("Versi Streamlit ini belum mendukung perekaman langsung. Perbarui Streamlit (≥ 1.40) atau gunakan unggah file.")
    else:
        samples = list_samples()
        if not samples:
            st.info("File contoh audio (neutral.wav, happy.wav, dst.) tidak tersedia.")
        else:
            names = [s.stem for s in samples]
            ch = st.selectbox("Pilih contoh (aktor 01, intensitas normal)", names, format_func=lambda s: U.emo_label(s), key="sample_pick")
            raw, name = samples[names.index(ch)].read_bytes(), f"{ch}.wav"

    if raw is None:
        st.markdown("<p class='side-note' style='margin-top:.6rem'>Belum ada audio. Pilih sumber audio di atas untuk memulai.</p>", unsafe_allow_html=True)
    else:
        st.audio(raw, format="audio/wav")
        cache = st.session_state.setdefault("pred_cache", {})
        h = hashlib.md5(raw).hexdigest()
        if h not in cache:
            try:
                with st.spinner("Menganalisis intonasi suara…"):
                    cache[h] = run_prediction(raw, name)
                while len(cache) > 8:
                    cache.pop(next(iter(cache)))
                hist = st.session_state.setdefault("history", [])
                r = cache[h]
                top = r["labels"][int(np.argmax(r["p"]))]
                row = {"waktu": r["time"], "sumber": src, "file": name, "prediksi": top, "confidence": round(float(r["p"].max()), 4), "sentimen": U.SENT[top]}
                row.update({f"p_{l}": round(float(v), 4) for l, v in zip(r["labels"], r["p"])})
                hist.append(row)
            except AudioError as e:
                st.error(str(e))
                return
        render_result(cache[h])

    hist = st.session_state.get("history", [])
    if hist:
        with st.expander(f"Riwayat prediksi sesi ini ({len(hist)})"):
            H = pd.DataFrame(hist)
            table(H, "hist_tbl")
            a, b = st.columns([1, 1])
            a.download_button("Unduh riwayat (CSV)", H.to_csv(index=False).encode("utf-8"), file_name="riwayat_prediksi.csv", mime="text/csv", key="dl_hist")
            if b.button("Hapus riwayat", key="clr_hist"):
                st.session_state["history"] = []
                st.rerun()


# ============================================================ HALAMAN 3 — PERFORMANCE
def page_performance(D) -> None:
    R, L, AB, sel = D["results_all"], D["leaderboard"], D["ablation_all"], D["sel"]
    page_header("Model Performance", "Perbandingan seluruh kombinasi **metode ekstraksi fitur × model** dengan 5-fold cross-validation berbasis aktor. Metrik utama: **macro-F1**.")
    lb = U.leaderboard_table(L, R)
    if lb is None:
        st.info("Aset hasil evaluasi tidak tersedia.")
        return
    sk = U.selected_key(sel)

    section("A", "Selected Model")
    if sel:
        html(f"""<div class='mcard' style='border-left:6px solid {PRIMARY}'><span class='badge'>Selected Model</span>
        <div style='display:flex;flex-wrap:wrap;gap:2.2rem;margin-top:.7rem;align-items:flex-end'>
        <div><div class='lab'>Metode fitur</div><div class='val' style='font-size:1.6rem'>{U.METHOD_NAMES.get(sel['metode'])}</div></div>
        <div><div class='lab'>Model</div><div class='val' style='font-size:1.6rem'>{sel['model']}</div></div>
        <div><div class='lab'>Macro-F1</div><div class='val' style='font-size:1.6rem;color:{PRIMARY}'>{sel['f1']:.4f}</div></div>
        <div><div class='lab'>Std antar fold</div><div class='val' style='font-size:1.6rem'>{sel['f1_std']:.4f}</div></div>
        <div><div class='lab'>Accuracy</div><div class='val' style='font-size:1.6rem'>{sel['acc']:.4f}</div></div></div></div>""")
    else:
        st.info("Informasi model terpilih (`selection.json`) tidak tersedia.")

    # filter
    f1c, f2c = st.columns(2)
    meth = f1c.multiselect("Filter metode fitur", ["mfcc", "mel", "prosodic"], default=["mfcc", "mel", "prosodic"], format_func=lambda m: U.METHOD_NAMES[m], key="pf_meth")
    lay = f2c.multiselect("Filter jenis model", ["ML", "MLP", "DL-khusus"], default=["ML", "MLP", "DL-khusus"], format_func=lambda m: U.LAYER_NAMES[m], key="pf_lay")
    flt = lb[lb["metode"].isin(meth) & lb["lapis"].isin(lay)]

    section("B", "Feature Extraction Comparison", "Rata-rata macro-F1 tiap model pada setiap metode fitur. Sel kosong (–) berarti kombinasi itu tidak diuji. Kotak terracotta menandai model terpilih.")
    if flt.empty:
        st.warning("Tidak ada kombinasi yang cocok dengan filter. Ubah pilihan metode atau jenis model.")
    else:
        a, b = st.columns([1.3, 1])
        with a:
            show(fig_heatmap(flt, sk), "pf_heat")
        with b:
            show(fig_method_avg(flt), "pf_avg")
        g = flt.groupby("metode")["f1"].agg(["mean", "max", "size"])
        best_m = g["mean"].idxmax()
        insight(f"Pada kombinasi yang ditampilkan, **{U.METHOD_NAMES[best_m]}** memiliki rata-rata macro-F1 tertinggi (**{g.loc[best_m, 'mean']:.3f}**, dari {int(g.loc[best_m, 'size'])} model). "
                "Perhatikan bahwa jumlah model per metode tidak sama (mis. CNN2D dan MobileNetV2 hanya untuk Log-Mel), jadi bandingkan sel pada baris yang sama untuk perbandingan yang adil.")

    section("C", "Model Ranking", "Diurutkan berdasarkan macro-F1. Garis galat = standar deviasi antar fold; warna = metode fitur; ★ = model terpilih.")
    if not flt.empty:
        show(fig_ranking(flt, sk), "pf_rank")
        insight("Garis galat yang saling tumpang-tindih berarti selisih antar model berdekatan belum tentu bermakna, karena hanya ada 5 fold dan 10 aktor.")
        t = flt.copy()
        t["Metode fitur"] = t["metode"].map(U.METHOD_NAMES)
        t["Lapisan model"] = t["lapis"].map(U.LAYER_NAMES)
        t = t.rename(columns={"model": "Algoritma", "f1": "Mean macro-F1", "sd": "Std macro-F1", "acc": "Accuracy", "ms": "Inference time (ms)"})
        t = t[["Metode fitur", "Lapisan model", "Algoritma", "Mean macro-F1", "Std macro-F1", "Accuracy", "Inference time (ms)"]]
        t.insert(0, "Peringkat", range(1, len(t) + 1))
        table(t, "pf_tbl", column_config={"Mean macro-F1": st.column_config.ProgressColumn("Mean macro-F1", min_value=0, max_value=1, format="%.4f"),
                                          "Std macro-F1": st.column_config.NumberColumn(format="%.4f"), "Accuracy": st.column_config.NumberColumn(format="%.4f"),
                                          "Inference time (ms)": st.column_config.NumberColumn(format="%.4f")})
        st.download_button("Unduh tabel ranking (CSV)", t.to_csv(index=False).encode("utf-8"), file_name="ranking_model.csv", mime="text/csv", key="dl_rank")

    section("D", "Evaluation Metrics", "Macro-F1 dipakai sebagai metrik utama karena merata-ratakan F1 setiap kelas dengan bobot sama, sehingga performa pada tiap emosi sama pentingnya.")
    mc = st.columns(5)
    metrics = [("Accuracy", "Proporsi prediksi benar dari seluruh rekaman. Mudah dibaca, tetapi bisa menyesatkan bila kelas tidak seimbang.", True),
               ("Macro-Precision", "Rata-rata precision per kelas: dari semua prediksi suatu emosi, berapa yang benar.", False),
               ("Macro-Recall", "Rata-rata recall per kelas: dari semua rekaman suatu emosi, berapa yang berhasil dikenali.", False),
               ("Macro-F1", "Rata-rata harmonik precision dan recall per kelas, lalu dirata-ratakan. Metrik utama perbandingan.", True),
               ("Inference Time", "Rata-rata waktu prediksi satu rekaman (ms) pada hasil evaluasi.", True)]
    for col, (nm, ds, ok) in zip(mc, metrics):
        extra = "" if ok else "<div class='note' style='color:#D9593D;margin-top:.4rem'>Tidak tersedia di file hasil evaluasi. Precision &amp; recall per kelas dihitung dari prediksi OOF di halaman Error Analysis.</div>"
        col.markdown(f"<div class='mcard'><div class='lab'>{nm}</div><div class='note' style='color:#1F2433;margin-top:.4rem'>{ds}</div>{extra}</div>", unsafe_allow_html=True)

    section("E", "Ablation Study", "Pengaruh denoising dan augmentasi dibanding langkah dasar (Base). Perlakuan hanya diuji pada sebagian kombinasi, sehingga hanya yang tersedia yang ditampilkan.")
    if not need(AB, "ablation_all.csv"):
        return
    S = U.ablation_summary(AB)
    if S.empty:
        st.info("Tidak ada hasil perlakuan selain Base pada `ablation_all.csv`.")
        return
    c1, c2 = st.columns(2)
    am = c1.multiselect("Filter metode fitur (ablasi)", sorted(S["metode"].unique()), default=sorted(S["metode"].unique()), format_func=lambda m: U.METHOD_NAMES[m], key="ab_meth")
    av = c2.multiselect("Filter perlakuan", sorted(S["varian"].unique()), default=sorted(S["varian"].unique()), format_func=lambda v: U.VARIAN_NAMES[v], key="ab_var")
    Sf = S[S["metode"].isin(am) & S["varian"].isin(av)].reset_index(drop=True)
    if Sf.empty:
        st.warning("Tidak ada hasil ablasi untuk filter yang dipilih.")
        return
    a, b = st.columns(2)
    with a:
        show(fig_ablation_pair(Sf), "ab_pair")
    with b:
        show(fig_ablation_delta(Sf), "ab_delta")
    Sf = Sf.assign(Interpretasi=Sf.apply(U.interpret_ablation, axis=1))
    T = pd.DataFrame({"Metode fitur": Sf["metode"].map(U.METHOD_NAMES), "Model": Sf["model"], "Perlakuan": Sf["varian"].map(U.VARIAN_NAMES),
                      "F1 Base": Sf["f1_base"], "F1 Perlakuan": Sf["f1_perlakuan"], "ΔF1 rata-rata": Sf["delta"], "Std ΔF1": Sf["delta_sd"],
                      "Fold naik": Sf["fold_naik"].astype(str) + "/" + Sf["n_fold"].astype(str), "Interpretasi": Sf["Interpretasi"]})
    table(T, "ab_tbl", column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ["F1 Base", "F1 Perlakuan", "ΔF1 rata-rata", "Std ΔF1"]})
    sel_note = ""
    if sel:
        r_ = S[(S["metode"] == sel["metode"]) & (S["model"] == sel["model"]) & (S["lapis"] == sel["lapis"])]
        if not r_.empty:
            r0_ = r_.iloc[0]
            sel_note = (f" Untuk model terpilih, perlakuan **{U.VARIAN_NAMES.get(r0_['varian'], r0_['varian'])}** mengubah macro-F1 sebesar **{r0_['delta']:+.4f}** "
                        f"({int(r0_['fold_naik'])}/{int(r0_['n_fold'])} fold naik), sehingga varian Base tetap dipertahankan.")
    n_cons = int(((Sf["fold_naik"] == Sf["n_fold"]) | (Sf["fold_turun"] == Sf["n_fold"])).sum())
    insight(f"Dari **{len(Sf)}** perlakuan yang ditampilkan, hanya **{n_cons}** yang konsisten di semua fold (naik atau turun di setiap fold). Selebihnya bercampur antar fold. "
            "Dengan hanya 5 fold dan tanpa uji statistik, selisih rata-rata **tidak** boleh dibaca sebagai bukti signifikan." + sel_note)


# ============================================================ HALAMAN 4 — ERROR ANALYSIS
def page_errors(D) -> None:
    O, M, sel = D["oof_all"], D["dataset_meta"], D["sel"]
    page_header("Error Analysis", "Pola kesalahan model berdasarkan prediksi **out-of-fold** (setiap rekaman diprediksi oleh model yang tidak melihat aktornya saat pelatihan).")
    if not need(O, "oof_all.csv"):
        return
    cols = [c for c in O.columns if c.count("|") >= 3]
    if not cols:
        st.info("Kolom prediksi model tidak ditemukan di `oof_all.csv`.")
        return
    default = cols.index(sel["key"]) if sel and sel["key"] in cols else 0
    c1, c2, c3 = st.columns([2.4, 1.6, 1.4])
    col = c1.selectbox("Model (prediksi OOF)", cols, index=default, format_func=U.nice_key, key="er_model")
    mode = c2.radio("Tampilan kelas", ["5 emosi", "3 sentimen"], horizontal=True, key="er_mode")
    norm = c3.checkbox("Normalisasi per baris", value=True, key="er_norm")

    sub = O[["y_true", col]].dropna()
    n_drop = len(O) - len(sub)
    yt, yp = sub["y_true"].values, sub[col].values
    is_sent = mode == "3 sentimen"
    if is_sent:
        yt, yp, labs = U.to_sent(yt), U.to_sent(yp), U.SENT_ORDER
        disp = [s.capitalize() for s in labs]
    else:
        labs = U.ORDER
        disp = [U.emo_label(e) for e in labs]
    if n_drop:
        st.caption(f"{n_drop} baris dengan nilai kosong dikeluarkan.")
    acc = float((yt == yp).mean()) if len(yt) else float("nan")

    section("A", "Confusion Matrix", "Baris = emosi sebenarnya, kolom = emosi yang diprediksi. Diagonal adalah prediksi benar.")
    cm = U.confusion(yt, yp, labs)
    a, b = st.columns([1.5, 1])
    with a:
        show(fig_confusion(cm, labs, disp, norm, f"{U.nice_key(col)} · {mode}"), "er_cm")
    with b:
        html(metric_card("Akurasi gabungan (OOF)", f"{acc:.1%}", f"{int((yt == yp).sum())} dari {len(yt)} rekaman benar", PRIMARY))
        rc = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(len(labs)), where=cm.sum(1) > 0)
        best, worst = int(np.argmax(rc)), int(np.argmin(rc))
        insight(f"Kelas paling mudah dikenali: **{disp[best]}** (recall {rc[best]:.0%}). Paling sulit: **{disp[worst]}** (recall {rc[worst]:.0%}).")
        st.caption("Pada mode 3 sentimen, kesalahan antar emosi dalam kelompok sentimen yang sama (mis. Senang ↔ Terkejut) tidak lagi dihitung sebagai salah.")

    section("B", "Classification Performance by Emotion" if not is_sent else "Classification Performance by Sentiment", "Precision, recall, dan F1 dihitung dari prediksi out-of-fold terhadap label sebenarnya.")
    pm = U.per_class_metrics(yt, yp, labs)
    show(fig_per_class(pm, disp), "er_pc")
    tbl = pm.assign(kelas=disp).rename(columns={"kelas": "Kelas", "precision": "Precision", "recall": "Recall", "f1": "F1-score", "support": "Jumlah rekaman"})
    table(tbl, "er_pc_tbl", column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["Precision", "Recall", "F1-score"]})
    insight(f"Macro-F1 gabungan dari prediksi OOF: **{pm['f1'].mean():.3f}**. Nilai ini dihitung dari seluruh 300 prediksi sekaligus, sehingga dapat sedikit berbeda dari rata-rata macro-F1 per fold di halaman Model Performance.")

    section("C", "Most Frequent Misclassifications", "Dihitung langsung dari matriks kebingungan di atas, bukan diasumsikan.")
    tc = U.top_confusions(yt, yp, labs, k=8)
    if tc.empty:
        st.success("Tidak ada kesalahan klasifikasi pada tampilan ini.")
    else:
        a, b = st.columns([1.5, 1])
        with a:
            show(fig_confusion_pairs(tc, is_sent), "er_pairs")
        with b:
            nm = (lambda e: e.capitalize()) if is_sent else U.emo_label
            t2 = pd.DataFrame({"Aktual → Prediksi": [f"{nm(a_)} → {nm(b_)}" for a_, b_ in zip(tc["aktual"], tc["prediksi"])], "Jumlah": tc["jumlah"], "% dari kelas aktual": 100 * tc["persen_baris"]})
            table(t2, "er_pairs_tbl", column_config={"% dari kelas aktual": st.column_config.NumberColumn(format="%.1f%%")})
        r0 = tc.iloc[0]
        insight(f"Kesalahan terbanyak pada model ini: **{(nm if not is_sent else (lambda e: e.capitalize()))(r0['aktual'])} → {(nm if not is_sent else (lambda e: e.capitalize()))(r0['prediksi'])}** ({int(r0['jumlah'])} rekaman, {r0['persen_baris']:.0%} dari kelas aktualnya).")

    section("D", "Performance by Subgroup", "Akurasi OOF berdasarkan gender, intensitas, aktor, dan fold, digabung lewat `filename` dengan metadata dataset.")
    if not need(M, "dataset_meta.csv"):
        return
    try:
        dd = U.merge_oof_meta(O, M, col)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Penggabungan metadata dengan prediksi gagal ({type(e).__name__}).")
        return
    overall = float(dd["benar"].mean())
    st.caption(f"Akurasi 5 emosi (seluruh data) untuk model ini: {overall:.1%}. Bagian ini selalu memakai 5 kelas emosi.")
    tabs = st.tabs(["Gender", "Intensitas", "Aktor", "Fold"])
    spec = [("gender", "Akurasi per gender", PRIMARY, U.GENDER_NAMES), ("intensity", "Akurasi per intensitas", TERRA, {"normal": "Normal", "strong": "Strong"}),
            ("actor", "Akurasi per aktor", AMBER, None), ("fold", "Akurasi per fold", PRIMARY_D, None)]
    for tab, (k, ttl, clr, lab) in zip(tabs, spec):
        with tab:
            s = U.subgroup_accuracy(dd, k)
            a, b = st.columns([1.5, 1])
            with a:
                show(fig_subgroup(s, k, ttl, overall, clr, lab), f"er_sub_{k}")
            with b:
                t3 = s.assign(**{k: [lab.get(v, str(v)) if lab else str(v) for v in s[k]]}).rename(columns={k: k.capitalize(), "akurasi": "Akurasi (%)", "benar": "Benar", "n": "Jumlah"})
                t3["Akurasi (%)"] = 100 * t3["Akurasi (%)"]
                table(t3, f"er_sub_tbl_{k}", column_config={"Akurasi (%)": st.column_config.NumberColumn(format="%.1f%%")})
            if len(s) > 1:
                hi, lo = s.loc[s["akurasi"].idxmax()], s.loc[s["akurasi"].idxmin()]
                f_ = (lambda v: lab.get(v, str(v))) if lab else str
                insight(f"Tertinggi: **{f_(hi[k])}** ({hi['akurasi']:.0%}); terendah: **{f_(lo[k])}** ({lo['akurasi']:.0%}). Rentang {hi['akurasi'] - lo['akurasi']:.0%}.")
    warnbox("Analisis subgroup bersifat **deskriptif**. Setiap subgroup hanya berisi puluhan rekaman dan dataset hanya memiliki 10 aktor, jadi perbedaan antar subgroup tidak boleh dibaca sebagai efek yang pasti.")


# ============================================================ HALAMAN 5 — DATASET & METHODOLOGY
def page_dataset(D) -> None:
    M = D["dataset_meta"]
    page_header("Dataset & Methodology", "Penjelasan dataset IndoWaveSentiment, metode ekstraksi fitur, rancangan eksperimen, dan keterbatasan penelitian.")

    section("A", "Dataset Overview")
    c = st.columns(5)
    for col, (a, b, n) in zip(c, [("Rekaman", "300", "file audio"), ("Aktor", "10", "5 L · 5 P"), ("Kelas emosi", "5", "diperankan"), ("Intensitas", "2", "normal & strong"), ("Pengulangan", "3×", "per kombinasi")]):
        col.markdown(metric_card(a, b, n, PRIMARY).replace("2.2rem", "1.8rem"), unsafe_allow_html=True)
    st.markdown("")
    st.markdown("Rekaman dilakukan dalam **kondisi studio**. Setiap aktor mengucapkan kalimat yang sama, *“Kualitas HP ini cukup bagus”*, dengan 5 emosi × 2 intensitas × 3 pengulangan = 30 rekaman per aktor.")
    insight("Karena kalimatnya selalu sama, model terutama mempelajari **karakteristik intonasi dan ekspresi vokal**, bukan memahami makna kalimat yang berbeda-beda.")
    samples = list_samples()
    if samples:
        with st.expander("🎧 Contoh audio (aktor 01, intensitas normal)", expanded=False):
            cc = st.columns(len(samples))
            for w, s in zip(cc, samples):
                w.markdown(f"**{U.emo_label(s.stem)}**")
                w.audio(s.read_bytes(), format="audio/wav")

    section("B", "Dataset Composition")
    if need(M, "dataset_meta.csv"):
        a, b = st.columns(2)
        with a:
            show(fig_emotion_dist(M), "ds_emo")
        with b:
            show(fig_donut(M, "intensity", {"normal": "Normal", "strong": "Strong"}, {"Normal": PRIMARY, "Strong": TERRA}, "Intensitas"), "ds_int")
        a, b, c3 = st.columns(3)
        with a:
            show(fig_donut(M, "gender", U.GENDER_NAMES, {"Laki-laki": PRIMARY, "Perempuan": AMBER}, "Gender"), "ds_gender")
        with b:
            show(fig_count_bar(M, "actor", "Rekaman per aktor", PRIMARY, "Aktor"), "ds_actor")
        with c3:
            show(fig_count_bar(M, "fold", "Rekaman per fold", TERRA, "Fold"), "ds_fold")
        insight("Seluruh distribusi seimbang, sehingga perbedaan performa antarkelas atau subgroup tidak berasal dari jumlah data yang timpang.")

    section("C", "Feature Extraction Methods")
    c = st.columns(3)
    meth = [("MFCC", PRIMARY, "Menggambarkan karakteristik spektral suara berdasarkan persepsi pendengaran manusia. Digunakan dalam bentuk **MFCC + delta + delta-delta**. Pada model final: 40 koefisien → 120 baris, diringkas dengan mean, std, min, max menjadi **480 fitur**."),
            ("Log-Mel Spectrogram", TERRA, "Merepresentasikan distribusi energi frekuensi dalam skala Mel dari waktu ke waktu. Digunakan sebagai representasi input untuk model berbasis citra atau urutan (CNN2D, MobileNetV2) maupun vektor ringkasan untuk ML."),
            ("Prosodic Features", AMBER, "Mencakup karakteristik pitch (F0), energi (RMS), dan fitur akustik lain (mis. ZCR, spectral centroid/rolloff/contrast, chroma). Membantu menggambarkan ekspresi vokal.")]
    for col, (t, k, d) in zip(c, meth):
        col.markdown(f"<div class='ecard' style='--c:{k};text-align:left'><div class='nm'>{t}</div><div class='ds' style='margin-top:.5rem;color:#1F2433'>{bold(d)}</div></div>", unsafe_allow_html=True)
    with st.expander("Alur preprocessing & ekstraksi pada model final (MFCC + SVM)"):
        st.markdown("1. Audio dimuat sebagai **mono, 16 kHz**.\n2. *Silence trimming* dengan `top_db = 30`.\n3. Di-*pad*/*crop* menjadi **3 detik**.\n4. Ekstraksi **40 MFCC**, delta, dan delta-delta (n_fft 1024, hop 512).\n5. Statistik **mean, std, min, max** → 480 fitur.\n6. Standardisasi dengan *scaler* dari model.\n7. SVM menghasilkan probabilitas lima emosi (`predict_proba`).\n\nProses ini memakai kode yang sama dengan saat pelatihan (`src/preprocess_mfcc.py`) agar tidak terjadi *training-serving skew*.")

    section("D", "Model Architecture", "Alur penelitian dari data mentah hingga prediksi akhir.")
    steps = [("Audio Dataset", "300 rekaman"), ("Preprocessing", "mono, 16 kHz, trim, 3 dtk"), ("Feature Extraction", "MFCC · Log-Mel · Prosodik"), ("Classification Models", "ML · MLP · DL khusus"),
             ("5-Fold CV", "berbasis aktor"), ("Model Selection", "macro-F1 tertinggi"), ("Final Prediction", "MFCC + SVM")]
    parts = []
    for i, (t, d) in enumerate(steps, 1):
        parts.append(f"<div class='step'><div class='n'>{i}</div><div class='t'>{t}</div><div class='d'>{d}</div></div>")
        if i < len(steps):
            parts.append("<div class='arrow'>→</div>")
    html("<div class='flow'>" + "".join(parts) + "</div>")
    c = st.columns(3)
    layers = [("Machine Learning", PRIMARY, "SVM, Random Forest, XGBoost pada vektor fitur ringkasan. Cepat dan efektif pada data kecil."),
              ("MLP", TERRA, "Multi-Layer Perceptron dengan arsitektur identik untuk ketiga metode fitur, sehingga perbandingan fitur lebih adil."),
              ("Deep Learning khusus", AMBER, "Arsitektur menyesuaikan representasi: 1D-CNN dan BiLSTM untuk urutan (MFCC/prosodik), CNN2D dan MobileNetV2 untuk Log-Mel sebagai citra.")]
    for col, (t, k, d) in zip(c, layers):
        col.markdown(f"<div class='ecard' style='--c:{k};text-align:left'><div class='nm'>{t}</div><div class='ds' style='margin-top:.5rem;color:#1F2433'>{d}</div></div>", unsafe_allow_html=True)

    section("E", "Experimental Design", "Evaluasi memakai **5-fold cross-validation berbasis aktor**, bukan pembagian acak per rekaman.")
    if need(M, "dataset_meta.csv"):
        fd = M.groupby("fold").agg(Aktor=("actor", lambda s: ", ".join(str(a) for a in sorted(s.unique()))), Rekaman=("filename", "size"),
                                   L=("gender", lambda s: int((s == "L").sum())), P=("gender", lambda s: int((s == "P").sum()))).reset_index().rename(columns={"fold": "Fold", "L": "Rekaman laki-laki", "P": "Rekaman perempuan"})
        a, b = st.columns([1, 1.2])
        with a:
            table(fd, "ds_fold_tbl")
        with b:
            insight(f"**{len(M)}** rekaman dibagi menjadi **{M['fold'].nunique()}** fold, masing-masing **{int(fd['Rekaman'].iloc[0])}** rekaman dari 2 aktor. Pada setiap putaran, satu fold menjadi data uji dan empat fold lain menjadi data latih.")
            st.markdown("**Mengapa berbasis aktor?** Rekaman dari aktor yang sama sangat mirip. Bila dibagi acak, model bisa 'mengenali aktor' dan bukan emosinya, sehingga skor terlalu optimis. Dengan memisahkan aktor, evaluasi mengukur kemampuan **generalisasi ke aktor baru** yang tidak dipakai saat pelatihan pada fold terkait.")

    section("F", "Limitations", "Keterbatasan penelitian dicantumkan secara transparan.")
    st.markdown("- Hanya **10 aktor**, sehingga variasi suara terbatas.\n- Dataset hanya memakai **satu kalimat**.\n- Emosi merupakan **ekspresi yang diperankan**, bukan emosi spontan.\n- Ada potensi kebingungan antara **happy** dan **surprise**.\n- Hasil belum tentu dapat digeneralisasikan ke **percakapan alami** atau populasi umum.")
    warnbox("Sistem ini **tidak boleh** digunakan untuk menyimpulkan kondisi psikologis atau emosi seseorang secara definitif. Keluaran model hanyalah estimasi probabilistik atas pola intonasi.")
    st.caption("Sumber dataset: Bustamin, A., Rizky, A.M., Warni, E., Areni, I.S., Indrabayu (2024). IndoWaveSentiment: Indonesian audio dataset for emotion classification. Data in Brief 57, 111138. DOI 10.1016/j.dib.2024.111138 · Lisensi CC BY-NC 4.0 (non-komersial).")


# ============================================================ main
def main() -> None:
    st.session_state.setdefault("page", "Overview")
    D = load_data()
    sidebar(D)
    page = st.session_state["page"]
    {"Overview": page_overview, "Emotion Prediction": page_prediction, "Model Performance": page_performance,
     "Error Analysis": page_errors, "Dataset & Methodology": page_dataset}[page](D)


main()
