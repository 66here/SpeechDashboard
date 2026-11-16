"""Fungsi bantu Suara & Emosi: konstanta, pemuatan aset, metrik, dan pengolahan sinyal.

Tidak bergantung pada Streamlit, sehingga bisa diuji terpisah.
Semua angka diturunkan dari CSV asli di assets/; tidak ada nilai buatan.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SR = 16000
ORDER = ["neutral", "happy", "surprise", "disgust", "disappointed"]
SENT_ORDER = ["positif", "netral", "negatif"]

# label -> (emoji, nama, warna identitas, deskripsi singkat)
EMO = {
    "neutral": ("😐", "Netral", "#8A8FA3", "Nada datar dan stabil, tanpa penekanan emosional yang kuat."),
    "happy": ("😄", "Senang", "#D98E04", "Nada cenderung naik, ceria, dan bertenaga."),
    "surprise": ("😲", "Terkejut", "#D9593D", "Lonjakan pitch tiba-tiba dengan penekanan yang tajam."),
    "disgust": ("🤢", "Jijik", "#4E8B6A", "Nada menekan dan tertahan, dengan artikulasi yang tegas."),
    "disappointed": ("😞", "Kecewa", "#3B4BB8", "Nada menurun, lebih lambat, dan energi melemah."),
}
SENT = {"neutral": "netral", "happy": "positif", "surprise": "positif", "disgust": "negatif", "disappointed": "negatif"}
SENT_COLOR = {"positif": "#2F8F83", "netral": "#8A8FA3", "negatif": "#B5473B"}
METHOD_NAMES = {"mfcc": "MFCC", "mel": "Log-Mel", "prosodic": "Prosodik"}
METHOD_COLOR = {"mfcc": "#3B4BB8", "mel": "#D9593D", "prosodic": "#D98E04"}
LAYER_NAMES = {"ML": "Machine Learning", "MLP": "MLP", "DL-khusus": "Deep Learning khusus"}
VARIAN_NAMES = {"base": "Base", "denoise": "Denoising", "aug": "Augmentation"}
GENDER_NAMES = {"L": "Laki-laki", "P": "Perempuan"}


def emo_name(e: str) -> str:
    return EMO[e][1] if e in EMO else str(e)


def emo_label(e: str) -> str:
    return f"{EMO[e][0]} {EMO[e][1]}" if e in EMO else str(e)


# ------------------------------------------------------------------ aset
def read_csv_safe(path: Path):
    try:
        return pd.read_csv(path) if Path(path).exists() else None
    except Exception:
        return None


def load_assets(assets_dir: Path) -> dict:
    d = {n: read_csv_safe(Path(assets_dir) / f"{n}.csv")
         for n in ("results_all", "ablation_all", "leaderboard", "dataset_meta", "oof_all")}
    try:
        p = Path(assets_dir) / "selection.json"
        d["sel"] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:
        d["sel"] = None
    return d


def selected_key(sel):
    """(metode, lapis, model) model terpilih, atau None."""
    return (sel["metode"], sel["lapis"], sel["model"]) if sel else None


# ------------------------------------------------------------------ hasil evaluasi
def summarize_results(R: pd.DataFrame) -> pd.DataFrame:
    """Rata-rata antar fold untuk varian base. std = std antar fold (ddof=1)."""
    b = R[R["varian"] == "base"]
    g = (b.groupby(["metode", "lapis", "model"], as_index=False)
         .agg(f1=("macro_f1", "mean"), sd=("macro_f1", "std"), acc=("accuracy", "mean"),
              ms=("infer_time_ms", "mean"), n_fold=("fold", "nunique")))
    g["sd"] = g["sd"].fillna(0.0)
    return g.sort_values("f1", ascending=False).reset_index(drop=True)


def leaderboard_table(L, R):
    """Ranking: utamakan leaderboard.csv; jika tidak ada, turunkan dari results_all.csv."""
    if L is not None and {"metode", "lapis", "model", "f1", "f1_std", "acc", "ms"} <= set(L.columns):
        t = L.rename(columns={"f1_std": "sd"}).copy()
        return t.sort_values("f1", ascending=False).reset_index(drop=True)
    if R is not None:
        return summarize_results(R)
    return None


def ablation_summary(AB: pd.DataFrame) -> pd.DataFrame:
    """ΔF1 = F1(perlakuan) − F1(base), dihitung pada fold yang sama."""
    rows = []
    for (m, lp, mod), g in AB.groupby(["metode", "lapis", "model"]):
        base = g[g["varian"] == "base"].set_index("fold")["macro_f1"]
        for v in [x for x in g["varian"].unique() if x != "base"]:
            tr = g[g["varian"] == v].set_index("fold")["macro_f1"]
            common = base.index.intersection(tr.index)
            if len(common) == 0:
                continue
            d = tr.loc[common] - base.loc[common]
            rows.append(dict(metode=m, lapis=lp, model=mod, varian=v, n_fold=len(common),
                             f1_base=float(base.loc[common].mean()), f1_perlakuan=float(tr.loc[common].mean()),
                             delta=float(d.mean()), delta_sd=float(d.std(ddof=1)) if len(common) > 1 else 0.0,
                             fold_naik=int((d > 0).sum()), fold_turun=int((d < 0).sum()),
                             delta_per_fold=[float(x) for x in d.values]))
    return pd.DataFrame(rows)


def interpret_ablation(row) -> str:
    n, up, dn = int(row["n_fold"]), int(row["fold_naik"]), int(row["fold_turun"])
    d = float(row["delta"])
    if up == n:
        return "Naik di semua fold."
    if dn == n:
        return "Turun di semua fold."
    if up == 0 and dn == 0:
        return "Tidak berubah."
    arah = "naik" if d > 0 else "turun" if d < 0 else "relatif sama"
    return f"Tidak konsisten: {up} fold naik, {dn} fold turun; rata-rata {arah}."


# ------------------------------------------------------------------ analisis kesalahan
def to_sent(s) -> np.ndarray:
    return pd.Series(s).map(SENT).values


def confusion(yt, yp, labels) -> np.ndarray:
    ct = pd.crosstab(pd.Categorical(yt, categories=labels), pd.Categorical(yp, categories=labels), dropna=False)
    return ct.values.astype(int)


def row_normalize(cm: np.ndarray) -> np.ndarray:
    s = cm.sum(1, keepdims=True).astype(float)
    return np.divide(cm, s, out=np.zeros(cm.shape, dtype=float), where=s > 0)


def per_class_metrics(yt, yp, labels) -> pd.DataFrame:
    from sklearn.metrics import precision_recall_fscore_support
    pr, rc, f1, sup = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    return pd.DataFrame({"kelas": labels, "precision": pr, "recall": rc, "f1": f1, "support": sup})


def top_confusions(yt, yp, labels, k: int = 5) -> pd.DataFrame:
    cm = confusion(yt, yp, labels)
    tot = cm.sum(1)
    rows = [dict(aktual=labels[i], prediksi=labels[j], jumlah=int(cm[i, j]),
                 persen_baris=float(cm[i, j] / tot[i]) if tot[i] else 0.0)
            for i in range(len(labels)) for j in range(len(labels)) if i != j and cm[i, j] > 0]
    out = pd.DataFrame(rows, columns=["aktual", "prediksi", "jumlah", "persen_baris"])
    return out.sort_values(["jumlah", "persen_baris"], ascending=False).head(k).reset_index(drop=True)


def merge_oof_meta(O: pd.DataFrame, M: pd.DataFrame, col: str) -> pd.DataFrame:
    """Gabungkan prediksi OOF dengan metadata lewat `filename`."""
    m = M.merge(O[["filename", "y_true", col]], on="filename", how="inner", validate="one_to_one")
    m["benar"] = (m["y_true"] == m[col]).astype(int)
    return m


def subgroup_accuracy(df: pd.DataFrame, key: str) -> pd.DataFrame:
    return df.groupby(key, as_index=False).agg(akurasi=("benar", "mean"), benar=("benar", "sum"), n=("benar", "size"))


def nice_key(k: str) -> str:
    p = k.split("|")
    if len(p) < 3:
        return k
    v = f" ({VARIAN_NAMES.get(p[3], p[3])})" if len(p) > 3 and p[3] != "base" else ""
    return f"{METHOD_NAMES.get(p[0], p[0])} · {p[2]}{v}"


# ------------------------------------------------------------------ prediksi
def sentiment_probs(labels, p) -> dict:
    out = {k: 0.0 for k in SENT_ORDER}
    for l, v in zip(labels, p):
        out[SENT[l]] += float(v)
    return out


def confidence_text(labels, p):
    """(tingkat, interpretasi). Confidence = probabilitas keluaran model, bukan peluang kebenaran terkalibrasi."""
    order = np.argsort(p)[::-1]
    top, second = float(p[order[0]]), float(p[order[1]])
    a, b = emo_name(labels[order[0]]), emo_name(labels[order[1]])
    if top >= 0.6 and (top - second) >= 0.25:
        return "Tinggi", (f"Model memberi probabilitas terbesar pada **{a}** dengan selisih cukup jauh dari kandidat kedua "
                          f"({b}, {second:.0%}).")
    if top >= 0.4:
        return "Sedang", (f"**{a}** unggul, tetapi **{b}** ({second:.0%}) masih cukup dekat. "
                          "Model belum sepenuhnya memisahkan keduanya.")
    return "Rendah", (f"Probabilitas tersebar pada beberapa emosi (tertinggi **{a}** {top:.0%}, lalu **{b}** {second:.0%}). "
                      "Model menunjukkan ketidakpastian.")


# ------------------------------------------------------------------ sinyal audio
def signal_views(y: np.ndarray, sr: int = SR, max_sec: float = 15.0) -> dict:
    """Waveform, log-mel, kontur F0 (pYIN) dan RMS. Untuk eksplorasi; BUKAN fitur model final."""
    import librosa
    y = np.asarray(y, dtype=np.float32)[: int(sr * max_sec)]
    hop, n_mels = 512, 96
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=hop, n_mels=n_mels, fmax=8000), ref=np.max)
    f0, vflag, _ = librosa.pyin(y, fmin=65, fmax=500, sr=sr, frame_length=2048, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    n = min(len(f0), len(rms))
    t = librosa.times_like(rms, sr=sr, hop_length=hop)[:n]
    f0n, vf = f0[:n], np.asarray(vflag[:n], dtype=bool)
    fv = f0n[vf & ~np.isnan(f0n)]
    return dict(
        t_wave=np.arange(len(y)) / sr, y=y, mel=mel,
        mel_freqs=librosa.mel_frequencies(n_mels=n_mels, fmax=8000),
        t_mel=librosa.times_like(mel[0], sr=sr, hop_length=hop),
        t=t, f0=f0n, rms=rms[:n], dur=len(y) / sr,
        f0_mean=float(np.mean(fv)) if len(fv) else float("nan"),
        f0_range=float(np.max(fv) - np.min(fv)) if len(fv) else float("nan"),
        voiced=float(np.mean(vf)) if n else 0.0,
        rms_mean=float(np.mean(rms)) if len(rms) else float("nan"),
    )
