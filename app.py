# file: app.py
from __future__ import annotations

import os, re, json, hashlib, random, datetime
from io import StringIO, BytesIO
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, NamedTuple

import numpy as np
import pandas as pd
import streamlit as st

# --- Optional libs (app works without them) ---
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

# ------------------ Constants ------------------
MOTIVATIONS = [
    "Sattva","Rajas","Tamas",
    "Prajna","Personal_Unconscious","Collective_Unconscious",
    "Cheng","Wu_Wei","Anatta",
    "Relational_Balance","Thymos","Eros",
]
STRATEGIES = ["Conform","Control","Flow","Risk"]          # subtype only (NOT used in centroid grading)
ORIENTATIONS = ["Cognitive","Energy","Relational","Surrender"]  # derived display only
SELF_SCALES = ["Self_Insight","Self_Serving_Bias"]
ALL_DIMS = MOTIVATIONS + STRATEGIES + ORIENTATIONS
ALL_REQ_FOR_Z = ALL_DIMS + SELF_SCALES
EPS = 1e-8

DEFAULT_DOMAIN_MAP = {
    "Energy": ["Sattva","Rajas","Tamas"],
    "Cognitive": ["Prajna","Personal_Unconscious","Collective_Unconscious"],
    "Integrative": ["Cheng","Wu_Wei","Anatta"],
    "Relational": ["Relational_Balance","Thymos","Eros"],
}

# --- Header canonicalization (accept flexible headers in questions/norms/centroids) ---
HEADER_TO_CANON = {
    "sattva":"Sattva","rajas":"Rajas","tamas":"Tamas",
    "prajna":"Prajna",
    "personal unconscious":"Personal_Unconscious","personal_unconscious":"Personal_Unconscious","pers.u":"Personal_Unconscious","pers u":"Personal_Unconscious",
    "collective unconscious":"Collective_Unconscious","collective_unconscious":"Collective_Unconscious","coll.u":"Collective_Unconscious","coll u":"Collective_Unconscious",
    "cheng":"Cheng","wu wei":"Wu_Wei","wu_wei":"Wu_Wei","anatta":"Anatta",
    "relational balance":"Relational_Balance","relational_balance":"Relational_Balance","rel.bal":"Relational_Balance","rel bal":"Relational_Balance",
    "thymos":"Thymos","eros":"Eros",
    "conform":"Conform","control":"Control","flow":"Flow","risk":"Risk",
    "cognitive":"Cognitive","inward":"Cognitive",
    "energy":"Energy","outward":"Energy",
    "relational":"Relational","relationship":"Relational",
    "surrender":"Surrender",
    "self insight":"Self_Insight","self_insight":"Self_Insight","self-insight":"Self_Insight",
    "self serving bias":"Self_Serving_Bias","self-serving bias":"Self_Serving_Bias","self_serving_bias":"Self_Serving_Bias",
}
def canon(name: str) -> Optional[str]:
    if not name: return None
    s = name.strip().lstrip("\ufeff")
    s = s.replace("—","-").replace("–","-").replace("：",":")
    s = re.sub(r"\(.*?\)", " ", s).lower()
    s = re.sub(r"[^a-z0-9_ :\-]+", " ", s).replace(":", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if s in HEADER_TO_CANON: return HEADER_TO_CANON[s]
    best = None
    for k, v in HEADER_TO_CANON.items():
        if k in s and (best is None or len(k) > len(best[0])): best = (k, v)
    return best[1] if best else None

# ------------------ Z params ------------------
@dataclass
class ZParams:
    mean: Dict[str,float]
    std: Dict[str,float]
    @classmethod
    def fit(cls, df: pd.DataFrame, cols: List[str]) -> "ZParams":
        mu = df[cols].mean().to_dict()
        sd = (df[cols].std(ddof=0) + EPS).to_dict()
        return cls(mu, sd)

def zparams_from_norms_or_single(person_scales: Dict[str,float], norms_df: Optional[pd.DataFrame]) -> "ZParams":
    df = norms_df if norms_df is not None else pd.DataFrame([person_scales])
    for col in ALL_REQ_FOR_Z:
        if col not in df.columns:
            df[col] = np.nan
    return ZParams.fit(df, list(ALL_REQ_FOR_Z))

# ------------------ Embedded centroids (latest you provided) ------------------
# NOTE: Strategies/Orientations present but NOT used in grading. We slice to MOTIVATIONS.
EMBEDDED_CENTROIDS_CSV = """Archetype,Sattva,Rajas,Tamas,Prajna,Personal_Unconscious,Collective_Unconscious,Cheng,Wu_Wei,Anatta,Relational_Balance,Thymos,Eros,Conform,Control,Flow,Risk,Cognitive,Energy,Relational,Surrender
Lucerna (Lantern),4.8,6.8,3.8,6.0,4.7,4.5,5.0,4.9,4.3,4.7,6.2,5.0,4.5,4.8,6.0,4.2,4.6,6.8,5.2,4.6
Arbor (Tree),6.4,4.5,4.7,4.4,4.5,4.8,6.0,4.8,4.3,6.7,5.0,5.3,6.0,4.8,4.2,3.8,4.9,4.9,6.5,5.1
Sharin (Wheel),5.0,6.2,4.2,4.8,4.5,4.7,5.4,5.8,4.5,5.3,6.5,5.0,5.5,4.8,6.0,4.5,4.7,5.2,6.7,5.2
Keras (Rhino),4.8,6.2,5.7,4.8,4.4,4.5,6.7,4.7,4.0,4.5,5.2,4.4,4.3,6.0,4.5,5.8,4.6,5.0,4.6,6.7
Hayabusa (Falcon),4.6,5.3,4.8,4.7,6.7,4.5,6.3,4.4,4.2,4.7,5.7,4.4,5.8,6.0,4.3,4.2,6.7,5.2,4.9,4.8
Arachna (Spider),4.2,4.8,6.5,4.5,4.7,6.3,4.3,4.8,6.4,4.2,4.8,4.5,3.8,4.3,4.8,6.0,4.8,6.6,4.3,4.9
Tempus (Hourglass),6.2,5.0,4.3,6.6,5.3,5.0,6.1,4.8,4.5,4.9,4.8,4.7,4.5,6.0,5.8,4.3,6.6,5.1,4.9,5.1
Simia (Monkey),4.8,5.8,4.2,4.6,4.7,4.8,4.6,6.5,4.3,5.0,5.0,6.2,4.0,4.3,6.0,4.8,4.7,5.0,5.6,6.6
Polvo (Octopus),4.6,4.7,5.5,6.3,5.4,5.1,4.8,4.5,6.4,4.4,4.6,4.5,4.8,4.8,4.2,4.0,6.5,4.6,4.4,6.5
Tigre (Tiger),6.5,6.2,4.4,4.9,4.7,5.8,4.9,5.6,4.3,4.8,4.8,4.7,3.9,4.7,4.8,6.0,4.7,6.7,4.8,4.9
Enguia (Eel),6.3,4.7,4.3,4.5,4.8,4.9,4.8,5.0,4.4,6.5,5.2,6.7,4.6,4.4,5.8,4.3,4.9,5.3,6.6,5.3
Dacia (Nomad),4.8,6.5,4.5,5.6,4.8,6.7,4.7,5.1,5.2,4.6,6.1,5.5,4.2,4.8,5.8,5.7,6.5,5.0,5.1,5.0
"""

def load_centroids_embedded() -> pd.DataFrame:
    df = pd.read_csv(StringIO(EMBEDDED_CENTROIDS_CSV))
    df = df.set_index("Archetype")
    return df

def load_centroids_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Archetype" not in df.columns:
        raise ValueError("Centroid CSV must include an 'Archetype' column.")
    df = df.set_index("Archetype")
    # normalize common header variants
    rename_map = {
        "Pers.U":"Personal_Unconscious", "Personal Unconscious":"Personal_Unconscious",
        "Coll.U":"Collective_Unconscious", "Collective Unconscious":"Collective_Unconscious",
        "Wu Wei":"Wu_Wei",
        "Rel.Bal":"Relational_Balance", "Relational Balance":"Relational_Balance",
    }
    df = df.rename(columns=rename_map)
    return df

def centroids_motivations_only(df: pd.DataFrame) -> pd.DataFrame:
    missing = [m for m in MOTIVATIONS if m not in df.columns]
    if missing:
        raise KeyError(f"Centroid sheet missing motivation columns: {missing}")
    return df[MOTIVATIONS].astype(float).copy()

# ------------------ Questions parsing ------------------
ITEM_KV_RE = re.compile(r"\[(\w+)\s*=\s*(.*?)\]")
def _kv_blocks(s: str) -> Dict[str,str]:
    return {k.upper(): v.strip() for k, v in ITEM_KV_RE.findall(s)}

def parse_txt_questions(raw: str) -> Dict:
    raw = raw.lstrip("\ufeff")
    lines = [l.rstrip() for l in raw.splitlines()]
    spec = {"scale": {"min":1, "max":7, "step":1}, "questions":[]}
    cur_dim: Optional[str] = None
    counts: Dict[str,int] = {}
    for line in lines:
        s = line.strip()
        if not s: continue
        if s.endswith(":") or s.endswith("："):
            cur_dim = canon(s[:-1]); continue
        text = s
        kvs = _kv_blocks(s)
        if kvs: text = ITEM_KV_RE.sub("", s).strip()
        dim = canon(kvs.get("DIM", cur_dim or ""))
        if dim is None: continue
        vmin = int(kvs.get("MIN", "1")); vmax = int(kvs.get("MAX", "7")); vstep = int(kvs.get("STEP", "1"))
        npoints = (vmax - vmin)//vstep + 1
        labels: Optional[List[str]] = None
        if "LABELS" in kvs:
            labels = [p.strip()] if '|' not in kvs["LABELS"] else [p.strip() for p in kvs["LABELS"].split("|")]
            if len(labels) != npoints:
                raise ValueError("LABELS count mismatch for item: " + text)
        else:
            L = kvs.get("L"); R = kvs.get("R")
            if L and R and npoints == 7:
                labels = [L, "Slightly "+L, "Somewhat "+L, "Neutral", "Somewhat "+R, "Slightly "+R, R]
            elif L and R and npoints == 5:
                labels = [L, "Somewhat "+L, "Neutral", "Somewhat "+R, R]
        counts[dim] = counts.get(dim, 0) + 1
        qid = f"q_{dim}_{counts[dim]}"
        spec["questions"].append({
            "id": qid, "dimension": dim, "text": text,
            "min": vmin, "max": vmax, "step": vstep,
            "labels": labels, "L": kvs.get("L"), "R": kvs.get("R"),
        })
    return spec

def aggregate_to_scales(responses: Dict[str,int], spec: Dict) -> Dict[str,float]:
    buckets: Dict[str, List[float]] = {d: [] for d in (ALL_DIMS + SELF_SCALES)}
    for q in spec["questions"]:
        if q["id"] in responses:
            buckets[q["dimension"]].append(float(responses[q["id"]]))
    means: Dict[str, float] = {}
    for d in ALL_DIMS + SELF_SCALES:
        vals = buckets.get(d, [])
        means[d] = float(np.mean(vals)) if len(vals) > 0 else np.nan
    return means

def direct_mean_for_dim(responses: dict, spec: dict, dimension: str) -> float:
    vals = [responses[q["id"]] for q in spec["questions"] if q["dimension"] == dimension and q["id"] in responses]
    return float(np.mean(vals)) if vals else float("nan")

def direct_values_for_dim(responses: dict, spec: dict, dimension: str) -> List[float]:
    return [float(responses[q["id"]]) for q in spec["questions"] if q["dimension"] == dimension and q["id"] in responses]

# ------------------ Utilities ------------------
def euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a-b)**2)))

def normalize_probs(v: np.ndarray) -> np.ndarray:
    s = float(v.sum())
    return v/s if s>0 else np.full_like(v, 1.0/len(v))

def derive_orientations(ps: Dict[str,float]) -> None:
    if np.isnan(ps.get("Energy", np.nan)):
        ps["Energy"] = float(np.nanmean([ps.get(m, np.nan) for m in DEFAULT_DOMAIN_MAP["Energy"]]))
    if np.isnan(ps.get("Cognitive", np.nan)):
        ps["Cognitive"] = float(np.nanmean([ps.get(m, np.nan) for m in DEFAULT_DOMAIN_MAP["Cognitive"]]))
    if np.isnan(ps.get("Relational", np.nan)):
        ps["Relational"] = float(np.nanmean([ps.get(m, np.nan) for m in DEFAULT_DOMAIN_MAP["Relational"]]))
    if np.isnan(ps.get("Surrender", np.nan)):
        ps["Surrender"] = float(np.nanmean([ps.get(m, np.nan) for m in DEFAULT_DOMAIN_MAP["Integrative"]]))

# ------------------ Strategy subtype (top-2) ------------------
BALANCE_DELTA = 0.08
def quadrant_label_from_pair(a: str, b: str) -> str:
    pair = {a, b}
    if "Control" in pair and "Conform" in pair: return "Controlled–Conformist"
    if "Control" in pair and "Risk"   in pair: return "Controlled–Risk"
    if "Flow"    in pair and "Conform" in pair: return "Flow–Conformist"
    if "Flow"    in pair and "Risk"    in pair: return "Flow–Risk"
    return "Ambiguous"

def strategy_subtype_from_means(str_means: Dict[str, float]) -> dict:
    ordered = sorted(str_means.items(), key=lambda kv: (-kv[1], kv[0]))
    (s1, v1), (s2, v2) = ordered[0], ordered[1]
    denom = max(v1 + v2, EPS)
    p1, p2 = v1/denom, v2/denom
    leaning = "Balanced" if abs(p1 - p2) <= BALANCE_DELTA else f"Leaning {s1}"
    quadrant = quadrant_label_from_pair(s1, s2)
    return {"top_pair": (s1, s2), "percentages": {s1: p1, s2: p2}, "leaning": leaning, "quadrant": quadrant}

# ------------------ Scoring (motivations ONLY; Euclidean) ------------------
class ScorePieces(NamedTuple):
    probs: Dict[str,float]
    top3: List[Tuple[str,float]]

@dataclass
class DistRow:
    archetype: str
    distance: float
    similarity: float

def score_single_mot_only(person: pd.Series, z: "ZParams", arch_mot: pd.DataFrame) -> Tuple[ScorePieces, pd.DataFrame]:
    # z-standardize person & centroids in motivation space
    z_person_m = np.array([(person[m] - z.mean[m]) / z.std[m] for m in MOTIVATIONS], dtype=float)
    arch_std = arch_mot.copy()
    for m in MOTIVATIONS:
        arch_std[m] = (arch_std[m] - z.mean.get(m, 0.0)) / (z.std.get(m, 1.0))
    names = list(arch_std.index)
    sims, rows = [], []
    for name in names:
        a = arch_std.loc[name, MOTIVATIONS].to_numpy(dtype=float)
        D = euclid(z_person_m, a)
        S = 1.0 / (1.0 + D)
        sims.append(S)
        rows.append(DistRow(name, D, S))
    probs_arr = normalize_probs(np.array(sims, dtype=float))
    order = np.argsort(-probs_arr)
    top3 = [(names[i], float(probs_arr[i])) for i in order[:3]]
    probs = {names[i]: float(probs_arr[i]) for i in range(len(names))}
    dist_df = pd.DataFrame([{"Archetype":r.archetype, "Distance":r.distance, "Similarity":r.similarity} for r in rows])\
                .sort_values(["Distance","Archetype"])
    return ScorePieces(probs=probs, top3=top3), dist_df

# ------------------ Plotly figures (guarded) ------------------
if HAS_PLOTLY:
    def fig_blend_triangle(top3: List[Tuple[str,float]]) -> "go.Figure":
        A = np.array([0.0, 0.0]); B = np.array([1.0, 0.0]); C = np.array([0.5, np.sqrt(3)/2])
        names = [top3[0][0], top3[1][0], top3[2][0]]
        vals = np.array([top3[0][1], top3[1][1], top3[2][1]], float)
        s = float(vals.sum()) or 1.0
        w = vals / s
        P = w[0]*A + w[1]*B + w[2]*C
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[A[0],B[0],C[0],A[0]], y=[A[1],B[1],C[1],A[1]], mode="lines", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=[A[0],B[0],C[0]], y=[A[1],B[1],C[1]], mode="markers+text",
                                 marker=dict(size=10), text=[names[0],names[1],names[2]], textposition="top center"))
        fig.add_trace(go.Scatter(x=[P[0]], y=[P[1]], mode="markers", marker=dict(size=14, symbol="star")))
        fig.update_layout(title="Personality Blend Triangle", xaxis=dict(visible=False), yaxis=dict(visible=False),
                          showlegend=False, height=350, margin=dict(l=10,r=10,t=40,b=10))
        return fig

    def fig_strategy_compass(str_means: Dict[str, float], sub: dict) -> "go.Figure":
        r = [str_means["Control"], str_means["Conform"], str_means["Flow"], str_means["Risk"], str_means["Control"]]
        theta = ["Control","Conform","Flow","Risk","Control"]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=r, theta=theta, fill='toself', name="Strategies"))
        s1, s2 = sub["top_pair"]
        ltxt = f"{sub['quadrant']} — {s1} {sub['percentages'][s1]*100:.0f}% + {s2} {sub['percentages'][s2]*100:.0f}% ({sub['leaning']})"
        fig.update_layout(title="Control–Flow Compass",
                          polar=dict(radialaxis=dict(range=[1,7], showticklabels=True)),
                          height=350, margin=dict(l=10,r=10,t=40,b=10), showlegend=False,
                          annotations=[dict(text=ltxt, x=0.5, y=1.15, xref="paper", yref="paper", showarrow=False)])
        return fig

    def fig_confidence_curves(si_mean: float, ssb_mean: float, C: float) -> "go.Figure":
        x = np.linspace(1,7,241); sigma = 1.2
        def pdf(mu): return np.exp(-0.5*((x-mu)/sigma)**2)
        y_si = pdf(si_mean); y_ssb = pdf(ssb_mean)
        y_si /= y_si.max() or 1.0; y_ssb /= y_ssb.max() or 1.0
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y_si, mode="lines", name="Self Insight", fill='tozeroy', opacity=0.5))
        fig.add_trace(go.Scatter(x=x, y=y_ssb, mode="lines", name="Self-Serving Bias", fill='tozeroy', opacity=0.5))
        fig.update_layout(title=f"Confidence Calibration — Index: {C:.3f}",
                          xaxis_title="Scale (1–7)", yaxis_title="Relative Density",
                          height=350, margin=dict(l=10,r=10,t=40,b=10))
        fig.add_annotation(text="Metacognitive Accuracy Zone = overlap", x=4, y=0.9, showarrow=False)
        return fig

    def fig_motivation_spectrum(series: pd.Series, label: str) -> "go.Figure":
        s = series.sort_values(ascending=True)
        lo = 1 if s.min() >= 1 else min(1, float(s.min())-0.2)
        hi = 7 if s.max() <= 7 else max(7, float(s.max())+0.2)
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation='h'))
        fig.update_layout(title=f"12-Bar Motivation Spectrum ({label})",
                          xaxis=dict(range=[lo, hi]),
                          height=420, margin=dict(l=80,r=10,t=40,b=10), showlegend=False)
        return fig

    def fig_motivation_wheel(series: pd.Series, label: str) -> "go.Figure":
        names = list(series.index)
        vals = [float(series.get(n, np.nan)) for n in names]
        theta = np.linspace(0, 360, num=len(names), endpoint=False)
        fig = go.Figure()
        fig.add_trace(go.Barpolar(r=vals, theta=theta, text=names,
                                  hovertext=[f"{n}: {v:.2f}" for n,v in zip(names, vals)], hoverinfo="text"))
        fig.update_layout(title=f"Motivational Wheel ({label})",
                          polar=dict(radialaxis=dict(range=[1,7])),
                          height=420, margin=dict(l=10,r=10,t=40,b=10), showlegend=False)
        return fig

    def fig_centroid_heatmap(centroids_df: pd.DataFrame, normalize: bool = False) -> "go.Figure":
        heat = centroids_df[MOTIVATIONS].astype(float)
        title = "Archetype × Motivation Centroids (1–7)"; zmin, zmax = 1, 7
        if normalize:
            heat = (heat - heat.mean(axis=0)) / (heat.std(axis=0) + EPS)
            title = "Archetype × Motivation (Column Z-normalized)"
            zmin, zmax = float(heat.min()), float(heat.max())
        fig = px.imshow(heat, x=heat.columns, y=heat.index, color_continuous_scale="Viridis",
                        aspect="auto", origin="upper", zmin=zmin, zmax=zmax,
                        labels=dict(color="Score"), title=title)
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10), xaxis=dict(side="top"))
        return fig

# ------------------ Master CSV ------------------
RESULTS_CSV_PATH = os.getenv("RESULTS_CSV_PATH", "data/results_master.csv")
def init_csv() -> None:
    folder = os.path.dirname(RESULTS_CSV_PATH) or "."
    os.makedirs(folder, exist_ok=True)

def save_result_to_csv(
    participant_id: str,
    top3: list[tuple[str, float]],
    C: float,
    si_mean: float,
    ssb_mean: float,
    subtype: dict,
    mot_means: dict[str, float],
    archetype_probs: dict[str, float],
    archetype_order: list[str],
) -> None:
    (p1, p1v), (p2, p2v), (p3, p3v) = top3
    s1, s2 = subtype["top_pair"]
    pct1 = float(subtype["percentages"][s1])
    pct2 = float(subtype["percentages"][s2])
    row = {
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "participant_id": participant_id,
        "primary_archetype": p1, "primary_prob": float(p1v),
        "secondary_archetype": p2, "secondary_prob": float(p2v),
        "tertiary_archetype": p3, "tertiary_prob": float(p3v),
        "confidence": float(C), "si_mean": float(si_mean), "ssb_mean": float(ssb_mean),
        "strategy_pair": f"{s1}+{s2}",
        "strategy_pct1": pct1, "strategy_pct2": pct2,
        "strategy_leaning": subtype["leaning"], "strategy_quadrant": subtype["quadrant"],
        # 12 motivations (means)
        "Sattva": float(mot_means.get("Sattva", np.nan)),
        "Rajas": float(mot_means.get("Rajas", np.nan)),
        "Tamas": float(mot_means.get("Tamas", np.nan)),
        "Prajna": float(mot_means.get("Prajna", np.nan)),
        "Personal_Unconscious": float(mot_means.get("Personal_Unconscious", np.nan)),
        "Collective_Unconscious": float(mot_means.get("Collective_Unconscious", np.nan)),
        "Cheng": float(mot_means.get("Cheng", np.nan)),
        "Wu_Wei": float(mot_means.get("Wu_Wei", np.nan)),
        "Anatta": float(mot_means.get("Anatta", np.nan)),
        "Relational_Balance": float(mot_means.get("Relational_Balance", np.nan)),
        "Thymos": float(mot_means.get("Thymos", np.nan)),
        "Eros": float(mot_means.get("Eros", np.nan)),
    }
    # add all archetype probabilities
    for name in archetype_order:
        row[f"Prob_{name}"] = float(archetype_probs.get(name, np.nan))
    exists = os.path.exists(RESULTS_CSV_PATH)
    df = pd.DataFrame([row])
    df.to_csv(RESULTS_CSV_PATH, mode="a", header=not exists, index=False, encoding="utf-8")

# ------------------ UI ------------------
st.set_page_config(page_title="Motivational Archetypes – Test", page_icon="🧭", layout="wide")
st.title("🧭 Motivational Archetypes – Test")

with st.sidebar:
    st.markdown("**Data sources**")
    q_up = st.file_uploader("Override questions.txt (optional)", type=["txt"])
    norms_up = st.file_uploader("Optional norms.csv (for z-standardization)", type=["csv"])
    centroids_up = st.file_uploader("Optional centroids CSV (override embedded)", type=["csv"])
    st.markdown("---")
    participant_id = st.text_input("Participant ID", value="P001")
    ranking_mode = st.selectbox("Motivation ranking metric", ["Raw means (1–7)", "Z-scores (vs norms)"])
    st.caption(f"Master CSV: `{RESULTS_CSV_PATH}`")

def load_questions_from_repo() -> Dict:
    q_path = os.getenv("QUESTIONS_PATH", "questions.txt")
    if not os.path.exists(q_path):
        raise FileNotFoundError(f"questions file not found at: {os.path.abspath(q_path)}")
    text = open(q_path, "r", encoding="utf-8").read().lstrip("\ufeff")
    return parse_txt_questions(text)

# questions
spec = parse_txt_questions(q_up.read().decode("utf-8")) if q_up is not None else load_questions_from_repo()
items: List[Dict] = list(spec.get("questions", []))
if not items:
    st.error("No items parsed. Check headers/items in questions.txt.")
    st.stop()

# centroids
if centroids_up is not None:
    centroids_df_full = load_centroids_from_csv(centroids_up)
else:
    centroids_df_full = load_centroids_embedded()
ARCHETYPE_CENTROIDS_MOT = centroids_motivations_only(centroids_df_full)

# stable shuffle per participant + spec
def stable_shuffle(items: List[Dict], pid: str, spec_obj: Dict) -> List[Dict]:
    spec_bytes = json.dumps({k:v for k,v in spec_obj.items()}, sort_keys=True).encode("utf-8")
    seed_hex = hashlib.sha256((pid + "|").encode("utf-8") + spec_bytes).hexdigest()[:16]
    rng = random.Random(int(seed_hex, 16))
    out = items.copy(); rng.shuffle(out); return out

spec_fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()
if "shuffle_meta" not in st.session_state or st.session_state.shuffle_meta != (participant_id, spec_fingerprint):
    st.session_state.shuffle_meta = (participant_id, spec_fingerprint)
    st.session_state.shuffled_items = stable_shuffle(items, participant_id, spec)
shuffled = st.session_state.shuffled_items

def value_to_label(item: Dict, val: int) -> str:
    vmin = int(item.get("min", 1)); vmax = int(item.get("max", 7)); step = int(item.get("step", 1))
    labels = item.get("labels"); idx = (val - vmin) // step
    if labels and 0 <= idx < len(labels): return labels[idx]
    L, R = item.get("L"), item.get("R"); npoints = (vmax - vmin)//step + 1
    if L and R and npoints == 7:
        return [L, f"Slightly {L}", f"Somewhat {L}", "Neutral", f"Somewhat {R}", f"Slightly {R}", R][idx]
    if L and R and npoints == 5:
        return [L, f"Somewhat {L}", "Neutral", f"Somewhat {R}", R][idx]
    return ""

# questionnaire UI
responses: Dict[str,int] = {}
scale_defaults = spec.get("scale", {"min":1,"max":7,"step":1})
st.subheader("📝 Questionnaire")
for it in shuffled:
    vmin = int(it.get("min", scale_defaults.get("min", 1)))
    vmax = int(it.get("max", scale_defaults.get("max", 7)))
    step = int(it.get("step", scale_defaults.get("step", 1)))
    default_val = vmin + ((vmax - vmin) // (2 * step)) * step
    c1, c2 = st.columns([2, 3])
    with c1: st.markdown(f"**{it['text']}**")
    with c2:
        cur = st.session_state.get(it["id"], default_val)
        curr_label = value_to_label(it, cur)
        if curr_label:
            st.markdown(f"<div style='font-size:0.9rem;opacity:.8;margin-bottom:-0.5rem'><b>{curr_label}</b></div>", unsafe_allow_html=True)
        elif it.get("L") or it.get("R"):
            st.markdown(f"<div style='font-size:0.9rem;opacity:.7;margin-bottom:-0.5rem'><b>{it.get('L','')}</b></div>", unsafe_allow_html=True)
        val = st.slider(label="", min_value=vmin, max_value=vmax, step=step,
                        value=cur, key=it["id"],
                        help=None if not (it.get("L") or it.get("R")) else f"{it.get('L','')} ↔ {it.get('R','')}")
    st.divider()
    responses[it["id"]] = val

if not st.button("Compute Results"):
    st.stop()

# aggregate
person_scales = aggregate_to_scales(responses, spec)
derive_orientations(person_scales)

missing_dims = [d for d in (MOTIVATIONS+STRATEGIES+ORIENTATIONS) if np.isnan(person_scales.get(d, np.nan))]
if missing_dims:
    st.error(f"Missing responses for: {missing_dims}")
    st.stop()

# confidence (Self Insight normal; Self-Serving Bias reversed)
def compute_confidence_from_means(si: float, ssb: float) -> tuple[float, str]:
    si_n  = (float(si)  - 1.0) / 6.0       # 1→0, 7→1
    ssb_n = (7.0 - float(ssb)) / 6.0       # 1→1, 7→0 (reversed)
    C = max(0.0, min(1.0, 0.5 * (si_n + ssb_n)))
    level = "High" if C >= 2/3 else ("Moderate" if C >= 0.45 else "Low")
    return C, level

si_vals  = direct_values_for_dim(responses, spec, "Self_Insight")
ssb_vals = direct_values_for_dim(responses, spec, "Self_Serving_Bias")
si_mean  = float(np.mean(si_vals))  if si_vals  else np.nan
ssb_mean = float(np.mean(ssb_vals)) if ssb_vals else np.nan
if np.isnan(si_mean) or np.isnan(ssb_mean):
    st.error("Missing Self Insight or Self Serving Bias items.")
    st.stop()
C, C_level = compute_confidence_from_means(si_mean, ssb_mean)

# strategy subtype (only for display)
str_means = {d: direct_mean_for_dim(responses, spec, d) for d in STRATEGIES}
sub = strategy_subtype_from_means(str_means)

# optional norms for z
norms_df = None
if norms_up is not None:
    norms_df = pd.read_csv(norms_up)
    ren = {"Inward":"Cognitive","Outward":"Energy","Relationship":"Relational"}
    have = [c for c in ren if c in norms_df.columns]
    if have: norms_df = norms_df.rename(columns={c: ren[c] for c in have})

# scoring (motivations only)
z = zparams_from_norms_or_single(person_scales, norms_df)
person_row = pd.Series({**person_scales, "participant_id": participant_id})
score, dist_df = score_single_mot_only(person_row, z, ARCHETYPE_CENTROIDS_MOT)

# results
probs = pd.Series(score.probs).sort_values(ascending=False).rename("probability")
(p1,p1v),(p2,p2v),(p3,p3v) = score.top3

def top3_percentages(top3: List[Tuple[str,float]]) -> List[Tuple[str,int]]:
    vals = [p for _, p in top3]; s = sum(vals) or 1.0
    raw = [p / s * 100.0 for p in vals]
    a = int(round(raw[0])); b = int(round(raw[1])); c = 100 - a - b
    return [(top3[0][0], a), (top3[1][0], b), (top3[2][0], c)]
mix = top3_percentages(score.top3)
mix_text = " · ".join([f"{pct}% {name}" for name, pct in mix])

# motivation ranking display series
if ranking_mode.startswith("Z-scores"):
    mot_series = pd.Series({m: (person_scales[m]-z.mean[m])/z.std[m] for m in MOTIVATIONS}).sort_values(ascending=False).rename("z")
    mot_df = mot_series.to_frame(); score_col_name = "z"; mot_label = "Z-scores"
else:
    mot_series = pd.Series({m: person_scales[m] for m in MOTIVATIONS}).sort_values(ascending=False).rename("mean")
    mot_df = mot_series.to_frame(); score_col_name = "mean"; mot_label = "Raw means (1–7)"
mot_df["rank"] = np.arange(1, len(mot_df)+1)

st.session_state["user_test_results"] = user_test_results

# ============================================================
# 📘 Full Personality Report Generation
# ============================================================
import streamlit as st
from modules.web_integration import generate_user_report

if "user_test_results" in st.session_state:
    user_data = st.session_state["user_test_results"]

    st.markdown("---")
    st.subheader("📘 Generate Your Full Personality Report")

    if st.button("Generate My Full Analytical Report"):
        with st.spinner("Building your detailed report..."):
            pdf_path, txt_path = generate_user_report(user_data, mode="full")

        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇️ Download Full Report (PDF)",
                f,
                file_name="Personality_Report.pdf",
                mime="application/pdf",
            )

        with open(txt_path, "r") as f:
            st.download_button(
                "⬇️ Download Text Version (.txt)",
                f,
                file_name="Personality_Report.txt",
                mime="text/plain",
            )


# ------------ Save to master CSV ------------
init_csv()
save_result_to_csv(
    participant_id=participant_id,
    top3=score.top3,
    C=C, si_mean=si_mean, ssb_mean=ssb_mean,
    subtype=sub,
    mot_means={m: person_scales[m] for m in MOTIVATIONS},
    archetype_probs=score.probs,
    archetype_order=list(ARCHETYPE_CENTROIDS_MOT.index)
)

# ------------------ Layout ------------------
left, right = st.columns([1,1])

with left:
    st.subheader("🏆 Top Archetypes")
    st.metric("Primary", p1, f"{p1v:.3f}")
    st.metric("Secondary", p2, f"{p2v:.3f}")
    st.metric("Tertiary", p3, f"{p3v:.3f}")
    st.markdown(f"**Top-3 mix:** {mix_text}")

    st.subheader("🧭 Strategy Subtype")
    s1, s2 = sub["top_pair"]; pct1 = sub["percentages"][s1]*100; pct2 = sub["percentages"][s2]*100
    st.write(f"**{sub['quadrant']}** — {s1} {pct1:.0f}% + {s2} {pct2:.0f}%")
    st.caption(sub["leaning"])

    st.subheader("🔒 Confidence")
    st.metric("Confidence Index (0–1)", f"{C:.3f}")
    st.caption(f"Self Insight mean: {si_mean:.2f} · Self Serving Bias mean: {ssb_mean:.2f} · Level: {C_level}")

    st.subheader("🧩 Motivation Ranking (Top 12)")
    inline_df = mot_df[[score_col_name]].copy().reset_index().rename(columns={"index":"Motivation", score_col_name: ("Z" if score_col_name=="z" else "Mean")})
    inline_df.insert(0, "Rank", np.arange(1, len(inline_df)+1))
    st.table(inline_df)

    with st.expander("🎨 Show visuals"):
        if HAS_PLOTLY:
            st.plotly_chart(fig_blend_triangle(score.top3), use_container_width=True)
            st.plotly_chart(fig_strategy_compass(str_means, sub), use_container_width=True)
            st.plotly_chart(fig_confidence_curves(si_mean, ssb_mean, C), use_container_width=True)
            tabs = st.tabs(["12-Bar Spectrum", "Motivational Wheel", "Centroid Heatmap"])
            with tabs[0]:
                st.plotly_chart(fig_motivation_spectrum(mot_series.rename("score"), mot_label), use_container_width=True)
            with tabs[1]:
                st.plotly_chart(fig_motivation_wheel(mot_series.rename("score"), mot_label), use_container_width=True)
            with tabs[2]:
                st.plotly_chart(fig_centroid_heatmap(ARCHETYPE_CENTROIDS_MOT, normalize=False), use_container_width=True)
        else:
            st.info("Plotly not installed. Add `plotly` to requirements.txt to enable charts.")

with right:
    st.subheader("📊 Archetype Probabilities")
    st.dataframe(probs.to_frame())

    st.subheader(f"🧩 Motivation Ranking — {'Z' if score_col_name=='z' else 'Raw'}")
    st.dataframe(mot_df[["rank", score_col_name]].rename(columns={score_col_name: ("Z" if score_col_name=='z' else "Mean")}))

    with st.expander("🛠 Diagnostics: Distances & Similarity (motivations only)"):
        st.dataframe(dist_df := dist_df, use_container_width=True)
        d_buf = StringIO(); dist_df.to_csv(d_buf, index=False)
        st.download_button("Download distances.csv", d_buf.getvalue(), "distances.csv", "text/csv")

    # Master CSV download
    if os.path.exists(RESULTS_CSV_PATH):
        with open(RESULTS_CSV_PATH, "rb") as f:
            st.download_button("Download master results CSV", f.read(), file_name="results_master.csv", mime="text/csv")
    else:
        st.caption("Master CSV not created yet.")

# ------------------ PDF (tables/text only) ------------------
if HAS_REPORTLAB:
    def build_pdf_report(participant_id: str, top3: List[Tuple[str,float]], mix: List[Tuple[str,int]],
                         probs_series: pd.Series, mot_df: pd.DataFrame, ranking_mode_label: str,
                         C: float, C_level: str, si_mean: float, ssb_mean: float, strategy_sub: dict) -> bytes:
        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=LETTER, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()
        story: List = []

        story.append(Paragraph(f"Motivational Archetypes Report – {participant_id}", styles["Title"]))
        story.append(Spacer(1, 8))

        (tp1,v1),(tp2,v2),(tp3,v3) = top3
        story.append(Paragraph("<b>Top Archetypes</b>", styles["Heading2"]))
        story.append(Paragraph(f"Primary: <b>{tp1}</b> ({v1:.3f})", styles["Normal"]))
        story.append(Paragraph(f"Secondary: <b>{tp2}</b> ({v2:.3f})", styles["Normal"]))
        story.append(Paragraph(f"Tertiary: <b>{tp3}</b> ({v3:.3f})", styles["Normal"]))
        mix_line = " · ".join([f"{pct}% {name}" for name, pct in mix])
        story.append(Paragraph(f"Top-3 mix: <b>{mix_line}</b>", styles["Normal"]))
        story.append(Spacer(1, 8))

        story.append(Paragraph("<b>Strategy Subtype</b>", styles["Heading2"]))
        s1, s2 = strategy_sub["top_pair"]
        p1 = strategy_sub["percentages"][s1]*100; p2 = strategy_sub["percentages"][s2]*100
        story.append(Paragraph(f"{strategy_sub['quadrant']} — {s1} {p1:.0f}% + {s2} {p2:.0f}%", styles["Normal"]))
        story.append(Paragraph(strategy_sub["leaning"], styles["Normal"]))
        story.append(Spacer(1, 8))

        story.append(Paragraph("<b>Confidence</b>", styles["Heading2"]))
        story.append(Paragraph(f"Confidence Index: <b>{C:.3f}</b> ({C_level})", styles["Normal"]))
        story.append(Paragraph(f"Self Insight mean: {si_mean:.2f} · Self Serving Bias mean: {ssb_mean:.2f}", styles["Normal"]))
        story.append(Spacer(1, 8))

        story.append(Paragraph(f"<b>Motivation Ranking — {ranking_mode_label}</b>", styles["Heading2"]))
        col_label = "Z" if "z" in mot_df.columns else "Mean"
        mot_tbl_data = [["Rank", "Motivation", col_label]]
        mot_iter = mot_df.reset_index().rename(columns={"index":"Motivation"})
        for i, (_, r) in enumerate(mot_iter.iterrows(), start=1):
            mot_tbl_data.append([i, r["Motivation"], f"{float(r.get('z', r.get('mean'))):.3f}"])
        mot_tbl = Table(mot_tbl_data, hAlign="LEFT")
        mot_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('ALIGN', (0,0), (-1,0), 'CENTER'),
            ('ALIGN', (2,1), (2,-1), 'RIGHT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ]))
        story.append(mot_tbl)

        story.append(Spacer(1, 8))

        story.append(Paragraph("<b>Archetype Probabilities</b>", styles["Heading2"]))
        probs_tbl_data = [["Archetype", "Probability"]]
        for name, val in probs_series.items():
            probs_tbl_data.append([name, f"{val:.3f}"])
        probs_tbl = Table(probs_tbl_data, hAlign="LEFT")
        probs_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('ALIGN', (1,1), (1,-1), 'RIGHT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ]))
        story.append(probs_tbl)
        doc.build(story)
        return buf.getvalue()

    pdf_bytes = build_pdf_report(
        participant_id=participant_id,
        top3=score.top3,
        mix=mix,
        probs_series=probs,
        mot_df=mot_df[[score_col_name]].rename(columns={score_col_name: score_col_name}).sort_values(by=score_col_name, ascending=False),
        ranking_mode_label=("Z-scores" if score_col_name=="z" else "Raw means (1–7)"),
        C=C, C_level=C_level, si_mean=si_mean, ssb_mean=ssb_mean,
        strategy_sub=sub,
    )
    st.download_button("📄 Download PDF report", data=pdf_bytes, file_name=f"{participant_id}_report.pdf", mime="application/pdf")
else:
    st.info("📄 PDF export disabled (install `reportlab`).")

