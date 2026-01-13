import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile

def _save_fig(fig, name):
    path = os.path.join(tempfile.gettempdir(), name)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path

def plot_motivational_hierarchy(means: dict) -> str:
    keys, vals = list(means.keys()), list(means.values())
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(keys, vals)
    ax.set_xlabel("Mean Score (1–7)")
    ax.set_title("Motivational Hierarchy")
    ax.invert_yaxis()
    return _save_fig(fig, "motivational_hierarchy.png")

def plot_behavioral_indices(indices: dict) -> str:
    keys, vals = list(indices.keys()), list(indices.values())
    fig = plt.figure(figsize=(4, 4))
    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    vals += vals[:1]
    angles += angles[:1]
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), keys)
    ax.set_title("Behavioral Indices Radar")
    return _save_fig(fig, "behavioral_indices.png")

def plot_temporal_focus(focus: dict) -> str:
    labels, vals = list(focus.keys()), list(focus.values())
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Temporal Orientation")
    return _save_fig(fig, "temporal_focus.png")

def plot_personality_balance(means: dict) -> str:
    dims = list(means.values())
    fig = plt.figure(figsize=(4, 4))
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    dims += dims[:1]
    angles += angles[:1]
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, dims, linewidth=1.5)
    ax.fill(angles, dims, alpha=0.2)
    ax.set_thetagrids(np.degrees(angles[:-1]), list(means.keys()))
    ax.set_title("Personality Balance Map")
    return _save_fig(fig, "personality_balance.png")

def create_all_visuals(results: dict) -> dict:
    mean_path = plot_motivational_hierarchy(results["Motivation_Means"])
    radar_path = plot_behavioral_indices(results["Behavioral_Indices"])
    focus_path = plot_temporal_focus(results["Advanced_Extensions"]["Temporal_Focus"])
    balance_path = plot_personality_balance(results["Motivation_Means"])
    return {
        "Motivational_Hierarchy": mean_path,
        "Behavioral_Indices": radar_path,
        "Temporal_Focus": focus_path,
        "Personality_Balance": balance_path,
    }
