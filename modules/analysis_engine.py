import numpy as np
import pandas as pd

MOTIVATION_KEYS = [
    "Sattva", "Rajas", "Tamas", "Prajna", "Personal_Unconscious",
    "Collective_Unconscious", "Cheng", "Wu_Wei", "Anatta",
    "Relational_Balance", "Thymos", "Eros"
]

def compute_motivation_means(data: dict) -> dict:
    means = {}
    for key in MOTIVATION_KEYS:
        questions = [v for k, v in data.items() if key.lower() in k.lower()]
        means[key] = np.mean(questions) if questions else np.nan
    return means

def normalize_scores(means: dict) -> dict:
    values = np.array(list(means.values()))
    mean, std = np.nanmean(values), np.nanstd(values)
    zscores = {k: (v - mean) / std if std != 0 else 0 for k, v in means.items()}
    return zscores

def compute_behavioral_indices(means: dict) -> dict:
    m = means
    return {
        "Self_Confidence": np.nanmean([m["Sattva"], m["Rajas"], m["Cheng"], m["Thymos"]]),
        "Introversion_Extroversion": (m["Rajas"] + m["Eros"]) - (m["Tamas"] + m["Wu_Wei"]),
        "Charisma": np.nanmean([m["Eros"], m["Rajas"], m["Thymos"], m["Relational_Balance"]]),
        "Analytical_Depth": np.nanmean([m["Prajna"], m["Personal_Unconscious"], m["Cheng"]]),
        "Emotional_Sensitivity": np.nanmean([m["Eros"], m["Anatta"], m["Wu_Wei"]]),
        "Adaptability": np.nanmean([m["Wu_Wei"], m["Anatta"], m["Tamas"], m["Sattva"]]),
        "Creativity": np.nanmean([m["Collective_Unconscious"], m["Anatta"], m["Wu_Wei"], m["Prajna"]]),
        "Moral_Integrity": np.nanmean([m["Sattva"], m["Cheng"], m["Thymos"]]),
    }

def compute_advanced_extensions(means: dict) -> dict:
    m = means
    cognitive_modes = {
        "Analytical": np.nanmean([m["Prajna"], m["Cheng"]]),
        "Intuitive": np.nanmean([m["Anatta"], m["Collective_Unconscious"]]),
        "Relational": np.nanmean([m["Relational_Balance"], m["Eros"]]),
        "Experiential": np.nanmean([m["Tamas"], m["Wu_Wei"]])
    }
    total = sum(cognitive_modes.values())
    cognitive_distribution = {k: (v / total) * 100 for k, v in cognitive_modes.items()}

    emotional_regulation = {
        "Containment": np.nanmean([m["Cheng"], m["Tamas"]]),
        "Adaptive": np.nanmean([m["Wu_Wei"], m["Sattva"]]),
        "Expressive": np.nanmean([m["Eros"], m["Anatta"]])
    }

    leadership_orientation = {
        "Visionary": m["Rajas"],
        "Servant": np.nanmean([m["Sattva"], m["Relational_Balance"]]),
        "Guardian": np.nanmean([m["Cheng"], m["Thymos"]]),
        "Adaptive": np.nanmean([m["Wu_Wei"], m["Prajna"]])
    }

    temporal_focus = {
        "Past": np.nanmean([m["Tamas"], m["Cheng"]]),
        "Present": np.nanmean([m["Wu_Wei"], m["Eros"]]),
        "Future": np.nanmean([m["Rajas"], m["Prajna"]])
    }

    cog_em_align = abs(np.nanmean(list(cognitive_modes.values())) -
                       np.nanmean([m["Relational_Balance"], m["Eros"], m["Thymos"]]))

    return {
        "Cognitive_Mode": cognitive_distribution,
        "Emotional_Regulation": emotional_regulation,
        "Leadership_Orientation": leadership_orientation,
        "Temporal_Focus": temporal_focus,
        "Cognitive_Emotional_Alignment": cog_em_align
    }

def compile_all_metrics(data: dict) -> dict:
    means = compute_motivation_means(data)
    zscores = normalize_scores(means)
    indices = compute_behavioral_indices(means)
    extensions = compute_advanced_extensions(means)
    return {
        "Motivation_Means": means,
        "ZScores": zscores,
        "Behavioral_Indices": indices,
        "Advanced_Extensions": extensions
    }
