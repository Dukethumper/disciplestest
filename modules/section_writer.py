from datetime import datetime

def build_section(title: str, content: str) -> str:
    return f"\n\n=== {title.upper()} ===\n{content.strip()}\n"

def section_core_overview(results: dict) -> str:
    m = results["Motivation_Means"]
    primary = max(m, key=m.get)
    overview = (
        f"The participant’s profile centers on **{primary}**, "
        f"indicating that this motivation currently defines their psychological focus. "
        "High values in this dimension suggest strong directional intent and self-organization."
    )
    return build_section("Core Overview", overview)

def section_behavioral_indices(results: dict) -> str:
    indices = results["Behavioral_Indices"]
    lines = []
    for k, v in indices.items():
        v = round(v, 2)
        if v >= 5:
            desc = "high, reflecting well-developed expression of this quality."
        elif v >= 3:
            desc = "moderate, indicating a balanced but situational expression."
        else:
            desc = "lower than average, suggesting under-utilization."
        lines.append(f"{k.replace('_',' ')}: {v} — {desc}")
    body = "Behavioral indices summarize secondary traits derived from motivational means.\n" + "\n".join(lines)
    return build_section("Behavioral Indices & Trait Profiles", body)

def section_advanced_extensions(results: dict) -> str:
    ext = results["Advanced_Extensions"]
    cog = ext["Cognitive_Mode"]
    leader = ext["Leadership_Orientation"]
    text = (
        "Cognitive Mode Distribution:\n"
        + ", ".join([f"{k}: {round(v,1)}%" for k, v in cog.items()])
        + "\nLeadership Orientation:\n"
        + ", ".join([f"{k}: {round(v,2)}" for k, v in leader.items()])
        + "\nThese scores describe how the participant organizes thought and social agency."
    )
    return build_section("Advanced Extensions", text)

def section_appendix(results: dict) -> str:
    m = results["Motivation_Means"]
    lines = [f"{k}: {round(v,2)}" for k, v in m.items()]
    appendix = "Raw Motivation Means\n" + "\n".join(lines)
    appendix += "\n\nGenerated on " + datetime.now().strftime("%Y-%m-%d %H:%M")
    return build_section("Appendix", appendix)

def compile_full_text_report(results: dict) -> str:
    sections = [
        section_core_overview(results),
        section_behavioral_indices(results),
        section_advanced_extensions(results),
        section_appendix(results)
    ]
    return "\n".join(sections)
