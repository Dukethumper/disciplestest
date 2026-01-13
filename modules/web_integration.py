import os
import tempfile
from modules.data_loader import extract_user_data, preprocess_data
from modules.analysis_engine import compile_all_metrics
from modules.section_writer import compile_full_text_report
from modules.visualizer import create_all_visuals
from modules.report_builder import build_reports

def generate_user_report(user_data: dict, mode: str = "full") -> tuple:
    """
    Orchestrates full report generation and returns (pdf_path, txt_path)
    for immediate download or streaming.
    """

    df = extract_user_data(user_data, mode="json")
    df = preprocess_data(df)
    user_record = df.to_dict(orient="records")[0]

    results = compile_all_metrics(user_record)
    text_report = compile_full_text_report(results)
    visuals = create_all_visuals(results)
    pdf_path, txt_path = build_reports(results, text_report, visuals)

    # optional temp file cleanup flag (manual removal after Streamlit download)
    for p in [pdf_path, txt_path] + list(visuals.values()):
        if not os.path.exists(p):
            continue
    return pdf_path, txt_path
