import os
from tqdm import tqdm
import pandas as pd

# Import internal modules
from modules.data_loader import extract_user_data, preprocess_data
from modules.analysis_engine import compile_all_metrics
from modules.section_writer import compile_full_text_report
from modules.visualizer import create_all_visuals
from modules.report_builder import build_reports

# Optional configuration imports
try:
    from config import CSV_PATH, REPORT_PATH, REPORT_MODE
except ImportError:
    # Fallbacks if config.py is not present
    CSV_PATH = "results_master.csv"
    REPORT_PATH = "reports/"
    REPORT_MODE = "full"


def generate_all_reports():
    """Main batch generation function."""

    # Ensure input file exists
    if not os.path.exists(CSV_PATH):
        print(f"❌ Input file not found: {CSV_PATH}")
        return

    # Ensure output directory exists
    if not os.path.exists(REPORT_PATH):
        os.makedirs(REPORT_PATH)

    # Load data
    df = extract_user_data(CSV_PATH, mode="csv")
    df = preprocess_data(df)

    print(f"🧠 Loaded {len(df)} participant records from {CSV_PATH}\n")

    # Iterate through each participant
    for idx, row in tqdm(
        df.iterrows(),
        total=df.shape[0],
        desc="Generating Reports",
        ncols=90,
    ):
        user_record = row.to_dict()
        participant_id = str(user_record.get("participant_id", f"user_{idx+1}"))

        # Compute all analytical data
        results = compile_all_metrics(user_record)

        # Generate text report
        text_report = compile_full_text_report(results)

        # Create visuals
        visuals = create_all_visuals(results)

        # Build and save reports
        pdf_path, txt_path = build_reports(results, text_report, visuals)

        # Move finished reports into the output folder
        pdf_dest = os.path.join(REPORT_PATH, f"{participant_id}_report.pdf")
        txt_dest = os.path.join(REPORT_PATH, f"{participant_id}_report.txt")

        os.replace(pdf_path, pdf_dest)
        os.replace(txt_path, txt_dest)

    print(f"\n✅ All reports generated successfully!")
    print(f"📁 Saved in: {os.path.abspath(REPORT_PATH)}")


if __name__ == "__main__":
    generate_all_reports()
