from fpdf import FPDF
import os
from config import REPORT_PATH  # ✅ make sure this is defined as "reports/" in config.py

# ---------------------------------------------------------------------
# Save text report
# ---------------------------------------------------------------------
def save_text_report(text: str, filename: str = "Personality_Report.txt") -> str:
    # ✅ Save directly into the reports folder, not temp
    if not os.path.exists(REPORT_PATH):
        os.makedirs(REPORT_PATH, exist_ok=True)

    path = os.path.join(REPORT_PATH, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# ---------------------------------------------------------------------
# PDF report class
# ---------------------------------------------------------------------
class PDFReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Personality Report", ln=True, align="C")
        self.ln(5)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, title, ln=True)
        self.ln(2)

    def section_body(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, text)
        self.ln(4)


# ---------------------------------------------------------------------
# Save PDF report
# ---------------------------------------------------------------------
def save_pdf_report(text: str, visuals: dict, filename: str = "Personality_Report.pdf") -> str:
    if not os.path.exists(REPORT_PATH):
        os.makedirs(REPORT_PATH, exist_ok=True)

    pdf = PDFReport()
    pdf.add_page()

    # Split text into sections using === markers (produced by section_writer)
    sections = text.split("===")
    for section in sections:
        section = section.strip()
        if not section:
            continue
        lines = section.split("\n", 1)
        title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        pdf.section_title(title)
        pdf.section_body(body)

        if "Motivational Hierarchy" in title and "Motivational_Hierarchy" in visuals:
            pdf.image(visuals["Motivational_Hierarchy"], w=140)
        if "Behavioral Indices" in title and "Behavioral_Indices" in visuals:
            pdf.image(visuals["Behavioral_Indices"], w=120)
        if "Advanced Extensions" in title and "Temporal_Focus" in visuals:
            pdf.image(visuals["Temporal_Focus"], w=100)
        if "Appendix" in title and "Personality_Balance" in visuals:
            pdf.image(visuals["Personality_Balance"], w=120)

    path = os.path.join(REPORT_PATH, filename)
    pdf.output(path)
    return path


# ---------------------------------------------------------------------
# Build both text + PDF reports
# ---------------------------------------------------------------------
def build_reports(results: dict, text_report: str, visuals: dict) -> tuple:
    participant_id = results.get("participant_id", "user")
    txt_name = f"{participant_id}_report.txt"
    pdf_name = f"{participant_id}_report.pdf"

    txt_path = save_text_report(text_report, txt_name)
    pdf_path = save_pdf_report(text_report, visuals, pdf_name)

    # ✅ Return full paths for Streamlit to use
    return pdf_path, txt_path
