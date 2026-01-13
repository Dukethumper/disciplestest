# config.py

# Application mode
PUBLIC_MODE = True       # if True, users can generate reports via Streamlit
REPORT_MODE = "full"     # "summary" or "full" report detail level

# File paths
CSV_PATH = "results_master.csv"
REPORT_PATH = "reports/"

# Archetype system metadata
ARCHETYPE_COUNT = 12
MOTIVATION_COUNT = 12
DEFAULT_TEMPLATE = "templates/report_template.html"
