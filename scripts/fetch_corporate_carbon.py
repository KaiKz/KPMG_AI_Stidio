from pathlib import Path
from urllib.request import urlretrieve

raw = Path("data/corporate_carbon/raw")
raw.mkdir(parents=True, exist_ok=True)

files = {
    # Google 2024 Environmental Report (official PDF)
    "google_2024_environmental_report.pdf": "https://www.gstatic.com/gumdrop/sustainability/google-2024-environmental-report.pdf",
    # Microsoft 2024 Environmental Sustainability Report (official PDF)
    "microsoft_2024_environmental_sustainability_report.pdf": "https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/msc/documents/presentations/CSR/Microsoft-2024-Environmental-Sustainability-Report.pdf",
    # Microsoft 2024 Environmental Data Fact Sheet (tables/methods)
    "microsoft_2024_env_data_fact_sheet.pdf": "https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/msc/documents/presentations/CSR/2024-Environmental-Sustainability-Report-Data-Fact.pdf",
}

for name, url in files.items():
    print(f"Downloading: {name}")
    urlretrieve(url, raw / name)

print("Saved files to:", raw.resolve())
