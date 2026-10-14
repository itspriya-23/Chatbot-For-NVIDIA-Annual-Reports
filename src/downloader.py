"""
Automated downloader for NVIDIA Annual Reports and Proxy Statements (1998 - 2026).
Fetches official PDF filings from NVIDIA Investor Relations archive.
"""

import os
import re
import requests
import certifi

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

# Curated landmark filings covering the full historical span (1998 - 2026)
# Including both Annual Reports (10-K) and Proxy Statements (DEF 14A)
OFFICIAL_DOCS = [
    # Modern Landmark Era (AI Boom & Accelerated Computing)
    {
        "year": 2026,
        "type": "Annual Report",
        "filename": "NVIDIA_2026_Annual_Report.pdf",
        "url": "https://s201.q4cdn.com/141608511/files/doc_financials/2026/ar/2026-Annual-Report-Web.pdf"
    },
    {
        "year": 2025,
        "type": "Annual Report",
        "filename": "NVIDIA_2025_Annual_Report.pdf",
        "url": "https://s201.q4cdn.com/141608511/files/doc_financials/2025/annual/NVIDIA-2025-Annual-Report.pdf"
    },
    {
        "year": 2024,
        "type": "Annual Report",
        "filename": "NVIDIA_2024_Annual_Report.pdf",
        "url": "https://s201.q4cdn.com/141608511/files/doc_financials/2024/ar/2024-annual-report-insert-web-IR-update.pdf"
    },
    {
        "year": 2023,
        "type": "Annual Report",
        "filename": "NVIDIA_2023_Annual_Report.pdf",
        "url": "https://s201.q4cdn.com/141608511/files/doc_financials/2023/ar/2023-annual-report-insert-web-IR-update.pdf"
    },
    {
        "year": 2020,
        "type": "Annual Report",
        "filename": "NVIDIA_2020_Annual_Report.pdf",
        "url": "https://s201.q4cdn.com/141608511/files/doc_financials/2020/ar/2020-annual-report-insert-web-ir-update.pdf"
    },
    # Intermediate Era (Deep Learning & GPU Computing emergence)
    {
        "year": 2018,
        "type": "Annual Report",
        "filename": "NVIDIA_2018_Annual_Report.pdf",
        "url": "https://s201.q4cdn.com/141608511/files/doc_financials/2018/ar/2018_annual-report-insert-web-ir-update.pdf"
    },
    {
        "year": 2015,
        "type": "Annual Report",
        "filename": "NVIDIA_2015_Annual_Report.pdf",
        "url": "https://s201.q4cdn.com/141608511/files/doc_financials/2015/ar/2015-annual-report-insert-web-ir-update.pdf"
    },
    # Early & Founding Era (CUDA launch, Mobile, PC 3D graphics)
    {
        "year": 2008,
        "type": "Proxy Statement",
        "filename": "NVIDIA_2008_Proxy_Statement.pdf",
        "url": "https://s201.q4cdn.com/141608511/files/doc_financials/annual/2008/SEC-AMDA-1XAJD4-1193125-09-75685.pdf"
    },
    {
        "year": 2004,
        "type": "Proxy Statement",
        "filename": "NVIDIA_2004_Proxy_Statement.pdf",
        "url": "https://s201.q4cdn.com/141608511/files/doc_financials/annual/2004/SEC-AMDA-1XAJD4-1047469-05-16088.pdf"
    },
    {
        "year": 1999,
        "type": "Proxy Statement",
        "filename": "NVIDIA_1999_Proxy_Statement.pdf",
        "url": "https://s201.q4cdn.com/141608511/files/doc_financials/annual/1999/SEC-AMDA-1XAJD4-1012870-00-3122.pdf"
    },
    {
        "year": 1998,
        "type": "Proxy Statement",
        "filename": "NVIDIA_1998_Proxy_Statement.pdf",
        "url": "https://s201.q4cdn.com/141608511/files/doc_financials/annual/1998/SEC-AMDA-1XAJD4-1012870-99-1632.pdf"
    }
]

def download_file(url: str, dest_path: str) -> bool:
    """Download a file with progress display."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"  [Cached] Already exists: {os.path.basename(dest_path)} ({os.path.getsize(dest_path)/1024:.1f} KB)")
        return True

    print(f"  [Downloading] {url} -> {os.path.basename(dest_path)}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, verify=certifi.where(), stream=True, timeout=30)
        if response.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            print(f"  [Success] Saved {os.path.basename(dest_path)} ({os.path.getsize(dest_path)/1024:.1f} KB)")
            return True
        else:
            print(f"  [Failed] HTTP {response.status_code} for {url}")
            return False
    except Exception as e:
        print(f"  [Error] Failed to download {url}: {e}")
        return False

def download_all():
    print("=" * 70)
    print("NVIDIA Annual Reports & Proxy Statements Downloader (1998 - 2026)")
    print("=" * 70)
    success_count = 0
    for doc in OFFICIAL_DOCS:
        dest = os.path.join(DATA_DIR, doc["filename"])
        if download_file(doc["url"], dest):
            success_count += 1
    print("=" * 70)
    print(f"Downloaded / Verified {success_count}/{len(OFFICIAL_DOCS)} documents into {DATA_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    download_all()

