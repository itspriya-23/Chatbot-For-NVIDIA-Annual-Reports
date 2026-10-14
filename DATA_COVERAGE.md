# NVIDIA Document Coverage & Data Ingestion Audit (1998 – 2026)

This document provides a comprehensive audit of the documents ingested into the NVIDIA Annual Reports & Proxies RAG Chatbot, fulfilling data provenance and transparency requirements:
> *"Clearly document which files your system actually ingests and any gaps in coverage."*

---

## 1. Executive Summary & Archive Overview

NVIDIA Corporation became a publicly traded company on January 22, 1999 (IPO on NASDAQ under symbol `NVDA`).
Its fiscal year follows a 52- or 53-week cycle ending on the last Sunday of January of the given calendar year (e.g., Fiscal Year 2024 ended January 28, 2024; Fiscal Year 2025 ended January 26, 2025; Fiscal Year 2026 ended late January 2026).

Documents are retrieved from the official [NVIDIA Investor Relations Archive](https://investor.nvidia.com/financial-info/annual-reports-and-proxies/default.aspx).

---

## 2. Ingested Documents Registry

The system ingests curated landmark filings representing each distinct era of NVIDIA's evolution:

| Filing Year | Document Type | Source File Name | File Size | Historical Significance & Coverage |
| :---: | :---: | :---: | :---: | :--- |
| **2026** | Form 10-K / Annual Report | `NVIDIA_2026_Annual_Report.pdf` | 15.5 MB | Full FY2026 Blackwell GPU deployment, Hopper & Sovereign AI infrastructure revenue. |
| **2025** | Form 10-K / Annual Report | `NVIDIA_2025_Annual_Report.pdf` | 48.8 MB | Accelerated computing scaling, H100/H200 adoption, generative AI enterprise expansion. |
| **2024** | Form 10-K / Annual Report | `NVIDIA_2024_Annual_Report.pdf` | 53.3 MB | Historic \$60.9B total revenue year; Data Center revenue surge to \$47.5B. |
| **2023** | Form 10-K / Annual Report | `NVIDIA_2023_Annual_Report.pdf` | 48.3 MB | Launch of ChatGPT inflection point; Ada Lovelace and Hopper architecture ramp. |
| **2020** | Form 10-K / Annual Report | `NVIDIA_2020_Annual_Report.pdf` | 23.3 MB | Mellanox acquisition closing, Turing GPU architecture, ray tracing expansion. |
| **2018** | Form 10-K / Annual Report | `NVIDIA_2018_Annual_Report.pdf` | 19.6 MB | Volta GPU architecture, Tensor Core introduction, initial deep learning enterprise wave. |
| **2015** | Form 10-K / Annual Report | `NVIDIA_2015_Annual_Report.pdf` | 30.1 MB | Maxwell architecture, Tegra mobile & automotive initiatives, early CUDA research. |
| **2008** | DEF 14A (Proxy Statement) | `NVIDIA_2008_Proxy_Statement.pdf` | 437 KB | Executive compensation, governance, board oversight during early CUDA GPU compute rollout. |
| **2004** | DEF 14A (Proxy Statement) | `NVIDIA_2004_Proxy_Statement.pdf` | 312 KB | GeForce FX era, governance structure, executive stock ownership during early PC graphics expansion. |
| **1999** | DEF 14A (Proxy Statement) | `NVIDIA_1999_Proxy_Statement.pdf` | 145 KB | First annual meeting of stockholders following NVIDIA's January 1999 IPO. |
| **1998** | DEF 14A (Proxy Statement) | `NVIDIA_1998_Proxy_Statement.pdf` | 196 KB | Earliest available governance filing in the NVIDIA SEC/IR archive covering founding team and stock incentive plans. |

---

## 3. Coverage Gaps & Format Discrepancies Explained

As noted across public filing archives:
> *"Document availability and formats may vary by year."*

Our audit identified the following document patterns across NVIDIA's 28-year public archive:

1. **Format Evolution (1998–2010 vs. 2011–2026):**
   - **1998–2010:** In NVIDIA's investor relations web archive, earlier filings are primarily preserved as official Proxy Statements (`DEF 14A`) and SEC ASCII/SGML filings. Full-color glossy Annual Reviews were not uniformly published in modern high-resolution PDF format prior to 2011.
   - **2011–2026:** NVIDIA provides complete, high-fidelity PDF Annual Reports containing the Chairman's Letter to Stockholders, Financial Highlights, and full Form 10-K consolidated financial statements.
2. **Intermediate Gap Filing Availability:**
   - Years between 2000 and 2007, and 2016–2017 are available as individual SEC EDGAR Form 10-K text filings on `sec.gov/edgar`.
   - The included automated script (`src/downloader.py`) can be extended with any additional direct URLs as needed.
3. **Fiscal Year vs Calendar Year Alignment:**
   - NVIDIA's fiscal year ends in late January.
   - *Example:* **Fiscal Year 2024** corresponds to the 12-month period ended **January 28, 2024**. Questions regarding "2023 calendar year performance" correspond to NVIDIA's **FY2024** 10-K. The system's prompt instructions explicitly preserve and clarify this distinction.

---

## 4. Citation and Verification Guarantee

Every extracted chunk is permanently stamped with:
- `source_file`: The exact source PDF filename (e.g. `NVIDIA_2024_Annual_Report.pdf`)
- `year`: The corresponding fiscal year (e.g. `2024`)
- `doc_type`: Either `Annual Report` or `Proxy Statement`
- `page_number`: The exact 1-indexed page number within the original PDF

This guarantees that every answer produced by the chatbot can be verified directly against NVIDIA's original documents by an auditor or analyst.

