# CITATIONS_TO_VERIFY.md — Registry of Unverified Literature and Guideline References

> **Purpose:** This document tracks every reference to academic literature, government manuals, statutory guidelines, or engineering standards used in the project.
> In accordance with project integrity rules, any value or rule not directly verified against an authoritative primary document is designated:
> **"assumed expert-style judgement; literature support UNVERIFIED"**.
> The repository owner should review and verify each item below before applying this pipeline to real-world watershed planning.

---

## Registry of Items Requiring Verification

| ID | Item / Claim | Current Project Location(s) | Claimed Source / Citation | Verification Task for Owner |
|---|---|---|---|---|
| **C-01** | Check dam AHP siting criteria and ranking weights | `src/ahp.py` (lines 25, 175)<br>`DECISIONS.md` (D-11, line 168)<br>`config.yaml` (`ahp:` block) | Previously cited Mukherjee et al. (2012); Nitheshnirmal et al. (2019) | Check whether Mukherjee (2012) and Nitheshnirmal (2019) specifically evaluate check dam siting in semi-arid Karnataka and report these exact criteria. |
| **C-02** | Stream order range for check dams (Order 1 to 3) | `config.yaml` (lines 55-56)<br>`src/constraints.py` (lines 18, 73)<br>`DECISIONS.md` (D-10) | IMSD (Integrated Mission for Sustainable Development) Guidelines / Karnataka Minor Irrigation Manual | Confirm that Karnataka Minor Irrigation Department restricts masonry/earthen check dams strictly to 1st–3rd order streams. |
| **C-03** | Slope threshold range ($1.0^\circ$ to $15.0^\circ$) | `config.yaml` (lines 57-58)<br>`src/constraints.py` (lines 75-76)<br>`DECISIONS.md` (D-10) | IMSD Guidelines / Central Ground Water Board (CGWB) artificial recharge manual | Verify upper slope limit for check dam structures ($15^\circ$ vs $10^\circ$ vs $5^\circ$) in Deccan hard-rock terrain. |
| **C-04** | Minimum catchment area ($\ge 1.0\text{ km}^2$) | `config.yaml` (line 59)<br>`src/constraints.py` (line 77)<br>`DECISIONS.md` (D-10) | IMSD check dam siting guidelines | Verify minimum contributing watershed area required for check dam storage efficiency in dry/semi-arid taluks. |
| **C-05** | Maximum valley width ($\le 120.0\text{ m}$) | `config.yaml` (line 60)<br>`src/constraints.py` (line 78)<br>`DECISIONS.md` (D-10) | Minor Irrigation design standard | Check economic crest length threshold for minor check dams versus major barrages. |
| **C-06** | Settlement buffer ($\ge 150\text{ m}$) | `config.yaml` (line 65)<br>`src/constraints.py` (line 82)<br>`DECISIONS.md` (D-10) | Disaster management & backwater inundation safety norms | Confirm statutory or standard engineering safety setback from habitation. |
| **C-07** | Construction cost proxy using Karnataka DWMS rates | `DECISIONS.md` (D-05)<br>`README.md` (line 125) | Karnataka Department of Watershed Development / DWMS tender data | Check availability of current Public Works Department (PWD) / Watershed Schedule of Rates (SoR). |
| **C-08** | Synthetic feature statistical distributions | `ARCHITECTURE.md` (line 127)<br>`src/generate_synthetic.py` (line 7) | Calibrated to Karnataka hydrology | Note: The distributions are mathematical approximations for pipeline testing, not field measurements. |

---

## Action Plan for Real Data (Stage 2 & Beyond)
1. Procure the official Karnataka Watershed Development Department or Minor Irrigation design manual.
2. Replace each item marked "literature support UNVERIFIED" with the verified chapter, table, and page number, or update the threshold in `config.yaml` to the officially prescribed standard.
3. Update this document with the verification date and primary source link/file.
