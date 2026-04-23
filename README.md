# PlayStation Gaming Behavior Analysis

**Course:** CPSC 222 — Introduction to Data Science, Spring 2025  
**Author:** Kyle Hays  

---

## Project Overview

This project analyzes personal PlayStation gaming behavior using two datasets:
1. A personal PlayStation Network playtime tracker (198 titles)
2. PlayStation Store official annual top-download charts (2023–2025, 200 entries)

The goal is to explore gaming habits through exploratory data analysis and train kNN and Decision Tree classifiers to predict **Engagement Tier** (Casual / Moderate / Engaged / Hardcore) based on session behavior, genre, and platform.

---

## Project Structure

```
ps_project/
│
├── PS_Gaming_Analysis.ipynb          # Main Jupyter Notebook (report + code)
├── utils.py                          # All reusable utility functions
│
├── personal_gaming_tracking.csv      # Raw input: personal PS tracking data
├── playstation_store_top_downloads.csv  # Raw input: PS Store top charts
│
├── cleaned_personal_tracking.csv     # Cleaned personal data (output)
├── cleaned_store_downloads.csv       # Cleaned store data (output)
├── merged_gaming_dataset.csv         # Merged + feature-engineered (output)
│
└── README.md                         # This file
```

---

## How to Run

### 1. Dependencies

Make sure you have Python 3.8+ and the following packages installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

### 2. Launch the Notebook

From inside the `ps_project/` directory:

```bash
jupyter notebook PS_Gaming_Analysis.ipynb
```

Or with JupyterLab:

```bash
jupyter lab PS_Gaming_Analysis.ipynb
```

### 3. Run All Cells

Use **Kernel → Restart & Run All** to execute the complete analysis from start to finish. All output CSVs will be regenerated automatically.

---

## Dataset Descriptions

### `personal_gaming_tracking.csv` (Input)
| Column | Type | Description |
|--------|------|-------------|
| Rank | int | Rank by hours played |
| Game | str | Title name |
| Platform | str | PS4 or PS5 |
| Hours Played | float | Total hours (sub-hour entries as decimals) |
| Sessions | int | Number of play sessions |
| First Played | date | Date first launched |
| Last Played | date | Most recent session date |
| Last Update | date | Data export date |

### `playstation_store_top_downloads.csv` (Input)
| Column | Type | Description |
|--------|------|-------------|
| Rank | int | Chart position within category/year/region |
| Game | str | Title name |
| Category | str | PS5 Games or PS4 Games |
| Year | int | Chart year (2023–2025) |
| Region | str | US/Canada or EU |
| Platform | str | PS4 or PS5 |
| Source | str | PlayStation Blog |

### Output CSVs
- **`cleaned_personal_tracking.csv`** — Media apps removed, derived features added (Days Active, Avg Session Hr, Genre, Engagement Tier)
- **`cleaned_store_downloads.csv`** — Deduplicated, whitespace-cleaned
- **`merged_gaming_dataset.csv`** — Personal tracking joined with store chart flag (`Was_Top_Download`)

---

## Classification Task

- **Target:** `Engagement Tier` (Casual: 0–10 hrs | Moderate: 10–50 hrs | Engaged: 50–150 hrs | Hardcore: 150+ hrs)
- **Features:** Sessions, Avg Session Hr, Days Active, Platform Generation (encoded), Genre (encoded)
- **Models:** k-Nearest Neighbors (k tuned by 5-fold CV), Decision Tree (max_depth tuned by 5-fold CV)

---

## Sources

- Personal PlayStation Network playtime data (https://ps-timetracker.com/profile/Haneisuru/official_playtimes)
- [PlayStation Blog — Top Downloads 2023](https://blog.playstation.com/2024/01/23/playstation-stores-top-downloads-of-2023/)
- [PlayStation Blog — Top Downloads 2024](https://blog.playstation.com/2025/01/23/playstation-stores-top-downloads-of-2024/)
- [PlayStation Blog — Top Downloads 2025](https://blog.playstation.com/2026/01/14/playstation-stores-top-downloads-of-2025/)
- scikit-learn: https://scikit-learn.org/stable/
- pandas: https://pandas.pydata.org/
