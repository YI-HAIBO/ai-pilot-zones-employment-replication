# AI Pilot Zones, Productivity Dividend, and Structural Employment Risk

**Replication package — data and code only**

This repository contains the data and estimation code for the paper:

> *AI Pilot Zones, Productivity Dividend, and Structural Employment Risk* (Applied Economics, submission)

It does **not** contain the manuscript, figures, or analysis reports. The manuscript and
its analysis output are available from the authors.

## Contents

```
code/                     Estimation and data-construction scripts (11 files)
data/
  did_city_panel_2017_2025_v8.csv    City-year panel (2,655 obs; 295 cities; 2017-2025)
  geo/W_knn5.json                    k-nearest-neighbour spatial weight matrix (k=5)
  geo/city_geo.csv                   City coordinates for spatial analyses
  raw/jobs/ai_jobs_v3_20261025.csv   AI job postings (job-board, anonymised)
  raw/jobs/commercial_jobs_20261025.csv  Non-AI (commercial) job postings (job-board)
  analysis/jobs_city_exposure.csv    City-level AI job exposure aggregates
```

## Reproduction

All analysis scripts compute the root directory from their own location, so the package
is **path-portable** — clone/copy it anywhere and run:

```bash
python code/did_v8_full.py      # main TWFE + event study + Bacon + CS + PSM + placebo
python code/did_v8_robust.py    # COVID-drop, regional heterogeneity, time placebo
python code/did_v8_sdm.py       # spatial DID (Moran's I, SUTVA / indirect effects)
python code/did_v8_cluster.py   # cluster-robust and wild-cluster-bootstrap inference
python code/w9_jobs_exposure.py # AI job exposure aggregation
python code/w11_mechanism.py    # panel mediation/moderation
python code/w12_mechanism_boost.py  # job-level substitute-vs-augment wage DD
python code/generate_figures_en.py  # regenerates the manuscript figures
```

Outputs are written to `data/analysis/*.md` inside the package (git-ignored).

### Dependencies

Python 3.10+, with `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`.

### Note on `build_panel_v8.py` / `parse_v5.py`

These two scripts reconstruct the v8 panel from the raw statistical-yearbook HTML.
The raw HTML is **not distributed** (it is scraped from public Chinese statistical
yearbook pages and is large); the panel is provided directly as
`data/did_city_panel_2017_2025_v8.csv`. The scripts are included for transparency and
to document the variable definitions. The DID analyses run entirely from the provided
panel.

## Data sources

- City-level GDP, income, employment, population, and structure: public statistical
  yearbooks (national, provincial, and city editions, 2017–2025).
- AI pilot-zone designation (18 cities; 2019/2020/2021 batches): Ministry of Science
  and Technology (MOST) public announcements.
- Job postings: anonymised job-board data (used only for the exposure and wage
  substitutability analyses).

## Key result (summary)

Across the 2019–2021 staggered adoption of national AI pilot zones, treated cities show
no significant acceleration in total employment or GDP growth, but a significant
positive effect on income per capita (+2,856 yuan, p = 0.011). The evidence is
consistent with a *structural* (compositional) employment risk — job substitution and
skill mismatch — rather than a *cyclical* unemployment effect.

## Citation

If you use this package, please cite the paper and this replication repository.

## License

Data and code are released for reproducibility. Data remain the property of their
original public sources; job-board data are aggregated/anonymised and are not for
redistribution in identifiable form.
