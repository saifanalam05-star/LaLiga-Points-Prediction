# La Liga 2026/27 Points Prediction

A beginner-friendly machine learning project that forecasts **2026/27 La Liga final points and league standings** using historical team statistics, **Linear Regression**, and a separate historical baseline for newly promoted clubs.

The project follows a complete data science workflow: **data preparation → exploratory analysis → feature engineering → model training → evaluation → predictions**.

## Project overview

The main question is: **Can a team's previous-season performance help predict how many points it will earn in the following La Liga season?**

- **Returning clubs (17):** A Linear Regression model predicts next-season points using previous-season performance.
- **Promoted clubs (3):** A historical-average baseline estimates points because these clubs have no previous-season La Liga statistics.
- **Output:** A 20-team projected standings table, with predicted points rounded to whole numbers for presentation.

> **Note:** These are statistical estimates, not official standings or guaranteed outcomes. Teams tied on projected points are not ranked using official league tiebreakers.

## Tools and technologies

- **Python** — analysis and modelling
- **pandas / NumPy** — data manipulation and calculations
- **Matplotlib** — charts and visualizations
- **scikit-learn** — Linear Regression and model evaluation
- **Jupyter Notebook / VS Code** — development environment

## Repository structure

```text
LinearRegression/
├── data/
│   ├── laligamatches.csv
│   ├── laliga_2025_2026.csv
│   ├── laliga_cleaned.csv
│   └── laliga_2026_2027_predictions.csv
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   └── 03_linear_regression.ipynb
└── README.md
```

## Dataset and preparation

The cleaned dataset contains **240 team-season observations** across **12 La Liga seasons (2014/15–2025/26)**, with 20 clubs per season.

Historical fields include league points, wins, draws, losses, goals scored, goals conceded, goal difference, and other season statistics. Data preparation involved:

1. Combining historical records with the 2025/26 season standings.
2. Checking for duplicate club-season records and inconsistent statistics.
3. Reviewing missing values, including unavailable expected-goals and attendance data.
4. Standardizing team names (for example, `Real Betis` and `Betis`) so the same club is consistently recognized across seasons.
5. Saving the processed dataset to `data/laliga_cleaned.csv`.

**Data source:** Historical La Liga standings were assembled from FBref season tables, with the 2025/26 standings added separately. Verify and cite the specific original source pages and applicable data-use terms before redistributing raw datasets.

## Exploratory data analysis

The EDA notebook investigates the distribution of points and relationships between season performance measures.

Selected findings from the dataset:

| Measure | Result |
|---|---:|
| Average team points per season | 52.08 |
| Median team points | 48 |
| Correlation: goals scored vs. points (same season) | 0.862 |
| Correlation: goal difference vs. points (same season) | 0.961 |

These correlations describe **same-season** relationships. To avoid data leakage, the forecasting model uses **completed previous-season statistics**, not statistics from the season it is trying to predict.

## Methodology

### 1. Create next-season training examples

For each club, the target is its points total in the *following* season:

```python
laliga["next_season_pts"] = laliga.groupby("squad")["pts"].shift(-1)
laliga["next_season"] = laliga.groupby("squad")["season"].shift(-1)

training_data = laliga[
    laliga["next_season"] == laliga["season"] + 1
].copy()
```

Only consecutive top-flight seasons are retained. This avoids treating a club's return after relegation as though it happened one season later.

### 2. Train a Linear Regression model

The model uses five previous-season features:

| Feature | Meaning |
|---|---|
| `pts` | League points |
| `w` | Wins |
| `gf` | Goals scored |
| `ga` | Goals conceded |
| `gd` | Goal difference |

**Target:** `next_season_pts` (points in the following La Liga season).

Data is split **chronologically**, using earlier seasons for training and later seasons for testing, rather than randomly mixing seasons.

### 3. Evaluate against a simple baseline

Model performance is measured with:

- **MAE (Mean Absolute Error):** Average prediction error in points.
- **RMSE (Root Mean Squared Error):** Error metric that penalizes large misses more heavily.
- **R²:** How much variation in next-season points is explained on the test set.

The **initial evaluation, before fixing a club-name inconsistency**, produced:

| Approach | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 7.95 | 10.92 | 0.606 |
| Previous-season points baseline | 8.00 | 11.44 | 0.567 |

The model slightly outperformed simply predicting that a team would repeat its previous points total. **These metrics must be recalculated after the Betis name-cleaning fix before being presented as final evaluation results.**

A three-season rolling-points feature was also tested, but did not improve validation performance, so the simpler five-feature model was retained.

### 4. Predict newly promoted teams

Promoted teams do not have previous-season La Liga statistics, so they are estimated separately using the historical points average for newly promoted clubs.

- Historical promoted club-seasons: **33**
- Historical average: **40.15 points**
- Expanding-window backtest MAE: **6.56 points**
- Expanding-window backtest RMSE: **9.27 points**

This method gives all promoted teams the same estimate; it **does not** use Segunda División results or distinguish their individual strengths. Its error metrics are calculated on a different sample than the regression model's, so the two MAEs should not be directly compared.

## 2026/27 predicted standings

| Position* | Club | Predicted points | Method |
|---:|---|---:|---|
| 1 | Barcelona | 87 | Linear Regression |
| 2 | Real Madrid | 78 | Linear Regression |
| 3 | Villarreal | 67 | Linear Regression |
| 4 | Atlético Madrid | 64 | Linear Regression |
| 5 | Betis | 57 | Linear Regression |
| 6 | Celta Vigo | 53 | Linear Regression |
| 7 | Getafe | 50 | Linear Regression |
| 8 | Rayo Vallecano | 49 | Linear Regression |
| 9 | Real Sociedad | 47 | Linear Regression |
| 10 | Valencia | 47 | Linear Regression |
| 11 | Osasuna | 45 | Linear Regression |
| 12 | Espanyol | 45 | Linear Regression |
| 13 | Elche | 44 | Linear Regression |
| 14 | Athletic Club | 44 | Linear Regression |
| 15 | Alavés | 44 | Linear Regression |
| 16 | Sevilla | 43 | Linear Regression |
| 17 | Levante | 43 | Linear Regression |
| 18 | Racing Santander | 40 | Historical Average |
| 19 | La Coruña | 40 | Historical Average |
| 20 | Málaga | 40 | Historical Average |

*Positions reflect sorting by predicted points prior to rounding. Whole-number ties are **not** resolved using La Liga's official tiebreak rules. In particular, the three promoted clubs all have the same baseline forecast and their order is arbitrary.*

The complete exported results are in [`data/laliga_2026_2027_predictions.csv`](data/laliga_2026_2027_predictions.csv).

## How to run

1. Clone the repository and open it in **VS Code** or Jupyter.
2. Create and activate a Python virtual environment (recommended).
3. Install the dependencies:

   ```bash
   pip install pandas numpy matplotlib scikit-learn notebook
   ```

   If the EDA notebook imports Seaborn, also run `pip install seaborn`.
4. Run the notebooks **in order**, from top to bottom:
   - `01_data_preparation.ipynb`
   - `02_exploratory_analysis.ipynb`
   - `03_linear_regression.ipynb`
5. Check the final predictions CSV in the `data/` directory.

> Notebook paths assume the notebooks are run from within the `notebooks/` directory and access CSV files using `../data/`.

## Limitations

- The model does not directly account for transfers, injuries, managerial changes, or club finances.
- Promoted clubs receive the same historical-average forecast rather than club-specific predictions.
- `gd = gf - ga`, so the five-feature regression has redundant inputs; individual coefficients should not be interpreted as independent effects.
- Forecasts are point estimates, not probabilities or confidence intervals.
- Ranking teams on predicted points does not enforce real-world league tiebreakers or match-by-match consistency.
- Large year-to-year changes in club performance can produce substantial errors.

## Future improvements

- Add Segunda División data for club-specific promoted-team forecasts.
- Compare Ridge Regression, Random Forest, and gradient-boosting models using the same chronological evaluation.
- Evaluate over multiple rolling train/test windows.
- Estimate uncertainty ranges around projected points.
- Build an interactive dashboard and track predictions against actual results.

## What I learned

This project gave me hands-on experience with **data cleaning, EDA, feature selection, supervised learning, chronological model evaluation, baseline comparisons, and communicating model limitations**. It also reinforced that accurate data preparation and honest evaluation are as important as the model itself.
