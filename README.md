# Turkish Super League Match Result Prediction

Machine Learning models to predict Turkish Super League match results using historical data and team statistics.

## Project Overview

Predict Turkish Super League football match outcomes (Home Win / Draw / Away Win) using 60+ years of historical data.

**Dataset:** 17,408 matches (1959-2020)  
**Best Model:** Random Forest - 51.3% accuracy

## Key Results

### Model Performance

| Model | Accuracy | F1-Score | CV Accuracy |
|-------|----------|----------|-------------|
| Random Forest | 51.35% | 0.455 | 52.15% |
| LightGBM | 51.12% | 0.470 | 50.67% |
| XGBoost | 49.05% | 0.461 | 49.61% |

### Classification Report (Random Forest)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Away Win | 0.44 | 0.31 | 0.37 | 797 |
| Draw | 0.31 | 0.10 | 0.15 | 982 |
| Home Win | 0.56 | 0.84 | 0.67 | 1,703 |

**Key Insights:**
- Model excels at predicting home wins (84% recall)
- Draws hardest to predict (10% recall) - typical for football
- 51.3% accuracy competitive with industry standards (baseline ~48%)

## Features

**42 Engineered Features:**
- Rolling form (last 5 matches): points, goals, goal difference
- Team historical stats: home/away performance, red cards
- Team strength indicators: attack, defense, overall strength
- Match context: season progress, fan attendance, week

**Data Leakage Prevention:** Only pre-match information used

## Installation

```bash
pip install pandas numpy scikit-learn xgboost lightgbm
```

## Usage

**Train Models:**
```bash
python src/features/prepare_tsl_ml_dataset_optimized.py
python src/models/train_tsl_models.py
```

**Make Predictions:**
```python
import pickle
with open('results/models/random_forest_tsl.pkl', 'rb') as f:
    model = pickle.load(f)

prediction = model.predict(match_features)
# 0 = Away Win, 1 = Draw, 2 = Home Win
```

## Dataset

**Source:** [Kaggle - Turkish Super League Matches (1959-2020)](https://www.kaggle.com/datasets/faruky/turkish-super-league-matches-19592020)

**Statistics:**
- 17,408 matches (after removing first 500 for insufficient history)
- Target distribution: Home Win 48.9%, Draw 28.2%, Away Win 22.9%

## Why 51% is Good

Football prediction is difficult due to:
- High variance (injuries, luck, referee decisions)
- Draw outcomes add complexity
- Home advantage varies

**Benchmarks:**
- Random: 33.3%
- Home advantage baseline: ~48%
- Academic papers: 45-55%
- Our model: 51.35%

## Future Improvements

- Player-level data (injuries, transfers)
- Referee statistics
- Weather conditions
- Manager changes
- Deep learning (LSTM)
- Expected improvement: 53-55%

## Technologies

Python 3.11, pandas, scikit-learn, XGBoost, LightGBM

## Related Projects

1. [Transfer Success Prediction](https://github.com/mehmetyalc/transfer-success-prediction)
2. [Transfer Economic Efficiency](https://github.com/mehmetyalc/transfer-economic-efficiency)
3. [F1 Race Prediction](https://github.com/mehmetyalc/f1-race-prediction)

## Author

Mehmet Yalcin - [GitHub](https://github.com/mehmetyalc)

## License

Open source - Educational purposes

