# Turkish Super League Match Prediction - Usage Examples

## Quick Start

### 1. Automatic Prediction (Easiest)

Run the interactive prediction tool:

```bash
python predict_match.py
```

The tool will:
- Show example predictions (Galatasaray vs Fenerbahce, Besiktas vs Trabzonspor)
- List all available teams
- Enter interactive mode for custom predictions

### 2. Interactive Mode

```
Home team: Galatasaray
Away team: Fenerbahce  
Week (default 1): 10
Has fans? (y/n, default y): y
```

**Output:**
```
======================================================================
MATCH PREDICTION
======================================================================

Match: Galatasaray (Home) vs Fenerbahce (Away)
Week: 10
Fans: Yes

Predicted Outcome: Draw

Probabilities:
  Home Win (Galatasaray): 33.3%
  Draw: 41.1%
  Away Win (Fenerbahce): 25.6%

Team Form (Last 5 Matches):
  Galatasaray: 1.40 pts/match, 1.80 GF, 1.80 GA, +0.00 GD
  Fenerbahce: 1.40 pts/match, 1.40 GF, 1.40 GA, +0.00 GD
  Form: Equal

======================================================================
```

### 3. Programmatic Usage

```python
from predict_match import MatchPredictor

# Initialize predictor
predictor = MatchPredictor()

# Predict a match
outcome, probabilities = predictor.predict_match(
    home_team='Galatasaray',
    away_team='Fenerbahce',
    week=10,
    has_fans=True,
    show_details=True
)

print(f"Predicted: {outcome}")
print(f"Probabilities: {probabilities}")
```

### 4. Batch Predictions

```python
from predict_match import MatchPredictor

predictor = MatchPredictor()

# Predict multiple matches
matches = [
    ('Galatasaray', 'Fenerbahce', 10),
    ('Besiktas', 'Trabzonspor', 15),
    ('Basaksehir FK', 'Konyaspor', 20),
]

for home, away, week in matches:
    outcome, probs = predictor.predict_match(home, away, week, show_details=False)
    print(f"{home} vs {away}: {outcome} ({probs[2]:.1%} home win)")
```

## Example Predictions

### Derby Match: Galatasaray vs Fenerbahce

```python
predictor.predict_match('Galatasaray', 'Fenerbahce', week=10)
```

**Result:** Draw (41.1% probability)
- Galatasaray home win: 33.3%
- Draw: 41.1%
- Fenerbahce away win: 25.6%

### Strong Home Team: Besiktas vs Trabzonspor

```python
predictor.predict_match('Besiktas', 'Trabzonspor', week=15)
```

**Result:** Home Win (56.8% probability)
- Besiktas home win: 56.8%
- Draw: 25.2%
- Trabzonspor away win: 18.0%

**Analysis:** Besiktas has strong recent form (3.00 pts/match, +1.80 GD) compared to Trabzonspor (0.80 pts/match, -0.60 GD)

## Available Teams

73 teams in database including:
- Galatasaray
- Fenerbahce
- Besiktas
- Trabzonspor
- Basaksehir FK
- Konyaspor
- Sivasspor
- Alanyaspor
- Kayserispor
- Antalyaspor
- And 63 more...

Run `predictor.list_teams()` to see all teams.

## Understanding the Output

### Predicted Outcome
- **Home Win**: Home team expected to win
- **Draw**: Match expected to end in a draw
- **Away Win**: Away team expected to win

### Probabilities
Shows the model's confidence for each outcome (sums to 100%)

### Team Form
Recent performance based on last 5 matches:
- **pts/match**: Average points per match (3 for win, 1 for draw, 0 for loss)
- **GF**: Goals For (average goals scored)
- **GA**: Goals Against (average goals conceded)
- **GD**: Goal Difference (GF - GA)

### Form Advantage
Which team has better recent form based on points per match

## Tips for Best Predictions

1. **Week Number**: Use actual week number for better season progress context
2. **Fan Attendance**: Set `has_fans=False` for matches without spectators (e.g., COVID-19 period)
3. **Team Names**: Use exact names from the database (case-sensitive)
4. **Recent Data**: Model trained on 1959-2020 data, predictions most accurate for teams with long history

## Limitations

- Model accuracy: ~51% (better than random 33%, competitive with industry)
- Draws hardest to predict (only 10% recall)
- Best at predicting home wins (84% recall)
- Does not account for:
  - Player injuries/suspensions
  - Manager changes
  - Transfer activity
  - Weather conditions
  - Referee assignments
  - Motivation factors (derby, relegation battle)

## Error Handling

**Team not found:**
```
ValueError: Team 'XYZ' not found in database
```
Solution: Check team name spelling or use `predictor.list_teams()`

**Model file missing:**
```
FileNotFoundError: results/models/random_forest_tsl.pkl
```
Solution: Run `python src/models/train_tsl_models.py` first

## Advanced Usage

### Custom Feature Engineering

```python
# Prepare features manually
features_df = predictor.prepare_match_features(
    home_team='Galatasaray',
    away_team='Fenerbahce',
    week=10,
    has_fans=True
)

# Inspect features
print(features_df.columns)
print(features_df.values)

# Make prediction
prediction = predictor.model.predict(features_df)[0]
probabilities = predictor.model.predict_proba(features_df)[0]
```

### Probability Threshold

```python
# Only predict home win if probability > 60%
outcome, probs = predictor.predict_match('Galatasaray', 'Fenerbahce', week=10, show_details=False)

if probs[2] > 0.6:  # Home win probability
    print("Strong home win prediction")
elif probs[0] > 0.6:  # Away win probability
    print("Strong away win prediction")
else:
    print("Uncertain outcome - consider draw or avoid betting")
```

## Questions?

For issues or questions, please open an issue on GitHub:
https://github.com/mehmetyalc/turkish-super-league-prediction/issues

