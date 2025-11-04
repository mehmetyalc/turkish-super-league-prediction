"""
Turkish Super League Match Prediction Tool
Predict match outcome for any two teams
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime

class MatchPredictor:
    """Predict Turkish Super League match results"""
    
    def __init__(self, model_path='results/models/random_forest_tsl.pkl', 
                 data_path='data/raw/tsl_dataset.csv'):
        """Initialize predictor with trained model and historical data"""
        print("Loading prediction model...")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print("  Model loaded successfully!")
        
        print("\nLoading historical data...")
        self.df = pd.read_csv(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        self.df = self.df.dropna(subset=['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Add points
        self.df['home_points'] = self.df['result'].map({'H': 3, 'D': 1, 'A': 0})
        self.df['visitor_points'] = self.df['result'].map({'H': 0, 'D': 1, 'A': 3})
        
        print(f"  Loaded {len(self.df)} historical matches")
        
        # Get unique teams
        self.teams = sorted(set(self.df['home'].unique()) | set(self.df['visitor'].unique()))
        print(f"  Found {len(self.teams)} teams in database")
    
    def get_team_recent_form(self, team, n_matches=5):
        """Calculate team's recent form (last N matches)"""
        # Get all matches for this team
        team_matches = self.df[(self.df['home'] == team) | (self.df['visitor'] == team)].tail(n_matches)
        
        if len(team_matches) == 0:
            return 0, 0, 0, 0  # No history
        
        points = 0
        goals_for = 0
        goals_against = 0
        
        for _, match in team_matches.iterrows():
            if match['home'] == team:
                goals_for += match['hgoal']
                goals_against += match['vgoal']
                points += match['home_points']
            else:
                goals_for += match['vgoal']
                goals_against += match['hgoal']
                points += match['visitor_points']
        
        avg_points = points / len(team_matches)
        avg_gf = goals_for / len(team_matches)
        avg_ga = goals_against / len(team_matches)
        avg_gd = avg_gf - avg_ga
        
        return avg_points, avg_gf, avg_ga, avg_gd
    
    def get_team_overall_stats(self, team, home=True):
        """Get team's overall statistics"""
        if home:
            team_data = self.df[self.df['home'] == team]
            if len(team_data) == 0:
                return 0, 0, 0, 0
            avg_gf = team_data['hgoal'].mean()
            avg_ga = team_data['vgoal'].mean()
            avg_points = team_data['home_points'].mean()
            avg_red = team_data['home_red_card'].mean()
        else:
            team_data = self.df[self.df['visitor'] == team]
            if len(team_data) == 0:
                return 0, 0, 0, 0
            avg_gf = team_data['vgoal'].mean()
            avg_ga = team_data['hgoal'].mean()
            avg_points = team_data['visitor_points'].mean()
            avg_red = team_data['visitor_red_card'].mean()
        
        return avg_gf, avg_ga, avg_points, avg_red
    
    def prepare_match_features(self, home_team, away_team, week=1, has_fans=True):
        """Prepare features for a match prediction"""
        print(f"\nPreparing features for: {home_team} vs {away_team}")
        
        # Check if teams exist
        if home_team not in self.teams:
            raise ValueError(f"Team '{home_team}' not found in database. Available teams: {', '.join(self.teams[:10])}...")
        if away_team not in self.teams:
            raise ValueError(f"Team '{away_team}' not found in database. Available teams: {', '.join(self.teams[:10])}...")
        
        # Initialize features in correct order
        features = {}
        
        # Must match training feature order exactly!
        
        # Basic features (in exact training order)
        features['Week'] = week
        features['tier'] = 1
        features['fans'] = 20000 if has_fans else 0
        features['neutral'] = 0
        
        # Home team rolling form
        h_points, h_gf, h_ga, h_gd = self.get_team_recent_form(home_team)
        features['home_rolling_points'] = h_points
        features['home_rolling_gf'] = h_gf
        features['home_rolling_ga'] = h_ga
        features['home_rolling_gd'] = h_gd
        
        # Away team rolling form
        a_points, a_gf, a_ga, a_gd = self.get_team_recent_form(away_team)
        features['visitor_rolling_points'] = a_points
        features['visitor_rolling_gf'] = a_gf
        features['visitor_rolling_ga'] = a_ga
        features['visitor_rolling_gd'] = a_gd
        
        # Form differences
        features['form_diff'] = h_points - a_points
        features['gd_diff'] = h_gd - a_gd
        
        # Home team home stats
        h_gf_home, h_ga_home, h_pts_home, h_red_home = self.get_team_overall_stats(home_team, home=True)
        features['avg_gf_home_h'] = h_gf_home
        features['avg_ga_home_h'] = h_ga_home
        features['avg_points_home_h'] = h_pts_home
        features['avg_red_home_h'] = h_red_home
        
        # Home team away stats
        h_gf_away, h_ga_away, h_pts_away, h_red_away = self.get_team_overall_stats(home_team, home=False)
        features['avg_gf_away_h'] = h_gf_away
        features['avg_ga_away_h'] = h_ga_away
        features['avg_points_away_h'] = h_pts_away
        features['avg_red_away_h'] = h_red_away
        
        # Home team overall
        features['avg_gf_overall_h'] = (h_gf_home + h_gf_away) / 2
        features['avg_ga_overall_h'] = (h_ga_home + h_ga_away) / 2
        features['avg_points_overall_h'] = (h_pts_home + h_pts_away) / 2
        
        # Away team home stats
        a_gf_home, a_ga_home, a_pts_home, a_red_home = self.get_team_overall_stats(away_team, home=True)
        features['avg_gf_home_v'] = a_gf_home
        features['avg_ga_home_v'] = a_ga_home
        features['avg_points_home_v'] = a_pts_home
        features['avg_red_home_v'] = a_red_home
        
        # Away team away stats
        a_gf_away, a_ga_away, a_pts_away, a_red_away = self.get_team_overall_stats(away_team, home=False)
        features['avg_gf_away_v'] = a_gf_away
        features['avg_ga_away_v'] = a_ga_away
        features['avg_points_away_v'] = a_pts_away
        features['avg_red_away_v'] = a_red_away
        
        # Away team overall
        features['avg_gf_overall_v'] = (a_gf_home + a_gf_away) / 2
        features['avg_ga_overall_v'] = (a_ga_home + a_ga_away) / 2
        features['avg_points_overall_v'] = (a_pts_home + a_pts_away) / 2
        
        # Team strength differences
        features['strength_diff'] = features['avg_points_overall_h'] - features['avg_points_overall_v']
        features['attack_diff'] = features['avg_gf_overall_h'] - features['avg_gf_overall_v']
        features['defense_diff'] = features['avg_ga_overall_v'] - features['avg_ga_overall_h']
        
        # Context features (at the end)
        features['season_progress'] = week / 34
        features['has_fans'] = 1 if has_fans else 0
        features['log_fans'] = np.log1p(features['fans'])
        
        # Create DataFrame with features in exact training order
        feature_order = ['Week', 'tier', 'fans', 'neutral',
                        'home_rolling_points', 'home_rolling_gf', 'home_rolling_ga', 'home_rolling_gd',
                        'visitor_rolling_points', 'visitor_rolling_gf', 'visitor_rolling_ga', 'visitor_rolling_gd',
                        'form_diff', 'gd_diff',
                        'avg_gf_home_h', 'avg_ga_home_h', 'avg_points_home_h', 'avg_red_home_h',
                        'avg_gf_away_h', 'avg_ga_away_h', 'avg_points_away_h', 'avg_red_away_h',
                        'avg_gf_overall_h', 'avg_ga_overall_h', 'avg_points_overall_h',
                        'avg_gf_home_v', 'avg_ga_home_v', 'avg_points_home_v', 'avg_red_home_v',
                        'avg_gf_away_v', 'avg_ga_away_v', 'avg_points_away_v', 'avg_red_away_v',
                        'avg_gf_overall_v', 'avg_ga_overall_v', 'avg_points_overall_v',
                        'strength_diff', 'attack_diff', 'defense_diff',
                        'season_progress', 'has_fans', 'log_fans']
        
        return pd.DataFrame([{k: features[k] for k in feature_order}])
    
    def predict_match(self, home_team, away_team, week=1, has_fans=True, show_details=True):
        """Predict match outcome"""
        # Prepare features
        features_df = self.prepare_match_features(home_team, away_team, week, has_fans)
        
        # Make prediction
        prediction = self.model.predict(features_df)[0]
        probabilities = self.model.predict_proba(features_df)[0]
        
        # Map prediction to outcome
        outcome_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        predicted_outcome = outcome_map[prediction]
        
        # Display results
        print("\n" + "=" * 70)
        print("MATCH PREDICTION")
        print("=" * 70)
        print(f"\nMatch: {home_team} (Home) vs {away_team} (Away)")
        print(f"Week: {week}")
        print(f"Fans: {'Yes' if has_fans else 'No'}")
        
        print(f"\nPredicted Outcome: {predicted_outcome}")
        print(f"\nProbabilities:")
        print(f"  Home Win ({home_team}): {probabilities[2]:.1%}")
        print(f"  Draw: {probabilities[1]:.1%}")
        print(f"  Away Win ({away_team}): {probabilities[0]:.1%}")
        
        if show_details:
            print(f"\nTeam Form (Last 5 Matches):")
            h_pts, h_gf, h_ga, h_gd = self.get_team_recent_form(home_team)
            a_pts, a_gf, a_ga, a_gd = self.get_team_recent_form(away_team)
            
            print(f"  {home_team}: {h_pts:.2f} pts/match, {h_gf:.2f} GF, {h_ga:.2f} GA, {h_gd:+.2f} GD")
            print(f"  {away_team}: {a_pts:.2f} pts/match, {a_gf:.2f} GF, {a_ga:.2f} GA, {a_gd:+.2f} GD")
            
            if h_pts > a_pts:
                print(f"  Form Advantage: {home_team}")
            elif a_pts > h_pts:
                print(f"  Form Advantage: {away_team}")
            else:
                print(f"  Form: Equal")
        
        print("\n" + "=" * 70)
        
        return predicted_outcome, probabilities
    
    def list_teams(self):
        """List all available teams"""
        print("\nAvailable Teams:")
        print("=" * 70)
        for i, team in enumerate(self.teams, 1):
            print(f"{i:2d}. {team}")
        print("=" * 70)
        print(f"Total: {len(self.teams)} teams")

def main():
    """Main function for interactive prediction"""
    print("=" * 70)
    print("TURKISH SUPER LEAGUE MATCH PREDICTION TOOL")
    print("=" * 70)
    
    # Initialize predictor
    predictor = MatchPredictor()
    
    # Example predictions
    print("\n\nEXAMPLE PREDICTIONS:")
    print("=" * 70)
    
    # Example 1: Galatasaray vs Fenerbahce
    predictor.predict_match('Galatasaray', 'Fenerbahce', week=10, has_fans=True)
    
    # Example 2: Besiktas vs Trabzonspor
    predictor.predict_match('Besiktas', 'Trabzonspor', week=15, has_fans=True)
    
    # List available teams
    print("\n\n")
    predictor.list_teams()
    
    # Interactive mode
    print("\n\nINTERACTIVE MODE")
    print("=" * 70)
    print("Enter team names to predict a match (or 'quit' to exit)")
    
    while True:
        try:
            home = input("\nHome team: ").strip()
            if home.lower() == 'quit':
                break
            
            away = input("Away team: ").strip()
            if away.lower() == 'quit':
                break
            
            week = input("Week (default 1): ").strip()
            week = int(week) if week else 1
            
            fans = input("Has fans? (y/n, default y): ").strip().lower()
            has_fans = fans != 'n'
            
            predictor.predict_match(home, away, week, has_fans)
            
        except ValueError as e:
            print(f"\nError: {e}")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
    
    print("\nThank you for using Turkish Super League Match Predictor!")

if __name__ == "__main__":
    main()

