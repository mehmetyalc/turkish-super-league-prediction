"""
Turkish Super League ML Dataset Preparation
Feature engineering for match result prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime

class TSLDatasetPreparator:
    """Prepares Turkish Super League data for ML"""
    
    def __init__(self, data_path):
        print("Loading Turkish Super League data...")
        self.df = pd.read_csv(data_path)
        print(f"  Loaded {len(self.df)} matches from {self.df['Season'].min()} to {self.df['Season'].max()}")
        
    def prepare_ml_dataset(self):
        """Prepare complete ML dataset with all features"""
        print("\nPreparing ML dataset...")
        
        # Clean and prepare base data
        df = self.clean_data()
        print(f"[1/7] Data cleaned: {len(df)} matches")
        
        # Add team historical features
        df = self.add_team_historical_features(df)
        print(f"[2/7] Added team historical features")
        
        # Add recent form features
        df = self.add_form_features(df)
        print(f"[3/7] Added form features")
        
        # Add head-to-head features
        df = self.add_h2h_features(df)
        print(f"[4/7] Added head-to-head features")
        
        # Add home/away specific features
        df = self.add_home_away_features(df)
        print(f"[5/7] Added home/away features")
        
        # Add goal scoring features
        df = self.add_goal_features(df)
        print(f"[6/7] Added goal scoring features")
        
        # Clean for ML
        df = self.clean_for_ml(df)
        print(f"[7/7] Final dataset ready: {df.shape}")
        
        return df
    
    def clean_data(self):
        """Clean and prepare base data"""
        df = self.df.copy()
        
        # Parse dates properly (handles both YYYY-MM-DD and DD/MM/YYYY)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Drop rows with missing dates
        df = df.dropna(subset=['Date'])
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Encode result as target variable
        # H = Home Win (2), D = Draw (1), A = Away Win (0)
        result_map = {'H': 2, 'D': 1, 'A': 0}
        df['result_encoded'] = df['result'].map(result_map)
        
        # Fill missing values
        df['fans'] = df['fans'].fillna(0)
        df['home_red_card'] = df['home_red_card'].fillna(0)
        df['visitor_red_card'] = df['visitor_red_card'].fillna(0)
        
        return df
    
    def add_team_historical_features(self, df):
        """Add historical performance features for each team"""
        # Calculate overall team statistics
        home_stats = df.groupby('home').agg({
            'hgoal': 'mean',
            'vgoal': 'mean',
            'result': lambda x: (x == 'H').sum() / len(x)  # Win rate
        }).reset_index()
        home_stats.columns = ['team', 'avg_goals_scored', 'avg_goals_conceded', 'win_rate']
        
        away_stats = df.groupby('visitor').agg({
            'vgoal': 'mean',
            'hgoal': 'mean',
            'result': lambda x: (x == 'A').sum() / len(x)
        }).reset_index()
        away_stats.columns = ['team', 'avg_goals_scored_away', 'avg_goals_conceded_away', 'win_rate_away']
        
        # Merge team stats
        team_stats = home_stats.merge(away_stats, on='team', how='outer').fillna(0)
        
        # Add to main dataframe
        df = df.merge(team_stats.rename(columns={'team': 'home'}), on='home', how='left', suffixes=('', '_home'))
        df = df.merge(team_stats.rename(columns={'team': 'visitor'}), on='visitor', how='left', suffixes=('', '_visitor'))
        
        return df
    
    def add_form_features(self, df):
        """Add recent form features (last 5 matches)"""
        print("    Calculating recent form...")
        
        form_window = 5
        
        # Initialize form columns
        df['home_form_points'] = 0.0
        df['home_form_goals_scored'] = 0.0
        df['home_form_goals_conceded'] = 0.0
        df['visitor_form_points'] = 0.0
        df['visitor_form_goals_scored'] = 0.0
        df['visitor_form_goals_conceded'] = 0.0
        
        # Calculate form for each match
        for idx in range(len(df)):
            current_date = df.loc[idx, 'Date']
            home_team = df.loc[idx, 'home']
            visitor_team = df.loc[idx, 'visitor']
            
            # Get last N matches before current date for home team
            home_recent = df[(df['Date'] < current_date) & 
                            ((df['home'] == home_team) | (df['visitor'] == home_team))].tail(form_window)
            
            if len(home_recent) > 0:
                home_points = 0
                home_gf = 0
                home_ga = 0
                
                for _, match in home_recent.iterrows():
                    if match['home'] == home_team:
                        home_gf += match['hgoal']
                        home_ga += match['vgoal']
                        if match['result'] == 'H':
                            home_points += 3
                        elif match['result'] == 'D':
                            home_points += 1
                    else:
                        home_gf += match['vgoal']
                        home_ga += match['hgoal']
                        if match['result'] == 'A':
                            home_points += 3
                        elif match['result'] == 'D':
                            home_points += 1
                
                df.loc[idx, 'home_form_points'] = home_points / len(home_recent)
                df.loc[idx, 'home_form_goals_scored'] = home_gf / len(home_recent)
                df.loc[idx, 'home_form_goals_conceded'] = home_ga / len(home_recent)
            
            # Get last N matches for visitor team
            visitor_recent = df[(df['Date'] < current_date) & 
                               ((df['home'] == visitor_team) | (df['visitor'] == visitor_team))].tail(form_window)
            
            if len(visitor_recent) > 0:
                visitor_points = 0
                visitor_gf = 0
                visitor_ga = 0
                
                for _, match in visitor_recent.iterrows():
                    if match['home'] == visitor_team:
                        visitor_gf += match['hgoal']
                        visitor_ga += match['vgoal']
                        if match['result'] == 'H':
                            visitor_points += 3
                        elif match['result'] == 'D':
                            visitor_points += 1
                    else:
                        visitor_gf += match['vgoal']
                        visitor_ga += match['hgoal']
                        if match['result'] == 'A':
                            visitor_points += 3
                        elif match['result'] == 'D':
                            visitor_points += 1
                
                df.loc[idx, 'visitor_form_points'] = visitor_points / len(visitor_recent)
                df.loc[idx, 'visitor_form_goals_scored'] = visitor_gf / len(visitor_recent)
                df.loc[idx, 'visitor_form_goals_conceded'] = visitor_ga / len(visitor_recent)
        
        # Form difference
        df['form_points_diff'] = df['home_form_points'] - df['visitor_form_points']
        
        return df
    
    def add_h2h_features(self, df):
        """Add head-to-head features"""
        print("    Calculating head-to-head stats...")
        
        df['h2h_home_wins'] = 0
        df['h2h_draws'] = 0
        df['h2h_away_wins'] = 0
        df['h2h_matches'] = 0
        
        for idx in range(len(df)):
            current_date = df.loc[idx, 'Date']
            home_team = df.loc[idx, 'home']
            visitor_team = df.loc[idx, 'visitor']
            
            # Get all previous matches between these teams
            h2h = df[(df['Date'] < current_date) & 
                    (((df['home'] == home_team) & (df['visitor'] == visitor_team)) |
                     ((df['home'] == visitor_team) & (df['visitor'] == home_team)))]
            
            if len(h2h) > 0:
                df.loc[idx, 'h2h_matches'] = len(h2h)
                
                # Count results from home team's perspective
                for _, match in h2h.iterrows():
                    if match['home'] == home_team:
                        if match['result'] == 'H':
                            df.loc[idx, 'h2h_home_wins'] += 1
                        elif match['result'] == 'D':
                            df.loc[idx, 'h2h_draws'] += 1
                        else:
                            df.loc[idx, 'h2h_away_wins'] += 1
                    else:
                        if match['result'] == 'A':
                            df.loc[idx, 'h2h_home_wins'] += 1
                        elif match['result'] == 'D':
                            df.loc[idx, 'h2h_draws'] += 1
                        else:
                            df.loc[idx, 'h2h_away_wins'] += 1
        
        # H2H win rate
        df['h2h_home_win_rate'] = df['h2h_home_wins'] / (df['h2h_matches'] + 1)
        
        return df
    
    def add_home_away_features(self, df):
        """Add home/away specific performance features"""
        # Home team home performance
        home_home_stats = df.groupby('home').agg({
            'hgoal': 'mean',
            'vgoal': 'mean',
            'result': lambda x: (x == 'H').sum() / len(x)
        }).reset_index()
        home_home_stats.columns = ['team', 'home_goals_at_home', 'home_conceded_at_home', 'home_win_rate_at_home']
        
        # Visitor team away performance
        visitor_away_stats = df.groupby('visitor').agg({
            'vgoal': 'mean',
            'hgoal': 'mean',
            'result': lambda x: (x == 'A').sum() / len(x)
        }).reset_index()
        visitor_away_stats.columns = ['team', 'visitor_goals_away', 'visitor_conceded_away', 'visitor_win_rate_away']
        
        # Merge
        df = df.merge(home_home_stats.rename(columns={'team': 'home'}), on='home', how='left')
        df = df.merge(visitor_away_stats.rename(columns={'team': 'visitor'}), on='visitor', how='left')
        
        return df
    
    def add_goal_features(self, df):
        """Add goal-related features"""
        # Goal difference in recent matches
        df['home_recent_goal_diff'] = df['home_form_goals_scored'] - df['home_form_goals_conceded']
        df['visitor_recent_goal_diff'] = df['visitor_form_goals_scored'] - df['visitor_form_goals_conceded']
        
        # Red card history
        home_red_cards = df.groupby('home')['home_red_card'].mean().reset_index()
        home_red_cards.columns = ['team', 'avg_red_cards']
        
        visitor_red_cards = df.groupby('visitor')['visitor_red_card'].mean().reset_index()
        visitor_red_cards.columns = ['team', 'avg_red_cards_away']
        
        df = df.merge(home_red_cards.rename(columns={'team': 'home'}), on='home', how='left')
        df = df.merge(visitor_red_cards.rename(columns={'team': 'visitor'}), on='visitor', how='left')
        
        return df
    
    def clean_for_ml(self, df):
        """Clean dataset for ML"""
        print("\nCleaning for ML...")
        
        # Drop non-numeric and identifier columns
        columns_to_drop = ['Date', 'home', 'visitor', 'FT', 'HT', 'result', 'Season', 'division']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        
        # Fill remaining NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Drop rows with missing target
        df = df.dropna(subset=['result_encoded'])
        
        # Remove early matches with insufficient history (first 100 matches)
        df = df.iloc[100:].reset_index(drop=True)
        
        print(f"  Final dataset: {df.shape}")
        print(f"  Features: {df.shape[1] - 1}")
        print(f"  Samples: {len(df)}")
        print(f"\n  Target distribution:")
        print(f"    Home Win: {(df['result_encoded'] == 2).sum()} ({(df['result_encoded'] == 2).sum()/len(df)*100:.1f}%)")
        print(f"    Draw: {(df['result_encoded'] == 1).sum()} ({(df['result_encoded'] == 1).sum()/len(df)*100:.1f}%)")
        print(f"    Away Win: {(df['result_encoded'] == 0).sum()} ({(df['result_encoded'] == 0).sum()/len(df)*100:.1f}%)")
        
        return df
    
    def save_dataset(self, df):
        """Save processed dataset"""
        output_path = "data/processed/tsl_ml_dataset.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")
        
        # Print feature list
        features = [col for col in df.columns if col != 'result_encoded']
        print(f"\nFeatures ({len(features)}):")
        for i, feat in enumerate(features, 1):
            print(f"  {i}. {feat}")
        
        return output_path

if __name__ == "__main__":
    print("=" * 70)
    print("TURKISH SUPER LEAGUE ML DATASET PREPARATION")
    print("=" * 70)
    
    preparator = TSLDatasetPreparator("data/raw/tsl_dataset.csv")
    ml_df = preparator.prepare_ml_dataset()
    preparator.save_dataset(ml_df)
    
    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETE!")
    print("=" * 70)

