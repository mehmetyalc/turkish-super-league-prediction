"""
Turkish Super League ML Dataset Preparation (Optimized)
Fast feature engineering using vectorized operations
"""

import pandas as pd
import numpy as np

class TSLDatasetPreparatorOptimized:
    """Optimized Turkish Super League data preparation"""
    
    def __init__(self, data_path):
        print("Loading Turkish Super League data...")
        self.df = pd.read_csv(data_path)
        print(f"  Loaded {len(self.df)} matches from {self.df['Season'].min()} to {self.df['Season'].max()}")
        
    def prepare_ml_dataset(self):
        """Prepare complete ML dataset with optimized features"""
        print("\nPreparing ML dataset...")
        
        # Clean and prepare base data
        df = self.clean_data()
        print(f"[1/5] Data cleaned: {len(df)} matches")
        
        # Add rolling statistics (optimized)
        df = self.add_rolling_features(df)
        print(f"[2/5] Added rolling features")
        
        # Add team aggregated features
        df = self.add_team_features(df)
        print(f"[3/5] Added team features")
        
        # Add match context features
        df = self.add_context_features(df)
        print(f"[4/5] Added context features")
        
        # Clean for ML
        df = self.clean_for_ml(df)
        print(f"[5/5] Final dataset ready: {df.shape}")
        
        return df
    
    def clean_data(self):
        """Clean and prepare base data"""
        df = self.df.copy()
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Encode result as target
        result_map = {'H': 2, 'D': 1, 'A': 0}
        df['result_encoded'] = df['result'].map(result_map)
        
        # Encode result_half as well
        df['result_half'] = df['result_half'].map(result_map).fillna(1)  # Default to draw if missing
        
        # Fill missing values
        df['fans'] = df['fans'].fillna(0)
        df['home_red_card'] = df['home_red_card'].fillna(0)
        df['visitor_red_card'] = df['visitor_red_card'].fillna(0)
        
        # Add points for each team
        df['home_points'] = df['result'].map({'H': 3, 'D': 1, 'A': 0})
        df['visitor_points'] = df['result'].map({'H': 0, 'D': 1, 'A': 3})
        
        return df
    
    def add_rolling_features(self, df):
        """Add rolling statistics (last 5 matches) - optimized"""
        print("    Calculating rolling features...")
        
        window = 5
        
        # Create separate dataframes for home and away matches
        home_matches = df[['Date', 'home', 'hgoal', 'vgoal', 'home_points']].copy()
        home_matches.columns = ['Date', 'team', 'goals_for', 'goals_against', 'points']
        
        away_matches = df[['Date', 'visitor', 'vgoal', 'hgoal', 'visitor_points']].copy()
        away_matches.columns = ['Date', 'team', 'goals_for', 'goals_against', 'points']
        
        # Combine all matches
        all_matches = pd.concat([home_matches, away_matches]).sort_values('Date')
        
        # Calculate rolling statistics per team
        all_matches['rolling_points'] = all_matches.groupby('team')['points'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )
        all_matches['rolling_gf'] = all_matches.groupby('team')['goals_for'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )
        all_matches['rolling_ga'] = all_matches.groupby('team')['goals_against'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )
        all_matches['rolling_gd'] = all_matches['rolling_gf'] - all_matches['rolling_ga']
        
        # Merge back to main dataframe
        home_rolling = all_matches[all_matches['team'].isin(df['home'])].copy()
        home_rolling = home_rolling.groupby(['Date', 'team']).last().reset_index()
        home_rolling = home_rolling[['Date', 'team', 'rolling_points', 'rolling_gf', 'rolling_ga', 'rolling_gd']]
        home_rolling.columns = ['Date', 'home', 'home_rolling_points', 'home_rolling_gf', 
                               'home_rolling_ga', 'home_rolling_gd']
        
        away_rolling = all_matches[all_matches['team'].isin(df['visitor'])].copy()
        away_rolling = away_rolling.groupby(['Date', 'team']).last().reset_index()
        away_rolling = away_rolling[['Date', 'team', 'rolling_points', 'rolling_gf', 'rolling_ga', 'rolling_gd']]
        away_rolling.columns = ['Date', 'visitor', 'visitor_rolling_points', 'visitor_rolling_gf',
                               'visitor_rolling_ga', 'visitor_rolling_gd']
        
        df = df.merge(home_rolling, on=['Date', 'home'], how='left')
        df = df.merge(away_rolling, on=['Date', 'visitor'], how='left')
        
        # Form difference
        df['form_diff'] = df['home_rolling_points'] - df['visitor_rolling_points']
        df['gd_diff'] = df['home_rolling_gd'] - df['visitor_rolling_gd']
        
        return df
    
    def add_team_features(self, df):
        """Add aggregated team features"""
        print("    Calculating team features...")
        
        # Overall team statistics (cumulative)
        home_stats = df.groupby('home').agg({
            'hgoal': 'mean',
            'vgoal': 'mean',
            'home_points': 'mean',
            'home_red_card': 'mean'
        }).reset_index()
        home_stats.columns = ['team', 'avg_gf_home', 'avg_ga_home', 'avg_points_home', 'avg_red_home']
        
        away_stats = df.groupby('visitor').agg({
            'vgoal': 'mean',
            'hgoal': 'mean',
            'visitor_points': 'mean',
            'visitor_red_card': 'mean'
        }).reset_index()
        away_stats.columns = ['team', 'avg_gf_away', 'avg_ga_away', 'avg_points_away', 'avg_red_away']
        
        # Merge team stats
        team_stats = home_stats.merge(away_stats, on='team', how='outer').fillna(0)
        
        # Overall stats
        team_stats['avg_gf_overall'] = (team_stats['avg_gf_home'] + team_stats['avg_gf_away']) / 2
        team_stats['avg_ga_overall'] = (team_stats['avg_ga_home'] + team_stats['avg_ga_away']) / 2
        team_stats['avg_points_overall'] = (team_stats['avg_points_home'] + team_stats['avg_points_away']) / 2
        
        # Add to main dataframe
        df = df.merge(team_stats.rename(columns={'team': 'home'}), on='home', how='left')
        df = df.merge(team_stats.rename(columns={'team': 'visitor'}), on='visitor', how='left', suffixes=('_h', '_v'))
        
        # Team strength difference
        df['strength_diff'] = df['avg_points_overall_h'] - df['avg_points_overall_v']
        df['attack_diff'] = df['avg_gf_overall_h'] - df['avg_gf_overall_v']
        df['defense_diff'] = df['avg_ga_overall_v'] - df['avg_ga_overall_h']  # Lower GA is better
        
        return df
    
    def add_context_features(self, df):
        """Add match context features"""
        # Season progress
        df['season_progress'] = df.groupby('Season')['Week'].transform(lambda x: x / x.max())
        
        # Home advantage (fans)
        df['has_fans'] = (df['fans'] > 0).astype(int)
        df['log_fans'] = np.log1p(df['fans'])
        
        # Goal-based features
        df['is_high_scoring'] = (df['totgoal'] > 3).astype(int)
        df['first_half_leader'] = np.sign(df['hgoal_half'] - df['vgoal_half'])
        
        return df
    
    def clean_for_ml(self, df):
        """Clean dataset for ML"""
        print("\nCleaning for ML...")
        
        # Drop non-numeric, identifier columns, and data leakage features
        # Data leakage: features that contain match result information
        columns_to_drop = ['Date', 'home', 'visitor', 'FT', 'HT', 'result', 
                          'Season', 'division', 'home_points', 'visitor_points',
                          'hgoal', 'vgoal', 'totgoal', 'goaldiff',  # Match result features
                          'hgoal_half', 'vgoal_half', 'half_totgoal', 'half_goaldiff', 'result_half',  # Half-time result
                          'home_red_card', 'visitor_red_card',  # In-match events
                          'is_high_scoring', 'first_half_leader']  # Derived from match results
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        
        # Fill NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Drop rows with missing target
        df = df.dropna(subset=['result_encoded'])
        
        # Remove first 500 matches (insufficient history)
        df = df.iloc[500:].reset_index(drop=True)
        
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
    print("TURKISH SUPER LEAGUE ML DATASET PREPARATION (OPTIMIZED)")
    print("=" * 70)
    
    preparator = TSLDatasetPreparatorOptimized("data/raw/tsl_dataset.csv")
    ml_df = preparator.prepare_ml_dataset()
    preparator.save_dataset(ml_df)
    
    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETE!")
    print("=" * 70)

