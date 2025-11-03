"""
Turkish Super League Match Result Prediction Models
Train and evaluate classification models
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class TSLModelTrainer:
    """Train Turkish Super League prediction models"""
    
    def __init__(self, data_path):
        print("Loading ML dataset...")
        self.df = pd.read_csv(data_path)
        print(f"  Loaded {len(self.df)} matches with {self.df.shape[1]-1} features")
        
        # Prepare X and y
        self.X = self.df.drop('result_encoded', axis=1)
        self.y = self.df['result_encoded']
        
        # Train-test split (80-20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"\n  Train set: {len(self.X_train)} matches")
        print(f"  Test set: {len(self.X_test)} matches")
        
        self.models = {}
        self.results = {}
    
    def train_random_forest(self):
        """Train Random Forest Classifier"""
        print("\n[1/3] Training Random Forest...")
        
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred = rf.predict(self.X_test)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        # Cross-validation
        cv_scores = cross_val_score(rf, self.X_train, self.y_train, cv=5, scoring='accuracy')
        
        self.models['Random Forest'] = rf
        self.results['Random Forest'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return rf
    
    def train_xgboost(self):
        """Train XGBoost Classifier"""
        print("\n[2/3] Training XGBoost...")
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        xgb_model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred = xgb_model.predict(self.X_test)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        # Cross-validation
        cv_scores = cross_val_score(xgb_model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        
        self.models['XGBoost'] = xgb_model
        self.results['XGBoost'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return xgb_model
    
    def train_lightgbm(self):
        """Train LightGBM Classifier"""
        print("\n[3/3] Training LightGBM...")
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        lgb_model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred = lgb_model.predict(self.X_test)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        # Cross-validation
        cv_scores = cross_val_score(lgb_model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        
        self.models['LightGBM'] = lgb_model
        self.results['LightGBM'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return lgb_model
    
    def train_all(self):
        """Train all models"""
        print("\n" + "=" * 70)
        print("TRAINING TURKISH SUPER LEAGUE PREDICTION MODELS")
        print("=" * 70)
        
        self.train_random_forest()
        self.train_xgboost()
        self.train_lightgbm()
        
        return self.models, self.results
    
    def save_models(self):
        """Save trained models"""
        print("\nSaving models...")
        
        for name, model in self.models.items():
            filename = f"results/models/{name.lower().replace(' ', '_')}_tsl.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"  Saved: {filename}")
    
    def generate_report(self):
        """Generate training report"""
        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 70)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [r['accuracy'] for r in self.results.values()],
            'F1-Score': [r['f1_score'] for r in self.results.values()],
            'CV Accuracy': [r['cv_mean'] for r in self.results.values()],
            'CV Std': [r['cv_std'] for r in self.results.values()]
        })
        
        results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
        
        print("\n" + results_df.to_string(index=False))
        
        # Best model
        best_model = results_df.iloc[0]['Model']
        best_accuracy = results_df.iloc[0]['Accuracy']
        
        print(f"\nBest Model: {best_model}")
        print(f"Test Accuracy: {best_accuracy:.4f}")
        
        # Detailed classification report for best model
        print(f"\nDetailed Classification Report ({best_model}):")
        print("=" * 70)
        y_pred = self.results[best_model]['y_pred']
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['Away Win', 'Draw', 'Home Win']))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        # Save report
        report_path = "results/reports/training_report.txt"
        with open(report_path, 'w') as f:
            f.write("TURKISH SUPER LEAGUE MATCH PREDICTION - TRAINING REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Dataset: {len(self.df)} matches\n")
            f.write(f"Features: {self.X.shape[1]}\n")
            f.write(f"Train/Test Split: {len(self.X_train)}/{len(self.X_test)}\n\n")
            f.write("MODEL PERFORMANCE\n")
            f.write("=" * 70 + "\n")
            f.write(results_df.to_string(index=False))
            f.write(f"\n\nBest Model: {best_model}\n")
            f.write(f"Test Accuracy: {best_accuracy:.4f}\n\n")
            f.write(f"Classification Report ({best_model}):\n")
            f.write("=" * 70 + "\n")
            f.write(classification_report(self.y_test, y_pred,
                                         target_names=['Away Win', 'Draw', 'Home Win']))
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm))
        
        print(f"\nReport saved: {report_path}")
        
        return results_df

if __name__ == "__main__":
    trainer = TSLModelTrainer("data/processed/tsl_ml_dataset.csv")
    models, results = trainer.train_all()
    trainer.save_models()
    results_df = trainer.generate_report()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)

