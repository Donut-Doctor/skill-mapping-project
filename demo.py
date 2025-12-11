"""
Demo Script
Demonstrates how to use the trained model for predictions
"""

import joblib
import pandas as pd
import json
import sys

sys.path.append('src')

def demo():
    """Run demonstration of trained model"""
    
    print("\n" + "="*70)
    print("SKILL MAPPING SYSTEM DEMO")
    print("="*70)
    

    print("\nLoading trained model...")
    model = joblib.load('models/skill_mapping_model.pkl')

    with open('models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print("\nModel Information:")
    print(f"   Model Type: {metadata['model_type']}")
    print(f"   NEP 2020 Compliant: {metadata['nep_2020_compliant']}")
    print(f"   Bias Checked: {metadata['bias_checked']}")
    print(f"   Explainable: {metadata['explainable']}")
    print(f"   Number of Features: {metadata['n_features']}")
    
    print("\nTop 5 Important Features:")
    importance_df = pd.read_csv('outputs/feature_importance.csv')
    print(importance_df.head().to_string(index=False))

    print("\nAudit Trail Sample:")
    with open('logs/skill_mapping_audit.log', 'r') as f:
        lines = f.readlines()
        for line in lines[:5]:
            print("   " + line.strip())
    
    print("\n" + "="*70)
    print("Demo complete! Model is ready for predictions.")
    print("="*70)


if __name__ == "__main__":
    demo()
