"""
Explainability Module
Provides interpretability and transparency for model predictions
"""

import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger('Explainability')


class ModelExplainer:
    
    def __init__(self, model, feature_names):

        self.model = model
        self.feature_names = feature_names
        logger.info("ModelExplainer initialized")
    
    def get_feature_importance(self):

        print("\n" + "="*70)
        print("EXPLAINABILITY: FEATURE IMPORTANCE ANALYSIS")
        print("="*70)

        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        logger.info("Feature importance calculated")
        
        return feature_importance
    
    def save_importance_report(self, feature_importance, output_path='outputs/feature_importance.csv'):

        os.makedirs('outputs', exist_ok=True)
        feature_importance.to_csv(output_path, index=False)
        print(f"Feature importance saved to {output_path}")
        logger.info(f"Feature importance report saved: {output_path}")
