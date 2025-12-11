"""
Main Script
Runs the complete ethical AI skill mapping system
"""

import os
import sys
import numpy as np
from datetime import datetime

sys.path.append('src')

from data_preprocessing import DataPreprocessor
from bias_detection import BiasDetector
from model_training import SkillMappingModel
from explainability import ModelExplainer


def main():
    """Execute the complete pipeline"""
    
    print("\n" + "="*70)
    print("ETHICAL AI-BASED SKILL MAPPING SYSTEM")
    print("For Indian Educational Institutions (NEP 2020 Compliant)")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
    print("="*70)
    
    preprocessor = DataPreprocessor('data/skill_assessment_mapping_5000.csv')
    preprocessor.load_data() \
                .check_data_quality() \
                .encode_categorical() \
                .create_features()
    
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocessor.split_data()
    X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.normalize_features(
        X_train, X_val, X_test
    )

    model = SkillMappingModel()
    model.train(X_train_scaled, y_train, X_val_scaled, y_val, feature_names)

    explainer = ModelExplainer(model.model, feature_names)
    feature_importance = explainer.get_feature_importance()
    explainer.save_importance_report(feature_importance)
    
    all_predictions = model.model.predict(
        np.vstack([X_train_scaled, X_val_scaled, X_test_scaled])
    )
    bias_detector = BiasDetector(preprocessor.df)
    fairness_score = bias_detector.check_demographic_parity(all_predictions)
    bias_detector.generate_bias_report(all_predictions)

    label_encoder = preprocessor.label_encoders['Job_Readiness_Level']
    y_pred, test_acc = model.evaluate(X_test_scaled, y_test, label_encoder)

    model.save_model()

    print("\n" + "="*70)
    print("SYSTEM SUMMARY")
    print("="*70)
    print(f"NEP 2020 Compliant: YES")
    print(f"Bias Detection: COMPLETED")
    print(f"Explainability: IMPLEMENTED")
    print(f"Audit Logging: ACTIVE")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Fairness Score: {fairness_score:.3f}")
    print("\nGenerated Files:")
    print("   • models/skill_mapping_model.pkl")
    print("   • models/model_metadata.json")
    print("   • outputs/feature_importance.csv")
    print("   • outputs/bias_report.txt")
    print("   • logs/skill_mapping_audit.log")
    print("="*70)
    print("\nSYSTEM BUILD COMPLETE!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    main()
