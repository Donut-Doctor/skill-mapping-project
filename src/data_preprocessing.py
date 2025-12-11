"""
Data Preprocessing Module
Handles data loading, cleaning, encoding, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import os

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, 'skill_mapping_audit.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataPreprocessor')


class DataPreprocessor:

    def __init__(self, filepath):

        self.filepath = filepath
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        logger.info("DataPreprocessor initialized")
        
    def load_data(self):
        """Load dataset from CSV file"""
        try:
            self.df = pd.read_csv(self.filepath)
            logger.info(f"Loaded {len(self.df)} records from {self.filepath}")
            print(f"Loaded {len(self.df)} student records")
            print(f"   Features: {self.df.shape[1]}")
            return self
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def check_data_quality(self):
        """Check for missing values and duplicates"""
        missing = self.df.isnull().sum().sum()
        duplicates = self.df.duplicated().sum()
        
        logger.info(f"Data quality check - Missing: {missing}, Duplicates: {duplicates}")
        print(f"   Missing values: {missing}")
        print(f"   Duplicate records: {duplicates}")
        
        return self
    
    def encode_categorical(self):

        categorical_cols = ['NCrF_Level', 'Primary_Skill_Gap', 
                           'Predicted_Job_Role', 'Job_Readiness_Level']
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_Encoded'] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
                logger.info(f"Encoded column: {col}")

        print(f"Encoded {len(categorical_cols)} categorical features")
        return self
    
    def create_features(self):

        tech_cols = ['Programming_Score', 'Data_Analysis_Score', 
                    'Web_Development_Score', 'Database_Management_Score',
                    'Cloud_Computing_Score', 'Machine_Learning_Score', 
                    'Cybersecurity_Score']
        
        self.df['Tech_Skills_Avg'] = self.df[tech_cols].mean(axis=1)
        
        soft_cols = ['Written_Communication_Score', 'Verbal_Communication_Score',
                    'Presentation_Skills_Score', 'Problem_Solving_Score',
                    'Analytical_Thinking_Score', 'Teamwork_Score', 
                    'Leadership_Score', 'Digital_Literacy_Score']
        
        self.df['Soft_Skills_Avg'] = self.df[soft_cols].mean(axis=1)
        
        self.df['Skills_Balance'] = abs(self.df['Tech_Skills_Avg'] - 
                                        self.df['Soft_Skills_Avg'])
        
        print(f"Created 3 engineered features")
        logger.info("Feature engineering completed")
        
        return self
    
    def get_feature_columns(self):
        
        feature_cols = [col for col in self.df.columns 
                       if col.endswith('_Score') or 
                          col.endswith('_Avg') or 
                          col == 'Skills_Balance' or 
                          col == 'Skill_Gap_Index']
        return feature_cols
    
    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):

        feature_cols = self.get_feature_columns()
        
        X = self.df[feature_cols]
        y = self.df['Job_Readiness_Level_Encoded']
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"\nData Split:")
        print(f"   Training:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   Testing:    {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
    
    def normalize_features(self, X_train, X_val, X_test):

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("Features normalized using StandardScaler")
        print("Features normalized")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
