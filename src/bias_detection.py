"""
Bias Detection Module
Detects and reports bias across different demographic groups
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('BiasDetector')


class BiasDetector:
    def __init__(self, df):
        self.df = df.copy()
        logger.info("BiasDetector initialized")
    
    def check_demographic_parity(self, predictions, group_col='NCrF_Level'):

        print("\n" + "="*70)
        print("BIAS DETECTION & FAIRNESS ANALYSIS")
        print("="*70)
        
        self.df['Predicted_Level'] = predictions
        
        print(f"\nFairness Analysis by {group_col}:")
        group_analysis = self.df.groupby(group_col)['Predicted_Level'].value_counts(
            normalize=True
        ).unstack(fill_value=0)
        print(group_analysis.round(3))
        
        max_diff = group_analysis.max().max() - group_analysis.min().min()
        fairness_score = 1 - max_diff
        
        print(f"\nFairness Score: {fairness_score:.3f} (1.0 = perfectly fair)")
        
        if fairness_score > 0.8:
            print("PASS: Model shows acceptable fairness")
            logger.info(f"Fairness check PASSED - Score: {fairness_score:.3f}")
        else:
            print("WARNING: Potential bias detected")
            logger.warning(f"Fairness check WARNING - Score: {fairness_score:.3f}")
        
        return fairness_score
    
    def generate_bias_report(self, predictions, output_path='outputs/bias_report.txt'):

        import os
        os.makedirs('outputs', exist_ok=True)
        
        fairness_score = self.check_demographic_parity(predictions)
        
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BIAS DETECTION REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Fairness Score: {fairness_score:.3f}\n")
            f.write(f"Status: {'PASS' if fairness_score > 0.8 else 'WARNING'}\n")
            f.write("\nDetailed Analysis:\n")
            f.write(str(self.df.groupby('NCrF_Level')['Predicted_Level'].value_counts(normalize=True)))
        
        print(f"Bias report saved to {output_path}")
        logger.info(f"Bias report generated: {output_path}")
