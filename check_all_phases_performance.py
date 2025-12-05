"""
בודק את כל השלבים ואת אחוזי הביצועים
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "src" / "phase1"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "phase2"))

print("="*80)
print("CHECKING ALL PHASES - PERFORMANCE METRICS")
print("="*80)

# Phase 1: Product and User Categorization
print("\n" + "="*80)
print("PHASE 1: Product and User Categorization")
print("="*80)

from ml_implementation import MLImplementation  # type: ignore

ml = MLImplementation(r"C:\Users\Reuven\Desktop\ML")
phase1_results = ml.run_phase1()

print("\n" + "="*80)
print("PHASE 1 RESULTS SUMMARY:")
print("="*80)
print(f"Product Categorization:")
print(f"  - Algorithm: Logistic Regression")
print(f"  - Number of categories: {len(set(phase1_results['product_clusters']))}")
print(f"  - Accuracy: {phase1_results['product_accuracy']:.3f} ({phase1_results['product_accuracy']*100:.1f}%)")
print(f"\nUser Categorization:")
print(f"  - Algorithm: {ml.user_clustering_algorithm}")
print(f"  - Number of clusters: {len(set(phase1_results['user_clusters']))}")
print(f"  - Silhouette Score: {phase1_results['user_silhouette']:.3f} ({phase1_results['user_silhouette']*100:.1f}%)")

# Phase 2: Recommendation System
print("\n" + "="*80)
print("PHASE 2: Hybrid Recommendation System")
print("="*80)

from recommendation_system_ml import RecommendationSystem  # type: ignore

rec_system = RecommendationSystem(r"C:\Users\Reuven\Desktop\ML")
phase2_results = rec_system.run_phase2()

print("\n" + "="*80)
print("PHASE 2 RESULTS SUMMARY:")
print("="*80)

# Load evaluation results
import pandas as pd
eval_path = Path(r"C:\Users\Reuven\Desktop\ML") / "datasets" / "ml_results" / "recommendation_evaluation.csv"
if eval_path.exists():
    eval_df = pd.read_csv(eval_path)
    
    if len(eval_df) > 0:
        print(f"\nRecommendation Evaluation (tested on {len(eval_df)} users):")
        
        # Calculate average metrics
        if 'precision@3' in eval_df.columns:
            avg_precision = eval_df['precision@3'].mean()
            print(f"  - Precision@3: {avg_precision:.3f} ({avg_precision*100:.1f}%)")
        
        # Category match
        if 'precision@3' in eval_df.columns:
            category_match = eval_df['precision@3'].mean()
            print(f"  - Category Match: {category_match:.3f} ({category_match*100:.1f}%)")
    else:
        print("  No evaluation results found")
else:
    print("  Evaluation file not found")

# Final Summary
print("\n" + "="*80)
print("FINAL PERFORMANCE SUMMARY")
print("="*80)
print(f"\n1. Product Categorization:")
print(f"   Accuracy: {phase1_results['product_accuracy']:.3f} ({phase1_results['product_accuracy']*100:.1f}%)")
print(f"   Status: {'EXCELLENT' if phase1_results['product_accuracy'] > 0.8 else 'GOOD' if phase1_results['product_accuracy'] > 0.6 else 'NEEDS IMPROVEMENT'}")

print(f"\n2. User Categorization:")
print(f"   Silhouette Score: {phase1_results['user_silhouette']:.3f} ({phase1_results['user_silhouette']*100:.1f}%)")
print(f"   Status: {'EXCELLENT' if phase1_results['user_silhouette'] > 0.7 else 'GOOD' if phase1_results['user_silhouette'] > 0.5 else 'NEEDS IMPROVEMENT'}")

if eval_path.exists() and len(eval_df) > 0:
    if 'precision@3' in eval_df.columns:
        avg_precision = eval_df['precision@3'].mean()
        print(f"\n3. Recommendation System (Phase 2):")
        print(f"   Precision@3: {avg_precision:.3f} ({avg_precision*100:.1f}%)")
        print(f"   Status: {'EXCELLENT' if avg_precision > 0.5 else 'GOOD' if avg_precision > 0.3 else 'NEEDS IMPROVEMENT'}")

print("\n" + "="*80)
