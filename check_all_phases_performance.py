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

from product_categorization import ProductCategorization  # type: ignore
from user_categorization import UserCategorization  # type: ignore

project_root = Path(__file__).parent

# Product Categorization
print("\nRunning Product Categorization...")
pc = ProductCategorization(str(project_root))
product_results = pc.run_product_categorization()

# User Categorization
print("\nRunning User Categorization...")
uc = UserCategorization(str(project_root))
user_results = uc.run_phase1()

# Combine results
phase1_results = {
    'product_categorization': product_results,
    'user_categorization': user_results
}

print("\n" + "="*80)
print("PHASE 1 RESULTS SUMMARY:")
print("="*80)
print(f"Product Categorization:")
print(f"  - Algorithm: Logistic Regression")
if 'products_with_categories' in product_results:
    n_categories = product_results['products_with_categories']['predicted_category'].nunique()
    print(f"  - Number of categories: {n_categories}")
if 'metrics' in product_results:
    accuracy = product_results['metrics'].get('accuracy_combined', 0)
    print(f"  - Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

print(f"\nUser Categorization:")
print(f"  - Algorithm: Random Forest Classifier")
print(f"  - Number of categories: {user_results.get('n_categories', 0)}")
print(f"  - Accuracy: {user_results.get('accuracy', 0):.3f} ({user_results.get('accuracy', 0)*100:.1f}%)")

# Phase 2: Recommendation System
print("\n" + "="*80)
print("PHASE 2: Hybrid Recommendation System")
print("="*80)

from recommendation_system_ml import RecommendationSystem  # type: ignore

rec_system = RecommendationSystem(str(project_root))
phase2_results = rec_system.run_phase2()

print("\n" + "="*80)
print("PHASE 2 RESULTS SUMMARY:")
print("="*80)

# Load evaluation results
import pandas as pd
eval_path = project_root / "datasets" / "results" / "phase2" / "recommendation_evaluation.csv"
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

# Product Categorization
if 'product_categorization' in phase1_results:
    product_metrics = phase1_results['product_categorization'].get('metrics', {})
    product_accuracy = product_metrics.get('accuracy_combined', 0)
    print(f"\n1. Product Categorization:")
    print(f"   Accuracy: {product_accuracy:.3f} ({product_accuracy*100:.1f}%)")
    print(f"   Status: {'EXCELLENT' if product_accuracy > 0.8 else 'GOOD' if product_accuracy > 0.6 else 'NEEDS IMPROVEMENT'}")

# User Categorization
if 'user_categorization' in phase1_results:
    user_accuracy = phase1_results['user_categorization'].get('accuracy', 0)
    print(f"\n2. User Categorization:")
    print(f"   Accuracy: {user_accuracy:.3f} ({user_accuracy*100:.1f}%)")
    print(f"   Status: {'EXCELLENT' if user_accuracy > 0.8 else 'GOOD' if user_accuracy > 0.6 else 'NEEDS IMPROVEMENT'}")

if eval_path.exists() and len(eval_df) > 0:
    if 'precision@3' in eval_df.columns:
        avg_precision = eval_df['precision@3'].mean()
        print(f"\n3. Recommendation System (Phase 2):")
        print(f"   Precision@3: {avg_precision:.3f} ({avg_precision*100:.1f}%)")
        print(f"   Status: {'EXCELLENT' if avg_precision > 0.5 else 'GOOD' if avg_precision > 0.3 else 'NEEDS IMPROVEMENT'}")

print("\n" + "="*80)
