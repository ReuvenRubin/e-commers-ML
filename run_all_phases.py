"""
Main script to run all phases of the E-Commerce Recommendation System
מריץ את כל השלבים של מערכת ההמלצות
"""

import sys
import subprocess
from pathlib import Path

# Get project root directory
project_root = Path(__file__).parent

# Add src directories to path
sys.path.insert(0, str(project_root / "src" / "phase1"))
sys.path.insert(0, str(project_root / "src" / "phase2"))
sys.path.insert(0, str(project_root / "src" / "phase3"))

def run_product_categorization():
    """Runs Product Categorization"""
    print("\n" + "="*80)
    print("Starting Product Categorization")
    print("="*80)
    
    # Run Product_Categorization.py
    product_cat_path = project_root / "ML_ofir" / "Machine_Learning" / "Product_Categorization.py"
    result = subprocess.run([sys.executable, str(product_cat_path)], cwd=str(project_root))
    
    if result.returncode != 0:
        raise RuntimeError("Product Categorization failed")
    
    print("\nProduct Categorization completed successfully!")
    return True

def run_phase1():
    """Runs Phase 1: User Categorization"""
    print("\n" + "="*80)
    print("Starting Phase 1: User Categorization")
    print("="*80)
    
    from ml_implementation import MLImplementation  # type: ignore
    
    ml = MLImplementation(str(project_root))
    results = ml.run_phase1()
    
    print("\nPhase 1 completed successfully!")
    return results

def run_phase1_with_train_test():
    """Runs Phase 1 with Train/Test Split"""
    print("\n" + "="*80)
    print("Starting Phase 1 with Train/Test Split")
    print("="*80)
    
    from ml_with_train_test import MLWithTrainTest  # type: ignore
    
    ml_system = MLWithTrainTest(str(project_root))
    results = ml_system.run_ml_pipeline()
    
    print("\nPhase 1 with Train/Test completed successfully!")
    return results

def run_phase2():
    """Runs Phase 2: Hybrid Recommendation System"""
    print("\n" + "="*80)
    print("Starting Phase 2: Hybrid Recommendation System")
    print("="*80)
    
    from recommendation_system_ml import RecommendationSystem  # type: ignore
    
    rec_system = RecommendationSystem(str(project_root))
    results = rec_system.run_phase2()
    
    print("\nPhase 2 completed successfully!")
    return results

def run_phase2_with_train_test():
    """Runs Phase 2 with Train/Test Split"""
    print("\n" + "="*80)
    print("Starting Phase 2 with Train/Test Split")
    print("="*80)
    
    from recommendation_system_ml_with_train_test import RecommendationSystemWithTrainTest  # type: ignore
    
    rec_system = RecommendationSystemWithTrainTest(str(project_root))
    results = rec_system.run_phase2_with_train_test()
    
    print("\nPhase 2 with Train/Test completed successfully!")
    return results

def run_phase3():
    """Runs Phase 3: NLP Search System"""
    print("\n" + "="*80)
    print("Starting Phase 3: NLP Search System")
    print("="*80)
    
    from nlp_search_system import NLPSearchSystem  # type: ignore
    
    nlp_system = NLPSearchSystem(str(project_root))
    results = nlp_system.run_phase3()
    
    print("\nPhase 3 completed successfully!")
    return results

def main():
    """Main function to run all phases"""
    print("="*80)
    print("E-Commerce Recommendation System - Full Pipeline")
    print("="*80)
    
    # Product Categorization (must run before Phase 1)
    print("\nStep 1: Running Product Categorization...")
    run_product_categorization()
    
    # Phase 1: User Categorization
    print("\nStep 2: Running Phase 1 (User Categorization)...")
    phase1_results = run_phase1()
    
    # Phase 2: Recommendation System (depends on Phase 1)
    print("\nStep 3: Running Phase 2 (Recommendation System)...")
    phase2_results = run_phase2()
    
    # Phase 3 (optional - skip by default in non-interactive mode)
    try:
        run_phase3_choice = input("\nDo you want to run Phase 3 (NLP Search)? (y/n): ").lower()
        if run_phase3_choice == 'y':
            print("\nStep 4: Running Phase 3 (NLP Search)...")
            phase3_results = run_phase3()
    except (EOFError, KeyboardInterrupt):
        # Skip Phase 3 in non-interactive mode
        print("\nSkipping Phase 3 (non-interactive mode)")
    
    print("\n" + "="*80)
    print("All phases completed successfully!")
    print("="*80)
    
    return {
        'product_categorization': True,
        'phase1': phase1_results,
        'phase2': phase2_results
    }

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

