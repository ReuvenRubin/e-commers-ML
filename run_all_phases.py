"""
Main script to run all phases of the E-Commerce Recommendation System
מריץ את כל השלבים של מערכת ההמלצות
"""

import sys
from pathlib import Path

# Add src directories to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "phase1"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "phase2"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "phase3"))

def run_phase1():
    """Runs Phase 1: Product and User Categorization"""
    print("\n" + "="*80)
    print("Starting Phase 1: Product and User Categorization")
    print("="*80)
    
    from ml_implementation import MLImplementation  # type: ignore
    
    ml = MLImplementation(r"C:\Users\Reuven\Desktop\ML")
    results = ml.run_phase1()
    
    print("\nPhase 1 completed successfully!")
    return results

def run_phase1_with_train_test():
    """Runs Phase 1 with Train/Test Split"""
    print("\n" + "="*80)
    print("Starting Phase 1 with Train/Test Split")
    print("="*80)
    
    from ml_with_train_test import MLWithTrainTest  # type: ignore
    
    ml_system = MLWithTrainTest(r"C:\Users\Reuven\Desktop\ML")
    results = ml_system.run_ml_pipeline()
    
    print("\nPhase 1 with Train/Test completed successfully!")
    return results

def run_phase2():
    """Runs Phase 2: Hybrid Recommendation System"""
    print("\n" + "="*80)
    print("Starting Phase 2: Hybrid Recommendation System")
    print("="*80)
    
    from recommendation_system_ml import RecommendationSystem  # type: ignore
    
    rec_system = RecommendationSystem(r"C:\Users\Reuven\Desktop\ML")
    results = rec_system.run_phase2()
    
    print("\nPhase 2 completed successfully!")
    return results

def run_phase2_with_train_test():
    """Runs Phase 2 with Train/Test Split"""
    print("\n" + "="*80)
    print("Starting Phase 2 with Train/Test Split")
    print("="*80)
    
    from recommendation_system_ml_with_train_test import RecommendationSystemWithTrainTest  # type: ignore
    
    rec_system = RecommendationSystemWithTrainTest(r"C:\Users\Reuven\Desktop\ML")
    results = rec_system.run_phase2_with_train_test()
    
    print("\nPhase 2 with Train/Test completed successfully!")
    return results

def run_phase3():
    """Runs Phase 3: NLP Search System"""
    print("\n" + "="*80)
    print("Starting Phase 3: NLP Search System")
    print("="*80)
    
    from nlp_search_system import NLPSearchSystem  # type: ignore
    
    nlp_system = NLPSearchSystem(r"C:\Users\Reuven\Desktop\ML")
    results = nlp_system.run_phase3()
    
    print("\nPhase 3 completed successfully!")
    return results

def main():
    """Main function to run all phases"""
    print("="*80)
    print("E-Commerce Recommendation System - Full Pipeline")
    print("="*80)
    
    # Phase 1
    phase1_results = run_phase1()
    
    # Phase 2 (depends on Phase 1)
    phase2_results = run_phase2()
    
    # Phase 3 (optional - skip by default in non-interactive mode)
    try:
        run_phase3_choice = input("\nDo you want to run Phase 3 (NLP Search)? (y/n): ").lower()
        if run_phase3_choice == 'y':
            phase3_results = run_phase3()
    except (EOFError, KeyboardInterrupt):
        # Skip Phase 3 in non-interactive mode
        print("\nSkipping Phase 3 (non-interactive mode)")
    
    print("\n" + "="*80)
    print("All phases completed successfully!")
    print("="*80)
    
    return {
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

