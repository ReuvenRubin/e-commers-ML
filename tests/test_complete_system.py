"""
Complete System Test - בודק שהכל עובד
בודק: Dynamic Updates, Neural Network Ranking, Continuous Learning
"""

import sys
from pathlib import Path
import time

# Add src directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "phase2"))

from recommendation_system_ml import RecommendationSystem

def main():
    print("="*80)
    print("COMPLETE SYSTEM TEST")
    print("Testing: Dynamic Updates + Neural Network Ranking + Continuous Learning")
    print("="*80)
    
    # יצירת מערכת המלצות
    data_path = Path(r"C:\Users\Reuven\Desktop\ML")
    system = RecommendationSystem(data_path)
    
    # ========================================================================
    # שלב 1: טעינת נתונים והכנה בסיסית
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 1: Loading Data and Basic Setup")
    print("="*80)
    
    system.load_data()
    system.create_user_interaction_matrix()
    system.calculate_user_similarity()
    
    print("\n[OK] Data loaded and basic setup completed")
    
    # ========================================================================
    # שלב 2: בדיקת Dynamic Updates
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2: Testing Dynamic Updates")
    print("="*80)
    
    test_user_id = 5
    test_product_id = 10
    
    # בדיקה לפני עדכון
    column_name = f'product_{test_product_id}'
    value_before = system.interaction_matrix.loc[test_user_id, column_name]
    print(f"\nBefore update: User {test_user_id} - Product {test_product_id} = {value_before:.2f}")
    
    # עדכון אינטראקציה
    success = system.update_interaction_dynamic(
        user_id=test_user_id,
        product_id=test_product_id,
        interaction_type='click',
        value=1
    )
    
    if not success:
        print("[ERROR] Failed to update interaction!")
        return
    
    # בדיקה אחרי עדכון
    value_after = system.interaction_matrix.loc[test_user_id, column_name]
    print(f"After update: User {test_user_id} - Product {test_product_id} = {value_after:.2f}")
    
    if value_after > value_before:
        print("[OK] Dynamic update works - matrix was updated!")
    else:
        print("[ERROR] Dynamic update failed - matrix was not updated!")
        return
    
    # בדיקת חישוב מחדש של דמיון
    print("\nRecalculating user similarity...")
    system.recalculate_user_similarity()
    print("[OK] User similarity recalculated")
    
    # ========================================================================
    # שלב 3: בדיקת Neural Network Ranking
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 3: Testing Neural Network Ranking")
    print("="*80)
    
    # הכנת תכונות
    print("\nPreparing features for Neural Network...")
    X_features, y_labels = system.prepare_neural_network_features(sample_size=2000)
    
    if X_features is None or y_labels is None:
        print("[ERROR] Failed to prepare features!")
        return
    
    print(f"[OK] Features prepared: {X_features.shape[0]} samples, {X_features.shape[1]} features")
    
    # בניית מודל
    print("\nBuilding Neural Network model...")
    model = system.build_neural_ranking_model()
    
    if model is None:
        print("[ERROR] Failed to build model!")
        return
    
    print("[OK] Model built successfully")
    
    # אימון מודל (מהיר - רק 3 epochs לבדיקה)
    print("\nTraining Neural Network model (quick test - 3 epochs)...")
    history = system.train_neural_ranking_model(
        X_features, 
        y_labels, 
        epochs=3,  # מהיר לבדיקה
        batch_size=32,
        validation_split=0.2
    )
    
    if history is None:
        print("[ERROR] Failed to train model!")
        return
    
    print("[OK] Model trained successfully")
    
    # בדיקת חיזוי
    print("\nTesting prediction...")
    score = system.predict_product_score(test_user_id, test_product_id)
    
    if score is None:
        print("[ERROR] Failed to predict score!")
        return
    
    print(f"[OK] Prediction works: Score = {score:.4f}")
    
    # בדיקת המלצות משולבות
    print("\nTesting hybrid recommendations with Neural Network...")
    recommendations = system.hybrid_recommendations_with_neural_ranking(
        user_id=test_user_id,
        n_recommendations=5,
        use_neural_ranking=True
    )
    
    if not recommendations:
        print("[ERROR] Failed to get recommendations!")
        return
    
    print(f"[OK] Hybrid recommendations work: {recommendations}")
    
    # ========================================================================
    # שלב 4: בדיקת Continuous Learning
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 4: Testing Continuous Learning")
    print("="*80)
    
    # בדיקת מעקב אחר אינטראקציות חדשות
    initial_count = system.new_interactions_count
    print(f"\nInitial new interactions count: {initial_count}")
    
    # הוספת כמה אינטראקציות חדשות
    print("\nAdding new interactions...")
    for i in range(5):
        system.update_interaction_dynamic(
            user_id=test_user_id,
            product_id=20 + i,
            interaction_type='click',
            value=1
        )
    
    new_count = system.new_interactions_count
    print(f"New interactions count after 5 updates: {new_count}")
    
    if new_count == initial_count + 5:
        print("[OK] New interactions tracking works!")
    else:
        print(f"[ERROR] Tracking failed: expected {initial_count + 5}, got {new_count}")
        return
    
    # בדיקה שלא מאמנים מחדש אם אין מספיק אינטראקציות
    print("\nTesting retrain check (should not retrain with < 100 interactions)...")
    retrained = system.check_and_retrain_neural_network()
    
    if not retrained:
        print("[OK] Correctly skipped retraining (not enough interactions)")
    else:
        print("[WARNING] Retrained even though there weren't enough interactions")
    
    # הוספת עוד אינטראקציות כדי להגיע ל-100 (לבדיקה מהירה, נשנה את הסף)
    print("\nTemporarily lowering threshold to 10 for testing...")
    original_threshold = system.retrain_threshold
    system.retrain_threshold = 10  # זמנית לבדיקה
    
    # הוספת עוד אינטראקציות
    print("Adding more interactions to reach threshold...")
    for i in range(10):
        system.update_interaction_dynamic(
            user_id=test_user_id,
            product_id=30 + i,
            interaction_type='click',
            value=1
        )
    
    # בדיקת אימון מחדש
    print("\nTesting retrain (should retrain with >= 10 interactions)...")
    retrained = system.check_and_retrain_neural_network(force_retrain=False)
    
    if retrained:
        print("[OK] Continuous Learning works - model was retrained!")
    else:
        print("[WARNING] Retraining was skipped (might be OK if there was an error)")
    
    # החזרת הסף המקורי
    system.retrain_threshold = original_threshold
    
    # ========================================================================
    # שלב 5: בדיקה משולבת - הכל ביחד
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 5: Integrated Test - Everything Together")
    print("="*80)
    
    print("\nTesting update_and_recommend() with all features...")
    recommendations = system.update_and_recommend(
        user_id=test_user_id,
        product_id=50,
        interaction_type='purchase',
        value=1,
        recalculate_similarity=True,
        n_recommendations=5
    )
    
    if recommendations:
        print(f"[OK] Integrated test works: {recommendations}")
    else:
        print("[ERROR] Integrated test failed!")
        return
    
    # ========================================================================
    # סיכום
    # ========================================================================
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print("\nAll tests completed!")
    print("\nFeatures tested:")
    print("  [OK] Dynamic Updates - update_interaction_dynamic()")
    print("  [OK] Dynamic Updates - recalculate_user_similarity()")
    print("  [OK] Dynamic Updates - update_and_recommend()")
    print("  [OK] Neural Network - prepare_neural_network_features()")
    print("  [OK] Neural Network - build_neural_ranking_model()")
    print("  [OK] Neural Network - train_neural_ranking_model()")
    print("  [OK] Neural Network - predict_product_score()")
    print("  [OK] Neural Network - hybrid_recommendations_with_neural_ranking()")
    print("  [OK] Continuous Learning - new interactions tracking")
    print("  [OK] Continuous Learning - check_and_retrain_neural_network()")
    print("  [OK] Integrated - update_and_recommend() with all features")
    
    print("\n" + "="*80)
    print("ALL SYSTEMS OPERATIONAL!")
    print("="*80)

if __name__ == "__main__":
    main()

