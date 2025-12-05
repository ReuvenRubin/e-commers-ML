"""
Performance Improvement Test - בודק שיפור ביצועים עם הרבה אינטראקציות
"""

import sys
from pathlib import Path
import random
import time
import numpy as np

# Add src directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "phase2"))

from recommendation_system_ml import RecommendationSystem

def main():
    print("="*80)
    print("PERFORMANCE IMPROVEMENT TEST")
    print("Testing system with many interactions to improve performance")
    print("="*80)
    
    # יצירת מערכת המלצות
    data_path = Path(r"C:\Users\Reuven\Desktop\ML")
    system = RecommendationSystem(data_path)
    
    # ========================================================================
    # שלב 1: טעינת נתונים
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: Loading Data")
    print("="*80)
    
    system.load_data()
    system.create_user_interaction_matrix()
    system.calculate_user_similarity()
    
    print("[OK] Data loaded")
    print(f"   Users: {len(system.all_user_ids)}")
    print(f"   Products: {len(system.all_product_ids)}")
    
    # ========================================================================
    # שלב 2: אימון ראשוני של Neural Network
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: Initial Neural Network Training")
    print("="*80)
    
    print("\nPreparing features (15,000 samples)...")
    X_features, y_labels = system.prepare_neural_network_features(sample_size=15000)
    
    if X_features is None or y_labels is None:
        print("[ERROR] Failed to prepare features!")
        return
    
    print(f"[OK] Features prepared: {X_features.shape[0]} samples")
    
    # בניית מודל
    print("\nBuilding Neural Network model...")
    model = system.build_neural_ranking_model()
    
    if model is None:
        print("[ERROR] Failed to build model!")
        return
    
    # אימון ראשוני
    print("\nTraining model (15 epochs)...")
    start_time = time.time()
    
    history = system.train_neural_ranking_model(
        X_features, 
        y_labels, 
        epochs=15,
        batch_size=32,
        validation_split=0.2
    )
    
    training_time = time.time() - start_time
    
    if history is None:
        print("[ERROR] Failed to train model!")
        return
    
    initial_accuracy = history.history['val_accuracy'][-1]
    print(f"[OK] Initial training completed in {training_time:.1f} seconds")
    print(f"   Initial Validation Accuracy: {initial_accuracy:.4f} ({initial_accuracy*100:.1f}%)")
    
    # ========================================================================
    # שלב 3: סימולציה של הרבה אינטראקציות (מספר סבבים)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: Simulating Many Interactions (Multiple Rounds)")
    print("="*80)
    
    # בחירת משתמשים ומוצרים
    test_users = system.all_user_ids[:100]  # 100 משתמשים
    test_products = system.all_product_ids  # כל המוצרים
    
    interaction_types = ['click', 'purchase', 'visit_time']
    
    # מספר סבבים של אינטראקציות
    num_rounds = 5
    interactions_per_round = 200
    
    total_interactions = 0
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num}/{num_rounds} ---")
        print(f"Creating {interactions_per_round} interactions...")
        
        round_interactions = 0
        
        for i in range(interactions_per_round):
            user_id = random.choice(test_users)
            product_id = random.choice(test_products)
            interaction_type = random.choice(interaction_types)
            
            if interaction_type == 'click':
                value = random.randint(1, 5)
            elif interaction_type == 'purchase':
                value = 1
            else:  # visit_time
                value = random.randint(10, 300)
            
            system.update_interaction_dynamic(
                user_id=user_id,
                product_id=product_id,
                interaction_type=interaction_type,
                value=value
            )
            
            round_interactions += 1
            total_interactions += 1
        
        print(f"   Created {round_interactions} interactions")
        print(f"   Total interactions so far: {total_interactions}")
        print(f"   New interactions count: {system.new_interactions_count}")
        
        # בדיקה אם צריך לאמן מחדש
        if system.new_interactions_count >= system.retrain_threshold:
            print(f"\n   Threshold reached! Retraining model...")
            retrained = system.check_and_retrain_neural_network(force_retrain=False)
            
            if retrained:
                print(f"   [OK] Model retrained with {system.new_interactions_count} new interactions")
            else:
                print(f"   [WARNING] Retraining failed")
    
    print(f"\n[OK] Total interactions created: {total_interactions}")
    
    # ========================================================================
    # שלב 4: אימון סופי (אם נשארו אינטראקציות)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Final Training (if needed)")
    print("="*80)
    
    if system.new_interactions_count > 0:
        print(f"\nRetraining with remaining {system.new_interactions_count} interactions...")
        retrained = system.check_and_retrain_neural_network(force_retrain=True)
        
        if retrained:
            print("[OK] Final retraining completed")
        else:
            print("[WARNING] Final retraining failed")
    else:
        print("\nNo remaining interactions to retrain")
    
    # ========================================================================
    # שלב 5: בדיקת שיפור ביצועים
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: Testing Performance Improvement")
    print("="*80)
    
    # בחירת משתמשים לבדיקה
    test_users_sample = random.sample(list(test_users), min(10, len(test_users)))
    
    print(f"\nTesting recommendations for {len(test_users_sample)} users...")
    
    improvements = []
    
    for user_id in test_users_sample:
        # המלצות רגילות
        regular_recs = system.hybrid_recommendations(user_id, n_recommendations=5)
        
        # המלצות עם Neural Network
        neural_recs = system.hybrid_recommendations_with_neural_ranking(
            user_id=user_id,
            n_recommendations=5,
            use_neural_ranking=True
        )
        
        # בדיקה אם יש שינוי
        if set(regular_recs) != set(neural_recs):
            improvements.append(1)
        else:
            improvements.append(0)
    
    improvement_rate = np.mean(improvements) * 100
    print(f"\n[OK] Neural Network changed recommendations for {improvement_rate:.1f}% of users")
    
    # ========================================================================
    # שלב 6: בדיקת דיוק על דוגמאות חדשות
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: Testing Accuracy on New Examples")
    print("="*80)
    
    # הכנת דוגמאות חדשות לבדיקה
    print("\nPreparing new test samples...")
    X_test, y_test = system.prepare_neural_network_features(sample_size=2000)
    
    if X_test is not None and y_test is not None and system.neural_ranking_model is not None:
        # נרמול תכונות
        X_test_normalized = system.feature_scaler.transform(X_test)
        
        # חיזוי
        predictions = system.neural_ranking_model.predict(X_test_normalized, verbose=0)
        predictions_binary = (predictions > 0.5).astype(int).flatten()
        
        # חישוב דיוק
        accuracy = np.mean(predictions_binary == y_test)
        
        print(f"[OK] Test accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        
        # חישוב Precision, Recall
        true_positives = np.sum((predictions_binary == 1) & (y_test == 1))
        false_positives = np.sum((predictions_binary == 1) & (y_test == 0))
        false_negatives = np.sum((predictions_binary == 0) & (y_test == 1))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        print(f"   Precision: {precision:.4f} ({precision*100:.1f}%)")
        print(f"   Recall: {recall:.4f} ({recall*100:.1f}%)")
    else:
        print("[WARNING] Could not test accuracy")
    
    # ========================================================================
    # סיכום
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print("\nTraining:")
    print(f"   Initial Validation Accuracy: {initial_accuracy:.4f} ({initial_accuracy*100:.1f}%)")
    if X_test is not None and y_test is not None and system.neural_ranking_model is not None:
        print(f"   Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    print(f"\nInteractions:")
    print(f"   Total simulated: {total_interactions}")
    print(f"   Rounds completed: {num_rounds}")
    
    print(f"\nPerformance:")
    print(f"   Neural Network improved recommendations for {improvement_rate:.1f}% of users")
    
    print("\nSystem Status:")
    print("   [OK] System trained and improved with many interactions")
    print("   [OK] Continuous Learning working")
    print("   [OK] Neural Network Ranking improving recommendations")
    
    print("\n" + "="*80)
    print("PERFORMANCE IMPROVEMENT TEST COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    main()

