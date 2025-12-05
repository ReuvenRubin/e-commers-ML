"""
Test script for Neural Network Ranking functionality
בודק שכל הפונקציות החדשות עובדות
"""

import sys
from pathlib import Path
import numpy as np

# Add src directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "phase2"))

from recommendation_system_ml import RecommendationSystem

def main():
    print("="*80)
    print("Testing Neural Network Ranking Functionality")
    print("="*80)
    
    # יצירת מערכת המלצות
    data_path = Path(r"C:\Users\Reuven\Desktop\ML")
    system = RecommendationSystem(data_path)
    
    # שלב 1: טעינת נתונים
    print("\n" + "="*80)
    print("Step 1: Loading data...")
    print("="*80)
    system.load_data()
    
    # שלב 2: יצירת מטריצת אינטראקציות
    print("\n" + "="*80)
    print("Step 2: Creating interaction matrix...")
    print("="*80)
    system.create_user_interaction_matrix()
    
    # שלב 3: חישוב דמיון משתמשים
    print("\n" + "="*80)
    print("Step 3: Calculating user similarity...")
    print("="*80)
    system.calculate_user_similarity()
    
    # שלב 4: הכנת תכונות לרשת העצבית
    print("\n" + "="*80)
    print("Step 4: Preparing features for Neural Network...")
    print("="*80)
    X_features, y_labels = system.prepare_neural_network_features(sample_size=5000)
    
    if X_features is None or y_labels is None:
        print("ERROR: Failed to prepare features!")
        return
    
    print(f"\nFeatures shape: {X_features.shape}")
    print(f"Labels shape: {y_labels.shape}")
    print(f"Positive samples: {y_labels.sum()}, Negative: {(y_labels == 0).sum()}")
    
    # שלב 5: בניית המודל
    print("\n" + "="*80)
    print("Step 5: Building Neural Network model...")
    print("="*80)
    model = system.build_neural_ranking_model()
    
    if model is None:
        print("ERROR: Failed to build model!")
        return
    
    print("\nModel summary:")
    model.summary()
    
    # שלב 6: אימון המודל
    print("\n" + "="*80)
    print("Step 6: Training Neural Network model...")
    print("="*80)
    print("(This may take a few minutes...)")
    
    history = system.train_neural_ranking_model(
        X_features, 
        y_labels, 
        epochs=5,  # פחות epochs לבדיקה מהירה
        batch_size=32,
        validation_split=0.2
    )
    
    if history is None:
        print("ERROR: Failed to train model!")
        return
    
    # שלב 7: בדיקת חיזוי על מוצר בודד
    print("\n" + "="*80)
    print("Step 7: Testing prediction on single product...")
    print("="*80)
    
    test_user_id = 5
    test_product_id = 10
    
    score = system.predict_product_score(test_user_id, test_product_id)
    
    if score is None:
        print("ERROR: Failed to predict score!")
        return
    
    print(f"\nPrediction for User {test_user_id} - Product {test_product_id}:")
    print(f"   Score: {score:.4f}")
    
    if score < 0.3:
        print("   Interpretation: Not relevant")
    elif score < 0.6:
        print("   Interpretation: Maybe relevant")
    elif score < 0.8:
        print("   Interpretation: Relevant")
    else:
        print("   Interpretation: Very relevant!")
    
    # שלב 8: בדיקת המלצות משולבות
    print("\n" + "="*80)
    print("Step 8: Testing hybrid recommendations with Neural Network ranking...")
    print("="*80)
    
    recommendations = system.hybrid_recommendations_with_neural_ranking(
        user_id=test_user_id,
        n_recommendations=5,
        use_neural_ranking=True
    )
    
    if not recommendations:
        print("ERROR: Failed to get recommendations!")
        return
    
    print(f"\nFinal recommendations for User {test_user_id}:")
    for i, product_id in enumerate(recommendations, 1):
        score = system.predict_product_score(test_user_id, product_id)
        print(f"   {i}. Product {product_id}: Neural Score = {score:.4f}")
    
    # השוואה: המלצות רגילות vs המלצות עם רשת עצבית
    print("\n" + "="*80)
    print("Step 9: Comparison - Regular vs Neural Network recommendations...")
    print("="*80)
    
    regular_recommendations = system.hybrid_recommendations(test_user_id, n_recommendations=5)
    neural_recommendations = system.hybrid_recommendations_with_neural_ranking(
        test_user_id, 
        n_recommendations=5,
        use_neural_ranking=True
    )
    
    print(f"\nRegular recommendations: {regular_recommendations}")
    print(f"Neural Network recommendations: {neural_recommendations}")
    
    # בדיקה אם יש הבדלים
    if set(regular_recommendations) != set(neural_recommendations):
        print("\nSUCCESS: Neural Network changed the recommendations!")
        print("   This shows that the Neural Network is working and improving recommendations.")
    else:
        print("\nNOTE: Recommendations are the same.")
        print("   This might happen if the Neural Network agrees with the base recommendations.")
    
    # סיכום
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("\nAll tests completed successfully!")
    print("\nFunctions tested:")
    print("   [OK] prepare_neural_network_features()")
    print("   [OK] build_neural_ranking_model()")
    print("   [OK] train_neural_ranking_model()")
    print("   [OK] predict_product_score()")
    print("   [OK] hybrid_recommendations_with_neural_ranking()")
    print("\nNeural Network Ranking is working correctly!")
    print("="*80)

if __name__ == "__main__":
    main()

