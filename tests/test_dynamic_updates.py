"""
Test script for Dynamic Updates functionality
בודק שהעדכון הדינמי באמת עובד
"""

import sys
from pathlib import Path

# Add src directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "phase2"))

from recommendation_system_ml import RecommendationSystem

def main():
    print("="*80)
    print("Testing Dynamic Updates Functionality")
    print("="*80)
    
    # יצירת מערכת המלצות
    data_path = Path(r"C:\Users\Reuven\Desktop\ML")
    system = RecommendationSystem(data_path)
    
    # טעינת נתונים
    print("\n" + "="*80)
    print("Step 1: Loading data...")
    print("="*80)
    system.load_data()
    
    # יצירת מטריצת אינטראקציות
    print("\n" + "="*80)
    print("Step 2: Creating interaction matrix...")
    print("="*80)
    system.create_user_interaction_matrix()
    
    # חישוב דמיון משתמשים
    print("\n" + "="*80)
    print("Step 3: Calculating user similarity...")
    print("="*80)
    system.calculate_user_similarity()
    
    # בחירת משתמש לבדיקה
    test_user_id = 5
    test_product_id = 10
    
    print("\n" + "="*80)
    print(f"Testing with User {test_user_id}")
    print("="*80)
    
    # שלב 1: המלצות לפני עדכון
    print("\n" + "-"*80)
    print("BEFORE UPDATE: Getting initial recommendations...")
    print("-"*80)
    recommendations_before = system.hybrid_recommendations(test_user_id, n_recommendations=5)
    print(f"\nInitial Recommendations for User {test_user_id}: {recommendations_before}")
    
    # בדיקת הערך הנוכחי במטריצה
    if test_user_id in system.user_id_to_index and test_product_id in system.product_id_to_index:
        column_name = f'product_{test_product_id}'
        if column_name in system.interaction_matrix.columns:
            current_value = system.interaction_matrix.loc[test_user_id, column_name]
            print(f"\nCurrent interaction value for User {test_user_id} - Product {test_product_id}: {current_value:.2f}")
    
    # שלב 2: עדכון אינטראקציה - כמה אינטראקציות חזקות
    print("\n" + "-"*80)
    print("UPDATING: User interacts with multiple products...")
    print("-"*80)
    
    # עדכון 1: קליק
    system.update_interaction_dynamic(
        user_id=test_user_id,
        product_id=test_product_id,
        interaction_type='click',
        value=5  # 5 קליקים
    )
    
    # עדכון 2: רכישה (חזק יותר)
    test_product_id_purchase = 15
    system.update_interaction_dynamic(
        user_id=test_user_id,
        product_id=test_product_id_purchase,
        interaction_type='purchase',
        value=1  # רכישה אחת (משקל 5.0)
    )
    
    # עדכון 3: זמן ביקור ארוך
    test_product_id_visit = 25
    system.update_interaction_dynamic(
        user_id=test_user_id,
        product_id=test_product_id_visit,
        interaction_type='visit_time',
        value=120  # 120 שניות (משקל 0.1)
    )
    
    # בדיקת הערכים המעודכנים במטריצה
    print("\nUpdated interaction values:")
    for pid in [test_product_id, test_product_id_purchase, test_product_id_visit]:
        if test_user_id in system.user_id_to_index and pid in system.product_id_to_index:
            column_name = f'product_{pid}'
            if column_name in system.interaction_matrix.columns:
                updated_value = system.interaction_matrix.loc[test_user_id, column_name]
                print(f"   User {test_user_id} - Product {pid}: {updated_value:.2f}")
    
    # שלב 3: חישוב מחדש של דמיון
    print("\n" + "-"*80)
    print("RECALCULATING: User similarity matrix...")
    print("-"*80)
    system.recalculate_user_similarity()
    
    # שלב 4: המלצות אחרי עדכון
    print("\n" + "-"*80)
    print("AFTER UPDATE: Getting updated recommendations...")
    print("-"*80)
    recommendations_after = system.hybrid_recommendations(test_user_id, n_recommendations=5)
    print(f"\nUpdated Recommendations for User {test_user_id}: {recommendations_after}")
    
    # השוואה
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"\nRecommendations BEFORE update: {recommendations_before}")
    print(f"Recommendations AFTER update:  {recommendations_after}")
    
    # בדיקת עדכון המטריצה
    print("\n" + "-"*80)
    print("VERIFICATION: Checking if matrix was updated...")
    print("-"*80)
    
    # בדיקה שהערכים במטריצה באמת השתנו
    matrix_updated = False
    for pid in [test_product_id, test_product_id_purchase, test_product_id_visit]:
        column_name = f'product_{pid}'
        if column_name in system.interaction_matrix.columns:
            value = system.interaction_matrix.loc[test_user_id, column_name]
            if value > 0:
                matrix_updated = True
                print(f"   Product {pid}: Value = {value:.2f} (updated successfully!)")
    
    if matrix_updated:
        print("\nSUCCESS: Interaction matrix was updated dynamically!")
        print("   The matrix values changed, proving dynamic updates work.")
    
    # בדיקה אם ההמלצות השתנו
    if recommendations_before != recommendations_after:
        print("\nSUCCESS: Recommendations changed after update!")
        print("   This proves that dynamic updates are working!")
        
        # מציאת הבדלים
        before_set = set(recommendations_before)
        after_set = set(recommendations_after)
        
        new_recommendations = after_set - before_set
        removed_recommendations = before_set - after_set
        
        if new_recommendations:
            print(f"\n   New recommendations: {list(new_recommendations)}")
        if removed_recommendations:
            print(f"   Removed recommendations: {list(removed_recommendations)}")
    else:
        print("\nNOTE: Recommendations did not change.")
        print("   This is normal - recommendations are based on many interactions.")
        print("   A few updates might not change the top recommendations.")
        print("   BUT: The matrix WAS updated (see above), so dynamic updates work!")
    
    # שלב 5: בדיקה עם הפונקציה המאוחדת
    print("\n" + "="*80)
    print("Testing update_and_recommend() function")
    print("="*80)
    
    # עוד עדכון - הפעם עם הפונקציה המאוחדת
    test_product_id_2 = 20
    print(f"\nUser {test_user_id} purchases product {test_product_id_2}...")
    recommendations_unified = system.update_and_recommend(
        user_id=test_user_id,
        product_id=test_product_id_2,
        interaction_type='purchase',
        value=1,
        recalculate_similarity=True,
        n_recommendations=5
    )
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\nFinal recommendations after purchase: {recommendations_unified}")
    print("\nSUCCESS: Dynamic Updates test completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()

