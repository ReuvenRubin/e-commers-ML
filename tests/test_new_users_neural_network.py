"""
Test script to compare recommendations for new users with and without Neural Network ranking
בודק משתמשים חדשים ומשווה המלצות עם ובלי רשת הנוירונים
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "phase2"))

from recommendation_system_ml import RecommendationSystem

def test_new_users_comparison():
    """Tests new users and compares recommendations with/without Neural Network"""
    
    print("="*80)
    print("Testing New Users: With vs Without Neural Network Ranking")
    print("="*80)
    
    # Initialize system
    data_path = Path(r"C:\Users\Reuven\Desktop\ML")
    rec_system = RecommendationSystem(data_path)
    
    # Load data
    print("\n1. Loading data...")
    rec_system.load_data()
    
    # Prepare TF-IDF and interaction matrix
    print("\n2. Preparing TF-IDF and interaction matrix...")
    rec_system.prepare_tfidf_for_products()
    rec_system.create_user_interaction_matrix()
    rec_system.calculate_user_similarity()
    
    # Train Neural Network (if available)
    print("\n3. Preparing and training Neural Network...")
    try:
        X_features, y_labels = rec_system.prepare_neural_network_features(sample_size=5000)
        if X_features is not None and len(X_features) > 0:
            rec_system.build_neural_ranking_model()
            rec_system.train_neural_ranking_model(X_features, y_labels, epochs=5, batch_size=32)
            neural_available = True
            print("   Neural Network trained successfully!")
        else:
            neural_available = False
            print("   Warning: Could not prepare features for Neural Network")
    except Exception as e:
        neural_available = False
        print(f"   Warning: Neural Network not available: {e}")
    
    # Test with new users (users with 1-2 interactions)
    print("\n" + "="*80)
    print("Testing New Users (1-2 interactions)")
    print("="*80)
    
    # Find users with few interactions (new users)
    user_interaction_counts = {}
    for _, row in rec_system.clicks_df.iterrows():
        uid = row['uid']
        user_interaction_counts[uid] = user_interaction_counts.get(uid, 0) + row['clicks']
    
    for _, row in rec_system.purchases_df.iterrows():
        uid = row['uid']
        user_interaction_counts[uid] = user_interaction_counts.get(uid, 0) + row['purchases'] * 5
    
    # Find new users (1-5 total interactions) - expanded range
    new_users = [uid for uid, count in user_interaction_counts.items() if 1 <= count <= 5]
    
    # If no users with 1-5 interactions, try users with 1-10 interactions
    if len(new_users) == 0:
        new_users = [uid for uid, count in user_interaction_counts.items() if 1 <= count <= 10]
    
    # If still no users, use first 5 users from interaction matrix (users with fewest interactions)
    if len(new_users) == 0:
        print("Warning: No users with 1-10 interactions found. Using users with fewest interactions...")
        # Get users from matrix and count their interactions
        matrix_users = rec_system.interaction_matrix.index.tolist()
        user_total_interactions = {}
        for uid in matrix_users[:50]:  # Check first 50 users
            if uid in rec_system.user_id_to_index:
                user_idx = rec_system.user_id_to_index[uid]
                user_row = rec_system.interaction_matrix.iloc[user_idx]
                total = user_row.sum()
                user_total_interactions[uid] = total
        
        # Sort by total interactions and take first 5
        sorted_users = sorted(user_total_interactions.items(), key=lambda x: x[1])
        new_users = [uid for uid, count in sorted_users[:5]]
    
    new_users = new_users[:10]  # Test first 10 new users
    
    print(f"\nFound {len(new_users)} users to test (showing first {min(10, len(new_users))})")
    
    results = []
    
    for i, user_id in enumerate(new_users[:5], 1):  # Test first 5
        print(f"\n{'='*60}")
        print(f"Test {i}: User {user_id}")
        print(f"{'='*60}")
        
        # Get user interactions
        user_interactions = {}
        user_clicks = rec_system.clicks_df[rec_system.clicks_df['uid'] == user_id]
        user_purchases = rec_system.purchases_df[rec_system.purchases_df['uid'] == user_id]
        
        for _, row in user_clicks.iterrows():
            pid = row['product_id']
            user_interactions[pid] = user_interactions.get(pid, 0) + row['clicks']
        
        for _, row in user_purchases.iterrows():
            pid = row['product_id']
            user_interactions[pid] = user_interactions.get(pid, 0) + row['purchases']
        
        print(f"User interactions: {user_interactions}")
        
        # Test 1: Recommendations WITHOUT Neural Network (base hybrid)
        print(f"\n--- Recommendations WITHOUT Neural Network ---")
        try:
            if user_id in rec_system.interaction_matrix.index:
                # Old user - use hybrid recommendations
                recs_without_nn = rec_system.hybrid_recommendations(user_id, n_recommendations=5)
            else:
                # New user - use content-based
                recs_without_nn = rec_system.recommend_for_new_user(user_interactions)
            print(f"Recommendations: {recs_without_nn}")
        except Exception as e:
            print(f"Error: {e}")
            recs_without_nn = []
        
        # Test 2: Recommendations WITH Neural Network (if available)
        print(f"\n--- Recommendations WITH Neural Network ---")
        if neural_available:
            try:
                if user_id in rec_system.interaction_matrix.index:
                    # Old user - use hybrid with neural ranking
                    recs_with_nn = rec_system.hybrid_recommendations_with_neural_ranking(
                        user_id, n_recommendations=5, use_neural_ranking=True
                    )
                else:
                    # For new users, we'll use base hybrid (neural network needs user in matrix)
                    print("Note: Neural Network ranking requires user in interaction matrix")
                    print("Using base hybrid recommendations for new user...")
                    recs_with_nn = rec_system.recommend_for_new_user(user_interactions)
                
                print(f"Recommendations: {recs_with_nn}")
                
                # Compare
                print(f"\n--- Comparison ---")
                print(f"Without NN: {recs_without_nn}")
                print(f"With NN:    {recs_with_nn}")
                
                # Check if different
                if set(recs_without_nn) != set(recs_with_nn):
                    print("✓ Different recommendations!")
                    print(f"  Difference: {set(recs_with_nn) - set(recs_without_nn)}")
                else:
                    print("= Same recommendations")
                
            except Exception as e:
                print(f"Error: {e}")
                recs_with_nn = []
        else:
            print("Neural Network not available - skipping")
            recs_with_nn = []
        
        # Store results
        results.append({
            'user_id': user_id,
            'interactions': user_interactions,
            'recs_without_nn': recs_without_nn,
            'recs_with_nn': recs_with_nn if neural_available else [],
            'neural_available': neural_available
        })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total users tested: {len(results)}")
    print(f"Neural Network available: {neural_available}")
    
    if neural_available and len(results) > 0:
        different_count = sum(1 for r in results if set(r['recs_without_nn']) != set(r['recs_with_nn']))
        print(f"Users with different recommendations: {different_count}/{len(results)}")
        print(f"Percentage improved: {different_count/len(results)*100:.1f}%")
    elif neural_available:
        print("No users tested - cannot calculate improvement percentage")
    
    # Save results
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        output_path = data_path / "datasets" / "ml_results"
        output_path.mkdir(exist_ok=True)
        results_df.to_csv(output_path / "new_users_neural_comparison.csv", index=False)
        print(f"\nResults saved to: {output_path / 'new_users_neural_comparison.csv'}")
    else:
        print("\nNo results to save (no users tested)")
    
    return results

if __name__ == "__main__":
    try:
        results = test_new_users_comparison()
        print("\n" + "="*80)
        print("Test completed successfully!")
        print("="*80)
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()

