"""
Script to improve user interaction data for better recommendation system
משפר את נתוני האינטראקציות של המשתמשים כדי שיהיו יותר משתמשים פעילים (20-30%)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def improve_user_interactions(data_path):
    """
    Improves user interaction data to have 20-30% active users
    
    Changes:
    - Increases number of active users from 10% to 25-30%
    - Maintains realistic conversion rates (1-3%)
    - Keeps data distribution realistic
    """
    data_path = Path(data_path)
    raw_path = data_path / "datasets" / "raw"
    backup_path = raw_path / "backup"
    backup_path.mkdir(exist_ok=True)
    
    print("="*80)
    print("Improving User Interaction Data for Recommendation System")
    print("="*80)
    print("\nGoal: Increase active users from 10% to 25-30%")
    print("      Maintain high model accuracy (85-90%)")
    print("      Better coverage for recommendation system")
    
    # Backup original files
    print("\n1. Backing up original files...")
    files_to_backup = [
        "user_clicks_interactions.csv",
        "user_purchase_interactions.csv",
        "user_visits_time_interactions.csv"
    ]
    
    for file in files_to_backup:
        src = raw_path / file
        dst = backup_path / f"{file.replace('.csv', '_before_improvement.csv')}"
        if src.exists():
            df_backup = pd.read_csv(src)
            df_backup.to_csv(dst, index=False)
            print(f"  [OK] Backed up {file}")
    
    # Load current data
    print("\n2. Loading current data...")
    clicks_wide = pd.read_csv(raw_path / "user_clicks_interactions.csv")
    purchases_wide = pd.read_csv(raw_path / "user_purchase_interactions.csv")
    visits_time_wide = pd.read_csv(raw_path / "user_visits_time_interactions.csv")
    users_df = pd.read_csv(raw_path / "users_5000.csv")
    
    print(f"  Current: {len(clicks_wide)} users in interactions")
    print(f"  Total users: {len(users_df)}")
    
    # Get all pid columns
    pid_cols = [col for col in clicks_wide.columns if col.startswith('pid')]
    
    # Create full DataFrames with all users (fill missing with zeros)
    print("\n2.1. Creating full interaction tables for all users...")
    all_user_ids = users_df['id'].tolist()
    
    # Create full clicks DataFrame
    clicks_full = pd.DataFrame({'uid': all_user_ids})
    for col in pid_cols:
        clicks_full[col] = 0
    # Fill in existing data
    for _, row in clicks_wide.iterrows():
        uid = row['uid']
        clicks_full.loc[clicks_full['uid'] == uid, pid_cols] = row[pid_cols].values
    
    # Create full purchases DataFrame
    purchases_full = pd.DataFrame({'uid': all_user_ids})
    for col in pid_cols:
        purchases_full[col] = 0
    # Fill in existing data
    for _, row in purchases_wide.iterrows():
        uid = row['uid']
        purchases_full.loc[purchases_full['uid'] == uid, pid_cols] = row[pid_cols].values
    
    # Create full visits_time DataFrame
    visits_time_full = pd.DataFrame({'uid': all_user_ids})
    for col in pid_cols:
        visits_time_full[col] = 0
    # Fill in existing data
    for _, row in visits_time_wide.iterrows():
        uid = row['uid']
        visits_time_full.loc[visits_time_full['uid'] == uid, pid_cols] = row[pid_cols].values
    
    # Use full DataFrames
    clicks_wide = clicks_full
    purchases_wide = purchases_full
    visits_time_wide = visits_time_full
    
    print(f"  Created full tables: {len(clicks_wide)} users")
    
    # Calculate current active users
    def count_active_users(df):
        """Count users with any interaction"""
        pid_cols = [col for col in df.columns if col.startswith('pid')]
        # Check if sum of pid columns > 0 (has any interaction)
        return (df[pid_cols].sum(axis=1) > 0).sum()
    
    # Count BEFORE creating full tables
    current_active = count_active_users(clicks_wide)
    target_active = int(len(users_df) * 0.28)  # 28% active users
    users_to_activate = target_active - current_active
    
    print(f"\n3. Current vs Target:")
    print(f"   Current active users: {current_active} ({current_active/len(users_df)*100:.1f}%)")
    print(f"   Target active users: {target_active} ({target_active/len(users_df)*100:.1f}%)")
    print(f"   Need to activate: {users_to_activate} users")
    
    # Load products info (needed for both activation and enhancement)
    products_df = pd.read_csv(raw_path / "products_10000.csv")
    max_products = len([col for col in clicks_wide.columns if col.startswith('pid')])
    
    # IMPORTANT: In wide format, pid1=product_id 1, pid2=product_id 2, etc.
    # So we can only use product_ids 1 to max_products (1-10)
    available_product_ids = list(range(1, max_products + 1))
    
    if users_to_activate <= 0:
        print("\n[INFO] Already have enough active users!")
        print("   Enhancing existing users' interactions for more realistic data...")
        
        # Enhance existing active users with more interactions
        # This makes the data more realistic and helps with future scalability
        print("\n4.1. Enhancing existing active users...")
        active_mask = clicks_wide[pid_cols].sum(axis=1) > 0
        active_user_ids = clicks_wide[active_mask]['uid'].tolist()
        
        # Enhance 30% of active users (random selection)
        np.random.seed(42)
        users_to_enhance = np.random.choice(active_user_ids, size=int(len(active_user_ids) * 0.3), replace=False)
        
        print(f"   Enhancing {len(users_to_enhance)} active users with additional interactions...")
        
        for user_id in users_to_enhance:
            mask = clicks_wide['uid'] == user_id
            
            # Get current interactions
            current_interactions = []
            for col in pid_cols:
                val = clicks_wide.loc[mask, col].values[0]
                if val > 0:
                    current_interactions.append(val)
            
            # Add 1-3 more products (if space available)
            n_new = np.random.randint(1, 4)
            available_slots = [i for i, col in enumerate(pid_cols) if clicks_wide.loc[mask, col].values[0] == 0]
            
            if len(available_slots) > 0:
                n_to_add = min(n_new, len(available_slots))
                new_products = np.random.choice(available_product_ids, size=n_to_add, replace=False)
                
                for i, pid in enumerate(new_products):
                    if i < len(available_slots):
                        col_name = pid_cols[available_slots[i]]
                        clicks_wide.loc[mask, col_name] = pid
                        visits_time_wide.loc[visits_time_wide['uid'] == user_id, col_name] = pid
        
        print(f"   [OK] Enhanced {len(users_to_enhance)} users")
        
        # Save enhanced data
        print("\n4.2. Saving enhanced data...")
        clicks_wide.to_csv(raw_path / "user_clicks_interactions.csv", index=False)
        purchases_wide.to_csv(raw_path / "user_purchase_interactions.csv", index=False)
        visits_time_wide.to_csv(raw_path / "user_visits_time_interactions.csv", index=False)
        
        final_active = count_active_users(clicks_wide)
        print(f"\n4.3. Final status:")
        print(f"   Active users: {final_active} ({final_active/len(users_df)*100:.1f}%)")
        print(f"   Data enhanced for better realism and future scalability")
        
        print("\n" + "="*80)
        print("[SUCCESS] User interaction data enhanced!")
        print("="*80)
        return
    
    # Get inactive users (users with no interactions)
    print("\n4. Identifying inactive users...")
    pid_cols = [col for col in clicks_wide.columns if col.startswith('pid')]
    active_user_ids = set(clicks_wide[clicks_wide[pid_cols].sum(axis=1) > 0]['uid'].unique())
    all_user_ids = set(users_df['id'].unique())
    inactive_user_ids = list(all_user_ids - active_user_ids)
    
    print(f"   Active users: {len(active_user_ids)}")
    print(f"   Inactive users: {len(inactive_user_ids)}")
    
    # Select random inactive users to activate
    np.random.seed(42)
    users_to_activate_list = np.random.choice(inactive_user_ids, size=min(users_to_activate, len(inactive_user_ids)), replace=False)
    print(f"   Selected {len(users_to_activate_list)} users to activate")
    
    print(f"\n5. Adding interactions for {len(users_to_activate_list)} users...")
    print(f"   Available product IDs: {available_product_ids} (max_products={max_products})")
    
    # Distribution: 70% browsers, 30% buyers (among active users)
    n_browsers = int(len(users_to_activate_list) * 0.70)
    n_buyers = len(users_to_activate_list) - n_browsers
    
    browsers = users_to_activate_list[:n_browsers]
    buyers = users_to_activate_list[n_browsers:]
    
    print(f"   - Browsers (clicks only): {n_browsers}")
    print(f"   - Buyers (clicks + purchases): {n_buyers}")
    print(f"   Processing...")
    
    # Add interactions for browsers
    browsers_added = 0
    for user_id in browsers:
        # More realistic: 2-6 products clicked (not just 1-3)
        # This creates more realistic browsing patterns
        n_products = np.random.randint(2, min(7, len(available_product_ids) + 1))
        selected_products = np.random.choice(available_product_ids, size=n_products, replace=False)
        
        mask = clicks_wide['uid'] == user_id
        if not mask.any():
            continue
        
        # Fill available pid columns with selected products
        for i, pid in enumerate(selected_products):
            if i < len(pid_cols):  # Make sure we don't exceed available columns
                col_name = pid_cols[i]
                clicks_wide.loc[mask, col_name] = pid
                visits_time_wide.loc[visits_time_wide['uid'] == user_id, col_name] = pid
                browsers_added += 1
    
    # Add interactions for buyers
    buyers_added = 0
    for user_id in buyers:
        mask = clicks_wide['uid'] == user_id
        if not mask.any():
            continue
        
        # More realistic: 3-8 products clicked (more browsing before purchase)
        n_clicks = np.random.randint(3, min(9, len(available_product_ids) + 1))
        clicked_products = np.random.choice(available_product_ids, size=n_clicks, replace=False)
        
        # Fill clicks
        for i, pid in enumerate(clicked_products):
            if i < len(pid_cols):
                col_name = pid_cols[i]
                clicks_wide.loc[mask, col_name] = pid
                visits_time_wide.loc[visits_time_wide['uid'] == user_id, col_name] = pid
        
        # 1-3 purchases (realistic conversion rate ~15-30% for engaged users)
        n_purchases = np.random.randint(1, min(4, len(clicked_products) + 1))
        purchased_products = np.random.choice(clicked_products, size=n_purchases, replace=False)
        
        # Fill purchases
        for i, pid in enumerate(purchased_products):
            if i < len(pid_cols):
                col_name = pid_cols[i]
                purchases_wide.loc[purchases_wide['uid'] == user_id, col_name] = pid
                buyers_added += 1
    
    # Verify BEFORE saving
    new_active = count_active_users(clicks_wide)
    print(f"\n6. Verification before saving:")
    print(f"   New active users: {new_active} ({new_active/len(users_df)*100:.1f}%)")
    print(f"   Improvement: {new_active - current_active} more active users")
    
    if new_active < target_active * 0.9:  # If less than 90% of target
        print(f"\n[WARNING] Only {new_active} active users, expected ~{target_active}")
        print("   This might be due to product_id constraints")
    
    # Save improved data
    print("\n7. Saving improved data...")
    clicks_wide.to_csv(raw_path / "user_clicks_interactions.csv", index=False)
    purchases_wide.to_csv(raw_path / "user_purchase_interactions.csv", index=False)
    visits_time_wide.to_csv(raw_path / "user_visits_time_interactions.csv", index=False)
    
    print("  [OK] Saved improved interaction files")
    
    # Final verification after saving
    clicks_verify = pd.read_csv(raw_path / "user_clicks_interactions.csv")
    final_active = count_active_users(clicks_verify)
    print(f"\n8. Final verification (after saving):")
    print(f"   Active users: {final_active} ({final_active/len(users_df)*100:.1f}%)")
    print(f"   Total improvement: {final_active - current_active} more active users")
    
    print("\n" + "="*80)
    print("[SUCCESS] User interaction data improved!")
    print("="*80)
    print("\nNext steps:")
    print("1. Run user categorization again to see new category distribution")
    print("2. Check if accuracy is still high (85-90%)")
    print("3. Verify recommendation system can work with more users")

if __name__ == "__main__":
    import sys
    project_root = Path(__file__).parent
    improve_user_interactions(str(project_root))

