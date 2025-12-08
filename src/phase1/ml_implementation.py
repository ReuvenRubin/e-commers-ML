"""
E-Commerce Recommendation System - ML Algorithms Implementation
Phase 1: User Categorization Only
(Product categorization is done separately in Product_Categorization.py)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from pathlib import Path
# matplotlib and seaborn are optional - only needed for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None
import warnings
warnings.filterwarnings('ignore')

class MLImplementation:
    def __init__(self, data_path):
        """
        Initializes the MLImplementation class
        
        Parameters:
        - data_path: Path to the data directory containing datasets
        
        What it does:
        - Creates empty containers (None) for all data tables
        - Creates empty containers for results
        - These will be filled when load_data() is called
        """
        self.data_path = Path(data_path)
        self.products_df = None
        self.users_df = None
        self.clicks_df = None
        self.purchases_df = None
        self.visits_time_df = None
        self.product_metadata_df = None
        
        # Results
        self.user_clusters = None
        self.user_features = None
        self.rf_model = None
        self.feature_selector = None
        self.scaler = None
        
    def _convert_wide_to_long(self, df, value_name):
        """
        Converts wide format interaction table to long format
        
        Parameters:
        - df: DataFrame in wide format (uid, pid1, pid2, ..., pid10)
        - value_name: Name for the value column (e.g., 'clicks', 'purchases', 'visit_time')
        
        Returns:
        - DataFrame in long format (uid, product_id, value_name)
        """
        # Melt the dataframe: uid stays as identifier, pid columns become rows
        long_df = df.melt(
            id_vars=['uid'],
            value_vars=[col for col in df.columns if col.startswith('pid')],
            var_name='product_col',
            value_name=value_name
        )
        
        # Extract product_id from pid1, pid2, etc. (pid1 -> 1, pid2 -> 2, etc.)
        long_df['product_id'] = long_df['product_col'].str.replace('pid', '').astype(int)
        
        # Remove rows with zero values (no interaction)
        long_df = long_df[long_df[value_name] > 0]
        
        # Select and rename columns
        long_df = long_df[['uid', 'product_id', value_name]].copy()
        
        return long_df
        
    def load_data(self):
        """
        Loads all required data from CSV files
        
        What it loads:
        - products_10000.csv: All 10000 products
        - users_5000.csv: All 5000 users
        - user_clicks_interactions.csv: Click interactions (Wide format - converted to Long)
        - user_purchase_interactions.csv: Purchase interactions (Wide format - converted to Long)
        - user_visits_time_interactions.csv: Visit time interactions (Wide format - converted to Long)
        - product_interaction_metadata.csv: Product metadata
        
        Returns:
        - None (data is stored in self.products_df, self.users_df, etc.)
        """
        print("Loading data...")
        
        # Load original data from datasets/raw - use all 10000 products
        all_products = pd.read_csv(self.data_path / "datasets" / "raw" / "products_10000.csv")
        self.products_df = all_products.copy()
        # Load users (now 15000 users with realistic distribution: 70% inactive)
        self.users_df = pd.read_csv(self.data_path / "datasets" / "raw" / "users_5000.csv")
        
        # Load interaction tables in Wide format and convert to Long format
        clicks_wide = pd.read_csv(self.data_path / "datasets" / "raw" / "user_clicks_interactions.csv")
        purchases_wide = pd.read_csv(self.data_path / "datasets" / "raw" / "user_purchase_interactions.csv")
        visits_time_wide = pd.read_csv(self.data_path / "datasets" / "raw" / "user_visits_time_interactions.csv")
        
        # Convert from wide to long format
        self.clicks_df = self._convert_wide_to_long(clicks_wide, 'clicks')
        self.purchases_df = self._convert_wide_to_long(purchases_wide, 'purchases')
        self.visits_time_df = self._convert_wide_to_long(visits_time_wide, 'visit_time')
        
        # Load product metadata if it exists
        metadata_path = self.data_path / "datasets" / "raw" / "product_interaction_metadata.csv"
        if metadata_path.exists():
            self.product_metadata_df = pd.read_csv(metadata_path)
        else:
            self.product_metadata_df = None
        
        print(f"Loaded {len(self.products_df)} products and {len(self.users_df)} users")
        print(f"Loaded {len(self.clicks_df)} click interactions, {len(self.purchases_df)} purchase interactions, {len(self.visits_time_df)} visit time interactions")
        print(f"Will process all {len(self.users_df)} users (realistic distribution: ~70% inactive)")
    
    def prepare_user_features(self):
        """
        Prepares features for user categorization
        
        What it does:
        - Processes ALL users (15000 with realistic distribution), not just those with interactions
        - Uses Long format interaction tables for efficiency
        - Calculates comprehensive features per user
        - Normalizes all features using RobustScaler + StandardScaler
        
        Returns:
        - DataFrame with normalized features per user
        """
        print("\nPreparing user features...")
        
        user_features = []
        
        # Get unique product IDs (all products, up to 10000)
        product_ids = set(self.products_df['id'].tolist())
        num_products = len(product_ids)
        
        # Group interactions by user_id for faster access (Long format)
        clicks_by_user = self.clicks_df.groupby('uid')['clicks'].sum().to_dict()
        purchases_by_user = self.purchases_df.groupby('uid')['purchases'].sum().to_dict()
        visits_time_by_user = self.visits_time_df.groupby('uid')['visit_time'].sum().to_dict()
        
        # Get unique products per user and additional statistics
        unique_products_by_user = {}
        user_product_categories = {}
        user_avg_price = {}
        user_avg_views = {}
        user_category_counts = {}
        
        all_users_with_interactions = set(self.clicks_df['uid'].unique()) | set(self.purchases_df['uid'].unique())
        for user_id in all_users_with_interactions:
            user_products = set()
            user_clicks = self.clicks_df[self.clicks_df['uid'] == user_id]
            if not user_clicks.empty:
                user_products.update(user_clicks['product_id'].unique())
            user_purchases = self.purchases_df[self.purchases_df['uid'] == user_id]
            if not user_purchases.empty:
                user_products.update(user_purchases['product_id'].unique())
            unique_products_by_user[user_id] = len([p for p in user_products if p in product_ids])
            
            # Additional features: average price, views, categories
            prices = []
            views = []
            categories = set()
            for product_id in user_products:
                if product_id in product_ids:
                    product_row = self.products_df[self.products_df['id'] == product_id]
                    if len(product_row) > 0:
                        prices.append(product_row.iloc[0].get('price', 0))
                        views.append(product_row.iloc[0].get('views', 0))
                        if 'main_category' in product_row.columns:
                            categories.add(product_row.iloc[0]['main_category'])
                        elif 'category' in product_row.columns:
                            categories.add(product_row.iloc[0]['category'])
            
            user_avg_price[user_id] = np.mean(prices) if prices else 0
            user_avg_views[user_id] = np.mean(views) if views else 0
            user_category_counts[user_id] = len(categories)
            user_product_categories[user_id] = categories
        
        # Calculate time-based features
        user_registration_dates = {}
        if 'created_at' in self.users_df.columns:
            for _, user in self.users_df.iterrows():
                try:
                    reg_date = pd.to_datetime(user['created_at'])
                    user_registration_dates[user['id']] = reg_date
                except:
                    user_registration_dates[user['id']] = None
        
        days_since_registration = {}
        if user_registration_dates:
            current_date = pd.Timestamp.now()
            for user_id, reg_date in user_registration_dates.items():
                if reg_date is not None:
                    days_since_registration[user_id] = (current_date - reg_date).days
                else:
                    days_since_registration[user_id] = 0
        else:
            days_since_registration = {}
        
        # Get favorite category for each user
        user_favorite_category = {}
        for user_id in all_users_with_interactions:
            user_categories = user_product_categories.get(user_id, set())
            if user_categories:
                category_interactions = {}
                user_clicks = self.clicks_df[self.clicks_df['uid'] == user_id]
                user_purchases = self.purchases_df[self.purchases_df['uid'] == user_id]
                for product_id in user_clicks['product_id'].unique():
                    product_row = self.products_df[self.products_df['id'] == product_id]
                    if len(product_row) > 0:
                        cat = product_row.iloc[0].get('main_category') or product_row.iloc[0].get('category', '')
                        if cat:
                            category_interactions[cat] = category_interactions.get(cat, 0) + 1
                for product_id in user_purchases['product_id'].unique():
                    product_row = self.products_df[self.products_df['id'] == product_id]
                    if len(product_row) > 0:
                        cat = product_row.iloc[0].get('main_category') or product_row.iloc[0].get('category', '')
                        if cat:
                            category_interactions[cat] = category_interactions.get(cat, 0) + 2
                
                if category_interactions:
                    user_favorite_category[user_id] = max(category_interactions, key=category_interactions.get)
                else:
                    user_favorite_category[user_id] = list(user_categories)[0] if user_categories else ''
            else:
                user_favorite_category[user_id] = ''
        
        # Process ALL users from users_df
        for _, user in self.users_df.iterrows():
            user_id = user['id']
            
            # Get interaction data (Long format)
            total_clicks = clicks_by_user.get(user_id, 0)
            total_purchases = purchases_by_user.get(user_id, 0)
            total_visit_time = visits_time_by_user.get(user_id, 0)
            unique_products = unique_products_by_user.get(user_id, 0)
            
            # Conversion rate
            conversion_rate = total_purchases / total_clicks if total_clicks > 0 else 0
            
            # Category diversity
            category_diversity = unique_products / num_products if num_products > 0 else 0
            
            # Additional features
            avg_price = user_avg_price.get(user_id, 0)
            avg_views = user_avg_views.get(user_id, 0)
            num_categories = user_category_counts.get(user_id, 0)
            
            # Derived features
            clicks_per_product = total_clicks / unique_products if unique_products > 0 else 0
            purchases_per_product = total_purchases / unique_products if unique_products > 0 else 0
            visit_time_per_product = total_visit_time / unique_products if unique_products > 0 else 0
            
            # Behavioral features
            engagement_score = (total_clicks * 0.3) + (total_purchases * 5.0) + (total_visit_time * 0.1)
            activity_intensity = total_clicks / (unique_products + 1)
            purchase_frequency = total_purchases / (total_clicks + 1)
            time_efficiency = total_visit_time / (total_clicks + 1)
            
            # Advanced features
            interaction_consistency = 1.0 / (clicks_per_product + 1) if clicks_per_product > 0 else 0
            purchase_decision_speed = total_purchases / (total_clicks + 1)
            category_loyalty = 1.0 / (num_categories + 1) if num_categories > 0 else 0
            price_sensitivity = 1.0 / (avg_price + 1) if avg_price > 0 else 0
            engagement_depth = total_visit_time / (total_purchases + 1) if total_purchases > 0 else total_visit_time
            product_exploration = unique_products / (total_clicks + 1)
            return_rate_estimate = (total_clicks - total_purchases) / (total_clicks + 1) if total_clicks > 0 else 0
            
            # Additional advanced features
            purchase_intensity = total_purchases / (unique_products + 1)
            click_purchase_ratio = total_clicks / (total_purchases + 1)
            time_per_click = total_visit_time / (total_clicks + 1)
            purchase_velocity = total_purchases / (total_visit_time + 1)
            category_concentration = 1.0 / (num_categories + 1) if num_categories > 0 else 0
            price_preference_strength = 1.0 / (avg_price + 1) if avg_price > 0 else 0
            interaction_frequency = (total_clicks + total_purchases) / (unique_products + 1)
            engagement_consistency = 1.0 / (abs(total_clicks - total_purchases) + 1)
            value_per_interaction = avg_price * total_purchases / (total_clicks + 1)
            exploration_ratio = unique_products / (total_clicks + total_purchases + 1)
            
            # Time-based features
            days_since_reg = days_since_registration.get(user_id, 0)
            
            # Favorite category (encoded as numeric hash for clustering)
            favorite_cat = user_favorite_category.get(user_id, '')
            favorite_category_hash = hash(str(favorite_cat)) % 1000 if favorite_cat else 0
            
            user_features.append({
                'user_id': user_id,
                'total_clicks': total_clicks,
                'total_purchases': total_purchases,
                'total_visit_time': total_visit_time,
                'unique_products': unique_products,
                'conversion_rate': conversion_rate,
                'category_diversity': category_diversity,
                'avg_price': avg_price,
                'avg_views': avg_views,
                'num_categories': num_categories,
                'clicks_per_product': clicks_per_product,
                'purchases_per_product': purchases_per_product,
                'visit_time_per_product': visit_time_per_product,
                'engagement_score': engagement_score,
                'activity_intensity': activity_intensity,
                'purchase_frequency': purchase_frequency,
                'time_efficiency': time_efficiency,
                'interaction_consistency': interaction_consistency,
                'purchase_decision_speed': purchase_decision_speed,
                'category_loyalty': category_loyalty,
                'price_sensitivity': price_sensitivity,
                'engagement_depth': engagement_depth,
                'product_exploration': product_exploration,
                'return_rate_estimate': return_rate_estimate,
                'purchase_intensity': purchase_intensity,
                'click_purchase_ratio': click_purchase_ratio,
                'time_per_click': time_per_click,
                'purchase_velocity': purchase_velocity,
                'category_concentration': category_concentration,
                'price_preference_strength': price_preference_strength,
                'interaction_frequency': interaction_frequency,
                'engagement_consistency': engagement_consistency,
                'value_per_interaction': value_per_interaction,
                'exploration_ratio': exploration_ratio,
                'days_since_registration': days_since_reg,
                'favorite_category_hash': favorite_category_hash
            })
        
        self.user_features = pd.DataFrame(user_features)
        
        # Store original values before normalization
        feature_columns = ['total_clicks', 'total_purchases', 'total_visit_time', 
                          'unique_products', 'conversion_rate', 'category_diversity',
                          'avg_price', 'avg_views', 'num_categories',
                          'clicks_per_product', 'purchases_per_product', 'visit_time_per_product',
                          'engagement_score', 'activity_intensity', 'purchase_frequency', 'time_efficiency',
                          'interaction_consistency', 'purchase_decision_speed', 'category_loyalty',
                          'price_sensitivity', 'engagement_depth', 'product_exploration', 'return_rate_estimate',
                          'purchase_intensity', 'click_purchase_ratio', 'time_per_click', 'purchase_velocity',
                          'category_concentration', 'price_preference_strength', 'interaction_frequency',
                          'engagement_consistency', 'value_per_interaction', 'exploration_ratio',
                          'days_since_registration', 'favorite_category_hash']
        self.user_features_original = self.user_features[feature_columns].copy()
        
        # Filter out users with zero interactions
        active_mask = (self.user_features['total_clicks'] > 0) | (self.user_features['total_purchases'] > 0) | (self.user_features['total_visit_time'] > 0)
        self.user_features['is_active'] = active_mask
        inactive_count = (~active_mask).sum()
        
        if inactive_count > 0:
            print(f"Identified {inactive_count} inactive users ({inactive_count/len(self.user_features)*100:.1f}%) - will be categorized separately")
        
        # Use RobustScaler for better handling of outliers, then StandardScaler for normalization
        robust_scaler = RobustScaler()
        standard_scaler = StandardScaler()
        self.user_features[feature_columns] = robust_scaler.fit_transform(self.user_features[feature_columns])
        self.user_features[feature_columns] = standard_scaler.fit_transform(self.user_features[feature_columns])
        self.scaler_robust = robust_scaler
        self.scaler_standard = standard_scaler
        
        print(f"Created {len(self.user_features)} users with {len(feature_columns)} features")
        return self.user_features
    
    def _create_user_categories(self, user_features_df):
        """
        Creates user categories based on their behavior patterns
        This creates the target variable (y) for supervised learning
        
        Categories:
        - 'high_value': High activity, high purchases, high engagement
        - 'active_browser': High clicks, low purchases (browsers)
        - 'occasional_buyer': Medium activity, occasional purchases
        - 'price_sensitive': Low price preference, selective purchases
        - 'category_loyal': Focused on specific categories
        - 'explorer': High product diversity, explores many products
        - 'inactive': Very low or no activity
        """
        categories = []
        
        for _, user in user_features_df.iterrows():
            total_clicks = user['total_clicks']
            total_purchases = user['total_purchases']
            total_visit_time = user['total_visit_time']
            unique_products = user['unique_products']
            conversion_rate = user['conversion_rate']
            category_diversity = user['category_diversity']
            
            # Calculate activity score
            activity_score = (total_clicks * 0.3 + total_purchases * 5.0 + total_visit_time * 0.1)
            
            # Determine category based on behavior patterns
            if activity_score == 0 or (total_clicks == 0 and total_purchases == 0):
                category = 'inactive'
            elif total_purchases >= 10 and conversion_rate >= 0.3:
                category = 'high_value'
            elif total_clicks >= 50 and total_purchases < 3:
                category = 'active_browser'
            elif total_purchases >= 3 and total_purchases < 10:
                category = 'occasional_buyer'
            elif category_diversity < 0.1 and unique_products < 10:
                category = 'category_loyal'
            elif unique_products >= 20 and category_diversity >= 0.3:
                category = 'explorer'
            elif activity_score < 10:
                category = 'inactive'
            else:
                category = 'occasional_buyer'
            
            categories.append(category)
        
        return np.array(categories)
    
    def user_categorization_random_forest(self):
        """
        Categorizes users using Random Forest Classifier with proper preprocessing
        
        What it does:
        1. Prepares user features (if not already done)
        2. Creates user categories based on behavior (target variable)
        3. Proper encoding and scaling of features
        4. Feature selection to keep most relevant features
        5. Train/test split with stratification
        6. Hyperparameter tuning with GridSearchCV
        7. Cross-validation for stable accuracy
        8. Final evaluation with accuracy, precision, recall, F1
        
        Returns:
        - final_labels: Array of category assignments for each user
        - accuracy: Classification accuracy
        """
        print("\n" + "="*60)
        print("User Categorization - Random Forest Classifier")
        print("="*60)
        
        # Prepare features
        if self.user_features is None:
            self.prepare_user_features()
        
        # Step 1: Create target variable (user categories)
        print("\nStep 1: Creating user categories based on behavior patterns...")
        feature_columns = ['total_clicks', 'total_purchases', 'total_visit_time', 
                          'unique_products', 'conversion_rate', 'category_diversity',
                          'avg_price', 'avg_views', 'num_categories',
                          'clicks_per_product', 'purchases_per_product', 'visit_time_per_product',
                          'engagement_score', 'activity_intensity', 'purchase_frequency', 'time_efficiency',
                          'interaction_consistency', 'purchase_decision_speed', 'category_loyalty',
                          'price_sensitivity', 'engagement_depth', 'product_exploration', 'return_rate_estimate',
                          'purchase_intensity', 'click_purchase_ratio', 'time_per_click', 'purchase_velocity',
                          'category_concentration', 'price_preference_strength', 'interaction_frequency',
                          'engagement_consistency', 'value_per_interaction', 'exploration_ratio',
                          'days_since_registration', 'favorite_category_hash']
        
        # Use original (non-normalized) features for category creation
        X_original = self.user_features_original[feature_columns].copy()
        y = self._create_user_categories(X_original)
        
        # Encode target variable
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        self.label_encoder = label_encoder
        
        print(f"Created {len(np.unique(y))} user categories:")
        total_percentage = 0.0
        for i, cat in enumerate(label_encoder.classes_):
            count = np.sum(y == cat)
            percentage = count/len(y)*100
            total_percentage += percentage
            print(f"  {cat}: {count} users ({percentage:.1f}%)")
        
        # Verify percentages sum to 100%
        print(f"\nTotal: {len(y)} users ({total_percentage:.1f}%)")
        if abs(total_percentage - 100.0) > 0.1:  # Allow small floating point error
            print(f"  ⚠️ Warning: Percentages sum to {total_percentage:.1f}% (expected 100.0%)")
        
        # Step 2: Prepare features (use normalized features)
        print("\nStep 2: Preparing features with proper preprocessing...")
        X = self.user_features[feature_columns].copy()
        
        # Convert to numpy array
        if hasattr(X, 'values'):
            X = X.values
        else:
            X = np.array(X)
        
        # Step 3: Feature selection
        print("\nStep 3: Feature selection to keep most relevant features...")
        # Use mutual information for feature selection (works well with Random Forest)
        selector = SelectKBest(score_func=mutual_info_classif, k='all')
        selector.fit(X, y_encoded)
        
        # Get feature scores and select top features
        feature_scores = selector.scores_
        # Select top 20 features (or all if less than 20)
        n_features_to_select = min(20, len(feature_columns))
        top_feature_indices = np.argsort(feature_scores)[-n_features_to_select:]
        
        X_selected = X[:, top_feature_indices]
        selected_feature_names = [feature_columns[i] for i in top_feature_indices]
        
        print(f"Selected {len(selected_feature_names)} most relevant features:")
        for i, idx in enumerate(top_feature_indices):
            print(f"  {feature_columns[idx]}: score = {feature_scores[idx]:.4f}")
        
        # Store feature selector
        self.feature_selector = selector
        self.selected_feature_indices = top_feature_indices
        self.selected_feature_names = selected_feature_names
        
        # Step 4: Train/Test split with stratification
        print("\nStep 4: Creating stratified train/test split...")
        # Check if we can use stratify (need at least 2 samples per class)
        unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
        min_class_count = class_counts.min()
        
        if min_class_count < 2:
            print(f"  Warning: Some classes have less than 2 samples (min: {min_class_count})")
            print(f"  Cannot use stratify. Using regular split instead.")
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_encoded,
                test_size=0.2,
                random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_encoded,
                test_size=0.2,
                random_state=42,
                stratify=y_encoded  # Maintain category distribution
            )
        
        print(f"Training set: {len(X_train)} users")
        print(f"Test set: {len(X_test)} users")
        print(f"Training set category distribution:")
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        for cat_idx, count in zip(unique_train, counts_train):
            print(f"  {label_encoder.classes_[cat_idx]}: {count} users")
        
        # Step 5: Hyperparameter tuning with GridSearchCV (FAST VERSION)
        print("\nStep 5: Hyperparameter tuning with GridSearchCV...")
        print("  Using FAST optimized grid (target: 5-10 minutes)...")
        
        # FAST VERSION: Minimal parameter grid for very fast execution while maintaining quality
        # Strategy: Focus on most impactful parameters, use proven good values
        # 2*2*2*2*2 = 32 combinations -> 96 fits (with 3-fold CV)
        # Time: ~5-10 minutes
        param_grid = {
            'n_estimators': [150, 200],  # Focus on proven good range
            'max_depth': [20, None],  # None (unlimited) often works best, 20 is good middle ground
            'min_samples_split': [2, 5],  # 2 is default (best), 5 prevents overfitting
            'min_samples_leaf': [1, 2],  # 1 is default (best), 2 prevents overfitting
            'max_features': ['sqrt', 'log2']  # Both are proven good, 'sqrt' is default
        }
        
        # Use StratifiedKFold for cross-validation (3 folds for speed)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Create base Random Forest
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        
        # GridSearchCV with cross-validation
        grid_search = GridSearchCV(
            rf_base,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,  # Use all CPU cores
            verbose=1
        )
        
        total_combinations = (len(param_grid['n_estimators']) * len(param_grid['max_depth']) * 
                             len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * 
                             len(param_grid['max_features']))
        total_fits = total_combinations * cv.n_splits
        
        print(f"  Testing {total_combinations} parameter combinations...")
        print(f"  With {cv.n_splits}-fold CV: ~{total_fits} model fits")
        print(f"  Estimated time: 5-10 minutes (depending on CPU)")
        
        grid_search.fit(X_train, y_train)
        
        print(f"\n  Best parameters: {grid_search.best_params_}")
        print(f"  Best cross-validation score: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")
        
        # Step 6: Train final model with best parameters
        print("\nStep 6: Training final model with best parameters...")
        self.rf_model = grid_search.best_estimator_
        
        # Step 7: Cross-validation on full training set (quick validation)
        print("\nStep 7: Quick cross-validation for stable accuracy estimation...")
        cv_quick = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Quick 3-fold CV
        cv_scores = cross_val_score(self.rf_model, X_train, y_train, cv=cv_quick, scoring='accuracy')
        print(f"  Cross-validation scores: {cv_scores}")
        print(f"  Mean CV accuracy: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
        print(f"  Std CV accuracy: {cv_scores.std():.4f} ({cv_scores.std()*100:.2f}%)")
        
        # Step 8: Final evaluation on test set
        print("\nStep 8: Final evaluation on test set...")
        y_pred = self.rf_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\nFinal Metrics:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1 Score: {f1:.4f} ({f1*100:.2f}%)")
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        # Get unique classes in y_test and y_pred
        unique_test_classes = np.unique(y_test)
        unique_pred_classes = np.unique(y_pred)
        all_classes = np.unique(np.concatenate([unique_test_classes, unique_pred_classes]))
        
        # Filter target_names to only include classes that appear in test/pred
        available_target_names = [label_encoder.classes_[i] for i in all_classes if i < len(label_encoder.classes_)]
        print(classification_report(y_test, y_pred, labels=all_classes, target_names=available_target_names, zero_division=0))
        
        # Predict on all users
        print("\nStep 9: Predicting categories for all users...")
        y_pred_all = self.rf_model.predict(X_selected)
        y_pred_all_labels = label_encoder.inverse_transform(y_pred_all)
        
        # Add predictions to user_features
        self.user_features['cluster'] = y_pred_all
        self.user_features['category'] = y_pred_all_labels
        
        # Store results
        self.user_clusters = y_pred_all
        
        # Feature importance
        print("\nTop 10 Most Important Features:")
        feature_importance = pd.DataFrame({
            'feature': selected_feature_names,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(feature_importance.head(10).to_string(index=False))
        
        return y_pred_all, accuracy
    
    def save_results(self):
        """
        Saves user categorization results to CSV files
        
        What it saves:
        - users_with_clusters.csv: User features DataFrame with category labels
        - clustering_summary.csv: Summary report with algorithm info and metrics
        
        Returns:
        - output_path: Path to the ml_results directory where files were saved
        """
        print("\nSaving results...")
        
        output_path = self.data_path / "datasets" / "results"
        output_path.mkdir(exist_ok=True)
        
        # Save users with clusters
        self.user_features.to_csv(output_path / "users_with_clusters.csv", index=False)
        
        # Create summary report
        summary = {
            'user_categories': {
                'algorithm': 'Random Forest Classifier',
                'n_categories': len(set(self.user_clusters)) if self.user_clusters is not None else 0,
                'n_users': len(self.user_features) if self.user_features is not None else 0
            }
        }
        
        summary_df = pd.DataFrame(summary).T
        summary_df.to_csv(output_path / "clustering_summary.csv")
        
        print(f"Results saved to: {output_path}")
        return output_path
    
    def run_phase1(self):
        """
        Runs the complete Phase 1 pipeline: User Categorization Only
        
        What it does (in order):
        1. Loads all data (products, users, interactions)
        2. Categorizes users into categories using Random Forest
        3. Saves results to CSV files
        
        Note: Product categorization is done separately in Product_Categorization.py
        
        Returns:
        - Dictionary containing:
          * user_clusters: Array of user category assignments
          * user_silhouette: Classification accuracy
          * output_path: Path where results were saved
        """
        print("="*80)
        print("Phase 1: User Categorization Only")
        print("(Product categorization is done separately in Product_Categorization.py)")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # User categorization with Random Forest
        user_labels, accuracy = self.user_categorization_random_forest()
        
        # Save results
        output_path = self.save_results()
        
        print(f"\n" + "="*80)
        print("Phase 1 completed successfully!")
        print("="*80)
        print(f"Users: {len(set(user_labels))} categories, Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Results saved to: {output_path}")
        print("="*80)
        
        return {
            'user_clusters': user_labels,
            'user_silhouette': accuracy,  # Using accuracy instead of silhouette for classification
            'output_path': output_path
        }

if __name__ == "__main__":
    import os
    # Get the project root directory (parent of src)
    project_root = Path(__file__).parent.parent.parent
    ml = MLImplementation(str(project_root))
    results = ml.run_phase1()
