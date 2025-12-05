"""
Phase 2: מערכת המלצות היברידית
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Neural Network imports (for ranking)
try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
    NEURAL_NETWORK_AVAILABLE = True
except (ImportError, Exception) as e:
    # מטפל גם ב-ImportError וגם בשגיאות אחרות (כמו DLL errors)
    NEURAL_NETWORK_AVAILABLE = False
    # לא מדפיסים את ההודעה כאן כדי לא להפריע - נבדוק בקוד

class RecommendationSystem:
    def __init__(self, data_path):
        """
        Initializes the RecommendationSystem class
        
        Parameters:
        - data_path: Path to the data directory containing datasets
        
        What it does:
        - Creates empty containers for all data tables
        - Creates empty containers for models (TF-IDF, similarity matrix, neural network)
        - Initializes Continuous Learning tracking (new_interactions, retrain_threshold)
        - Initializes Dynamic Updates mappings (user_id_to_index, product_id_to_index)
        """
        self.data_path = Path(data_path)
        
        # נתונים
        self.products_df = None
        self.users_df = None
        self.clicks_df = None
        self.purchases_df = None
        self.visits_time_df = None
        self.product_metadata_df = None
        
        # תוצאות קטגוריזציה
        self.products_with_clusters = None
        self.users_with_clusters = None
        
        # מודלים
        self.tfidf_vectorizer = None
        self.product_tfidf_matrix = None
        self.user_similarity_matrix = None
        self.neural_ranking_model = None  # Neural Network for ranking
        self.feature_scaler = None  # Scaler for normalizing features
        
        # Continuous Learning - מעקב אחר אינטראקציות חדשות
        self.new_interactions = []  # רשימת אינטראקציות חדשות (לפני אימון מחדש)
        self.new_interactions_count = 0  # מונה של אינטראקציות חדשות
        self.retrain_threshold = 100  # מתי לאמן מחדש (כל 100 אינטראקציות)
        
        # Dynamic Updates - מיפויים לעדכון מהיר
        self.user_id_to_index = {}  # מיפוי user_id -> index במטריצה
        self.product_id_to_index = {}  # מיפוי product_id -> index במטריצה
        self.all_user_ids = []  # רשימת כל user_ids במטריצה
        self.all_product_ids = []  # רשימת כל product_ids במטריצה
        
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
    
    def _categorize_products_using_logistic_regression(self):
        """
        Categorizes products using Logistic Regression (from Product_Categorization.py logic)
        
        This function implements the same logic as Product_Categorization.py:
        1. Combines product_name and description
        2. Creates TF-IDF features (1000 features, ngrams 1-3)
        3. Scales price feature
        4. Trains Logistic Regression model
        5. Predicts categories for all products
        
        Returns:
        - DataFrame with products and their predicted categories (ml_cluster column)
        """
        print("  Categorizing products using Logistic Regression (Product_Categorization.py logic)...")
        
        # Copy products data
        products = self.products_df.copy()
        
        # Fill missing text fields
        products['product_name'] = products['product_name'].fillna('')
        products['description'] = products['description'].fillna('')
        
        # Remove completely empty products
        initial_size = len(products)
        products = products[~((products['product_name'].str.strip() == '') & 
                            (products['description'].str.strip() == ''))]
        if len(products) < initial_size:
            print(f"  Removed {initial_size - len(products)} empty products")
        
        # Combine text features (like Product_Categorization.py)
        products['combined_text'] = products['product_name'] + ' ' + products['description']
        
        # Separate features and target
        X_text = products['combined_text']
        X_price = products['price'].values.reshape(-1, 1)
        y = products['main_category']
        
        # TF-IDF (like Product_Categorization.py)
        tfidf = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.90,
            stop_words='english'
        )
        X_text_tfidf = tfidf.fit_transform(X_text).toarray()
        
        # Scale price feature
        price_scaler = StandardScaler()
        X_price_scaled = price_scaler.fit_transform(X_price)
        
        # Combine features
        X_combined = np.hstack([X_text_tfidf, X_price_scaled])
        
        # Split data (like Product_Categorization.py)
        # Check if we can use stratify
        category_counts = y.value_counts()
        min_samples = category_counts.min()
        use_stratify = min_samples >= 2
        
        if use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y,
                test_size=0.20,
                random_state=21,
                stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y,
                test_size=0.20,
                random_state=21
            )
        
        # Train Logistic Regression model (like Product_Categorization.py)
        model = LogisticRegression(
            max_iter=1000,
            multi_class='auto',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_test = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_test)
        print(f"  Product categorization accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Predict categories for all products
        y_pred_all = model.predict(X_combined)
        
        # Add predicted categories to products DataFrame
        products['ml_cluster'] = y_pred_all
        products['predicted_category'] = y_pred_all
        
        print(f"  Categorized {len(products)} products into {len(set(y_pred_all))} categories")
        
        return products
    
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
        
        What it does:
        - Runs product categorization using Product_Categorization.py logic (Logistic Regression)
        - Loads user clustering results from Phase 1 (users_with_clusters.csv)
        
        Note:
        - Product categorization is done here (not loaded from Phase 1)
        - Uses the same logic as Product_Categorization.py
        
        Returns:
        - None (data is stored in self.products_df, self.users_df, etc.)
        """
        print("Loading data for recommendation system...")
        
        # נתונים מקוריים מ-ML_Ofir/datasets - כל 10000 המוצרים
        products_all = pd.read_csv(self.data_path / "ML_Ofir" / "datasets" / "products_10000.csv")
        self.products_df = products_all.copy()
        self.users_df = pd.read_csv(self.data_path / "ML_Ofir" / "datasets" / "users_5000.csv")
        
        # טבלאות אינטראקציות - Wide format, נמיר ל-Long format
        clicks_wide = pd.read_csv(self.data_path / "ML_Ofir" / "datasets" / "user_clicks_interactions.csv")
        purchases_wide = pd.read_csv(self.data_path / "ML_Ofir" / "datasets" / "user_purchase_interactions.csv")
        visits_time_wide = pd.read_csv(self.data_path / "ML_Ofir" / "datasets" / "user_visits_time_interactions.csv")
        
        # המרה מ-wide ל-long format
        self.clicks_df = self._convert_wide_to_long(clicks_wide, 'clicks')
        self.purchases_df = self._convert_wide_to_long(purchases_wide, 'purchases')
        self.visits_time_df = self._convert_wide_to_long(visits_time_wide, 'visit_time')
        
        # Load product metadata if it exists
        metadata_path = self.data_path / "ML_Ofir" / "datasets" / "product_interaction_metadata.csv"
        if metadata_path.exists():
            self.product_metadata_df = pd.read_csv(metadata_path)
        else:
            self.product_metadata_df = None
        
        # תוצאות קטגוריזציה - משתמשים בקטגור המוצרים מ-Product_Categorization.py
        # במקום לטעון מ-Phase 1, מריצים את קטגור המוצרים בעצמנו
        print("\nRunning product categorization (from Product_Categorization.py logic)...")
        self.products_with_clusters = self._categorize_products_using_logistic_regression()
        
        # משתמשים עדיין בקטגור המשתמשים מ-Phase 1 (או אפשר גם לרוץ כאן)
        users_clusters_path = self.data_path / "datasets/ml_results/users_with_clusters.csv"
        if users_clusters_path.exists():
            self.users_with_clusters = pd.read_csv(users_clusters_path)
        else:
            print("Warning: users_with_clusters.csv not found. User clustering will be unavailable.")
            self.users_with_clusters = None
        
        print("Data loaded successfully!")
        
    def prepare_tfidf_for_products(self):
        """
        Prepares TF-IDF vectors for product descriptions
        
        What it does:
        - Converts product descriptions to TF-IDF vectors
        - Uses 100 most important words (max_features=100)
        - Removes common English stop words
        
        Returns:
        - None (TF-IDF matrix stored in self.product_tfidf_matrix)
        """
        print("Preparing TF-IDF for product descriptions...")
        
        # TF-IDF על תיאורי מוצרים
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.product_tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.products_df['description'].fillna('')
        )
        
        print(f"Created TF-IDF matrix: {self.product_tfidf_matrix.shape}")
        
    def create_user_interaction_matrix(self):
        """
        Creates a weighted user-product interaction matrix
        
        What it does:
        - Identifies all unique users and products from interaction tables
        - Creates mappings (user_id_to_index, product_id_to_index) for dynamic updates
        - Builds weighted interaction matrix:
          * Clicks: weight 1.0
          * Purchases: weight 5.0 (more important)
          * Visit time: weight 0.1 (less important)
        - Converts to DataFrame with meaningful row/column labels
        
        Returns:
        - None (matrix stored in self.interaction_matrix)
        """
        print("Creating interaction matrix...")
        
        # שימוש בכל 5000 המשתמשים (לא רק אלה עם אינטראקציות)
        # זה מאפשר לתת המלצות גם למשתמשים חדשים (cold start)
        all_user_ids = sorted(self.users_df['id'].tolist())  # כל 5000 המשתמשים
        
        # זיהוי כל המוצרים הייחודיים מהאינטראקציות
        all_product_ids = set(self.clicks_df['product_id'].unique()) | set(self.purchases_df['product_id'].unique()) | set(self.visits_time_df['product_id'].unique())
        all_product_ids = sorted(list(all_product_ids))
        
        num_users = len(all_user_ids)
        num_products = len(all_product_ids)
        
        # שמירת המיפויים לשימוש בעדכונים דינמיים
        self.all_user_ids = all_user_ids
        self.all_product_ids = all_product_ids
        self.user_id_to_index = {uid: idx for idx, uid in enumerate(all_user_ids)}
        
        # יצירת מיפוי product_id -> index
        product_id_to_index = {pid: idx for idx, pid in enumerate(all_product_ids)}
        self.product_id_to_index = product_id_to_index
        
        # יצירת מטריצת אינטראקציות משוקללת
        interaction_matrix = np.zeros((num_users, num_products))
        
        # מילוי מטריצה מקליקים
        for _, row in self.clicks_df.iterrows():
            user_idx = all_user_ids.index(row['uid'])
            product_idx = product_id_to_index[row['product_id']]
            interaction_matrix[user_idx, product_idx] += row['clicks'] * 1.0
        
        # מילוי מטריצה מרכישות
        for _, row in self.purchases_df.iterrows():
            user_idx = all_user_ids.index(row['uid'])
            product_idx = product_id_to_index[row['product_id']]
            interaction_matrix[user_idx, product_idx] += row['purchases'] * 5.0
        
        # מילוי מטריצה מזמן ביקור
        for _, row in self.visits_time_df.iterrows():
            user_idx = all_user_ids.index(row['uid'])
            product_idx = product_id_to_index[row['product_id']]
            interaction_matrix[user_idx, product_idx] += row['visit_time'] * 0.1
        
        # המרה ל-DataFrame
        self.interaction_matrix = pd.DataFrame(
            interaction_matrix, 
            index=all_user_ids,
            columns=[f'product_{pid}' for pid in all_product_ids]
        )
        
        print(f"Created interaction matrix: {self.interaction_matrix.shape}")
        
    def calculate_user_similarity(self):
        """
        Calculates cosine similarity between users based on their interactions
        
        What it does:
        - Normalizes the interaction matrix using StandardScaler
        - Calculates cosine similarity between all user pairs
        - Stores similarity matrix for collaborative filtering
        
        Returns:
        - None (similarity matrix stored in self.user_similarity_matrix)
        """
        print("Calculating user similarity...")
        
        # נרמול המטריצה
        scaler = StandardScaler()
        normalized_matrix = scaler.fit_transform(self.interaction_matrix)
        
        # חישוב דמיון קוסינוס
        self.user_similarity_matrix = cosine_similarity(normalized_matrix)
        
        print(f"Created user similarity matrix: {self.user_similarity_matrix.shape}")
    
    def update_interaction_dynamic(self, user_id, product_id, interaction_type='click', value=1):
        """
        Dynamically updates the interaction matrix for a specific user-product interaction
        
        What it does:
        - Updates the interaction matrix in real-time
        - Applies weights: clicks (1.0), purchases (5.0), visit_time (0.1)
        - Tracks new interactions for Continuous Learning
        - Increments new_interactions_count
        
        Why is this important?
        - Allows real-time updates without full matrix recalculation
        - Enables Dynamic Updates feature
        - Tracks interactions for Continuous Learning retraining
        
        Parameters:
        - user_id: User ID
        - product_id: Product ID
        - interaction_type: 'click', 'purchase', or 'visit_time'
        - value: Interaction value (number of clicks, purchases, or visit time)
        
        Returns:
        - bool: True if update succeeded, False if user/product not found in matrix
        """
        # בדיקה אם המשתמש והמוצר קיימים במטריצה
        if user_id not in self.user_id_to_index:
            print(f"Warning: User {user_id} not found in interaction matrix")
            return False
        
        if product_id not in self.product_id_to_index:
            print(f"Warning: Product {product_id} not found in interaction matrix")
            return False
        
        # מציאת האינדקסים במטריצה
        user_idx = self.user_id_to_index[user_id]
        product_idx = self.product_id_to_index[product_id]
        
        # משקלים לפי סוג האינטראקציה (כמו ב-create_user_interaction_matrix)
        weights = {
            'click': 1.0,
            'purchase': 5.0,
            'visit_time': 0.1
        }
        
        if interaction_type not in weights:
            print(f"Warning: Unknown interaction type '{interaction_type}'. Using 'click' weight.")
            weight = weights['click']
        else:
            weight = weights[interaction_type]
        
        # עדכון המטריצה
        column_name = f'product_{product_id}'
        if column_name in self.interaction_matrix.columns:
            # עדכון הערך במטריצה (שימוש ב-loc לעדכון ישיר)
            current_value = self.interaction_matrix.loc[self.all_user_ids[user_idx], column_name]
            new_value = current_value + (value * weight)
            self.interaction_matrix.loc[self.all_user_ids[user_idx], column_name] = new_value
            
            print(f"Updated interaction: User {user_id} - Product {product_id} ({interaction_type}): {current_value:.2f} -> {new_value:.2f}")
            
            # Continuous Learning: שמירת אינטראקציה חדשה
            # שומרים את האינטראקציה החדשה ברשימה כדי לאמן מחדש את הרשת מאוחר יותר
            self.new_interactions.append({
                'user_id': user_id,
                'product_id': product_id,
                'interaction_type': interaction_type,
                'value': value,
                'weighted_value': value * weight
            })
            
            # עדכון המונה
            self.new_interactions_count += 1
            
            return True
        else:
            print(f"Warning: Column '{column_name}' not found in interaction matrix")
            return False
    
    def recalculate_user_similarity(self, force_full_recalc=False):
        """
        Recalculates user similarity matrix after interactions have been updated
        
        Why is this important?
        - After updating the interaction matrix (with update_interaction_dynamic),
          user similarities change. To keep recommendations accurate,
          we need to recalculate similarities.
        
        What it does:
        - Normalizes the updated interaction matrix
        - Recalculates cosine similarity between all user pairs
        - Updates self.user_similarity_matrix
        
        Parameters:
        - force_full_recalc: bool (currently not used, kept for future use)
        
        Returns:
        - None (updates self.user_similarity_matrix)
        """
        if self.interaction_matrix is None:
            print("Error: Interaction matrix not created yet. Call create_user_interaction_matrix() first.")
            return
        
        print("Recalculating user similarity matrix...")
        
        # נרמול המטריצה (כמו ב-calculate_user_similarity)
        scaler = StandardScaler()
        normalized_matrix = scaler.fit_transform(self.interaction_matrix)
        
        # חישוב דמיון קוסינוס מחדש
        self.user_similarity_matrix = cosine_similarity(normalized_matrix)
        
        print(f"User similarity matrix recalculated: {self.user_similarity_matrix.shape}")
        print("Note: Recommendations will now use the updated similarity matrix.")
    
    def update_and_recommend(self, user_id, product_id, interaction_type='click', value=1, 
                             recalculate_similarity=True, n_recommendations=5):
        """
        Convenient function that updates an interaction and returns updated recommendations
        
        What it does (in order):
        1. Updates the interaction matrix (update_interaction_dynamic)
        2. Recalculates user similarity (recalculate_user_similarity) - optional
        3. Checks for Continuous Learning retraining (if threshold met)
        4. Returns updated recommendations (hybrid_recommendations)
        
        Why is this useful?
        - Instead of calling 3 functions separately, call one that does everything
        - Especially useful when a user interacts with a product and we want new recommendations immediately
        
        Parameters:
        - user_id: User ID
        - product_id: Product ID that user interacted with
        - interaction_type: 'click', 'purchase', or 'visit_time'
        - value: Interaction value (number of clicks, purchases, or visit time)
        - recalculate_similarity: If True, recalculates user similarity (recommended)
        - n_recommendations: Number of recommendations to return
        
        Returns:
        - List of recommended product_ids (updated based on new interaction)
        """
        print(f"\n{'='*60}")
        print(f"Dynamic Update & Recommendation for User {user_id}")
        print(f"{'='*60}")
        
        # שלב 1: עדכון אינטראקציה
        print(f"\nStep 1: Updating interaction...")
        update_success = self.update_interaction_dynamic(
            user_id=user_id, 
            product_id=product_id, 
            interaction_type=interaction_type, 
            value=value
        )
        
        if not update_success:
            print("Warning: Failed to update interaction. Returning recommendations based on current data.")
            return self.hybrid_recommendations(user_id, n_recommendations)
        
        # שלב 2: חישוב מחדש של דמיון (אופציונלי)
        if recalculate_similarity:
            print(f"\nStep 2: Recalculating user similarity...")
            self.recalculate_user_similarity()
        else:
            print(f"\nStep 2: Skipping similarity recalculation (for faster response)")
        
        # שלב 2.5: Continuous Learning - בדיקה אם צריך לאמן מחדש
        # בודקים אם יש מספיק אינטראקציות חדשות (100+)
        if self.new_interactions_count >= self.retrain_threshold:
            print(f"\nStep 2.5: Continuous Learning - Checking for retraining...")
            print(f"   Found {self.new_interactions_count} new interactions (threshold: {self.retrain_threshold})")
            retrained = self.check_and_retrain_neural_network()
            if retrained:
                print(f"   Neural network retrained with new data!")
            else:
                print(f"   Retraining skipped or failed.")
        else:
            print(f"\nStep 2.5: Continuous Learning - Not enough new interactions yet")
            print(f"   Current: {self.new_interactions_count}/{self.retrain_threshold} interactions")
        
        # שלב 3: המלצות מעודכנות
        print(f"\nStep 3: Getting updated recommendations...")
        recommendations = self.hybrid_recommendations(user_id, n_recommendations)
        
        print(f"\n{'='*60}")
        print(f"Updated Recommendations for User {user_id}: {recommendations}")
        print(f"{'='*60}\n")
        
        return recommendations
    
    def prepare_neural_network_features(self, user_ids=None, product_ids=None, sample_size=10000):
        """
        Prepares features for Neural Network ranking model
        
        What it does:
        - Creates combined features from users and products:
          * User features: cluster, total interactions, number of products interacted with
          * Product features: cluster, price, category
        - Creates labels: 1 if user interacted with product, 0 otherwise
        - Returns features and labels for training
        
        Features (6 total):
        1. user_cluster: User's cluster ID
        2. product_cluster: Product's cluster ID
        3. product_price: Product price
        4. product_category: Product category (encoded as number)
        5. total_interactions: User's total interactions
        6. num_products: Number of unique products user interacted with
        
        Parameters:
        - user_ids: List of user IDs (if None, uses first 1000 users from matrix)
        - product_ids: List of product IDs (if None, uses first 500 products from matrix)
        - sample_size: Number of samples to create (for speed)
        
        Returns:
        - tuple: (X_features, y_labels)
          * X_features: Array of features (num_samples, 6)
          * y_labels: Array of labels (1 = interaction, 0 = no interaction)
        """
        if not NEURAL_NETWORK_AVAILABLE:
            print("Error: TensorFlow not available. Cannot prepare neural network features.")
            return None, None
        
        print("Preparing features for Neural Network ranking...")
        
        # בדיקה שהנתונים נטענו
        if self.interaction_matrix is None:
            print("Error: Interaction matrix not created. Call create_user_interaction_matrix() first.")
            return None, None
        
        if self.products_with_clusters is None or self.users_with_clusters is None:
            print("Error: Clustering results not loaded. Call load_data() first.")
            return None, None
        
        # בחירת משתמשים ומוצרים
        if user_ids is None:
            user_ids = self.all_user_ids[:min(1000, len(self.all_user_ids))]  # מגביל ל-1000 משתמשים
        
        if product_ids is None:
            product_ids = self.all_product_ids[:min(500, len(self.all_product_ids))]  # מגביל ל-500 מוצרים
        
        print(f"   Processing {len(user_ids)} users and {len(product_ids)} products...")
        
        # יצירת רשימת תכונות
        features_list = []
        labels_list = []
        
        # ספירת אינטראקציות לכל משתמש (לחישוב תכונות)
        user_interaction_stats = {}
        for uid in user_ids:
            if uid in self.user_id_to_index:
                user_idx = self.user_id_to_index[uid]
                user_row = self.interaction_matrix.iloc[user_idx]
                total_interactions = user_row.sum()
                num_products_interacted = (user_row > 0).sum()
                user_interaction_stats[uid] = {
                    'total_interactions': total_interactions,
                    'num_products': num_products_interacted
                }
        
        # יצירת דוגמאות
        count = 0
        for uid in user_ids:
            if count >= sample_size:
                break
            
            # תכונות משתמש
            user_cluster = 0
            if self.users_with_clusters is not None and 'user_id' in self.users_with_clusters.columns:
                if uid in self.users_with_clusters['user_id'].values:
                    user_row = self.users_with_clusters[self.users_with_clusters['user_id'] == uid]
                    if 'cluster' in user_row.columns:
                        user_cluster = user_row['cluster'].iloc[0]
            
            user_stats = user_interaction_stats.get(uid, {'total_interactions': 0, 'num_products': 0})
            
            for pid in product_ids:
                if count >= sample_size:
                    break
                
                # תכונות מוצר
                product_cluster = 0
                product_price = 0
                product_category = 0
                
                if pid in self.products_df['id'].values:
                    product_row = self.products_df[self.products_df['id'] == pid].iloc[0]
                    product_price = product_row.get('price', 0)
                    # קטגוריה - נמיר למספר
                    if 'main_category' in product_row:
                        category_str = str(product_row['main_category'])
                        product_category = hash(category_str) % 100  # המרה פשוטה למספר
                    elif 'category' in product_row:
                        category_str = str(product_row['category'])
                        product_category = hash(category_str) % 100  # המרה פשוטה למספר
                
                if pid in self.products_with_clusters['id'].values:
                    product_cluster_row = self.products_with_clusters[self.products_with_clusters['id'] == pid]
                    if 'ml_cluster' in product_cluster_row.columns:
                        cluster_value = product_cluster_row['ml_cluster'].iloc[0]
                        # Convert cluster to number (if it's a string category name, hash it)
                        if isinstance(cluster_value, str):
                            product_cluster = hash(cluster_value) % 1000  # Convert string to number
                        else:
                            product_cluster = float(cluster_value) if pd.notna(cluster_value) else 0
                
                # תכונות משולבות - וודא שכל הערכים הם מספרים
                feature_vector = [
                    float(user_cluster),                    # 0: אשכול משתמש
                    float(product_cluster),                  # 1: אשכול מוצר
                    float(product_price),                    # 2: מחיר מוצר
                    float(product_category),                 # 3: קטגוריה מוצר
                    float(user_stats['total_interactions']), # 4: סך אינטראקציות משתמש
                    float(user_stats['num_products']),       # 5: מספר מוצרים שהמשתמש התקשר איתם
                ]
                
                # תווית: האם יש אינטראקציה?
                label = 0
                if uid in self.user_id_to_index and pid in self.product_id_to_index:
                    column_name = f'product_{pid}'
                    if column_name in self.interaction_matrix.columns:
                        interaction_value = self.interaction_matrix.loc[uid, column_name]
                        label = 1 if interaction_value > 0 else 0
                
                features_list.append(feature_vector)
                labels_list.append(label)
                count += 1
        
        # המרה למערכים
        X_features = np.array(features_list)
        y_labels = np.array(labels_list)
        
        print(f"   Created {len(features_list)} samples with {X_features.shape[1]} features")
        print(f"   Positive samples (interactions): {y_labels.sum()}, Negative: {(y_labels == 0).sum()}")
        
        return X_features, y_labels
    
    def build_neural_ranking_model(self, input_dim=6, hidden_units_1=64, hidden_units_2=32):
        """
        Builds a Neural Network model for ranking products
        
        Architecture:
        - Input: 6 features (user_cluster, product_cluster, price, category, total_interactions, num_products)
        - Hidden Layer 1: 64 neurons with ReLU activation + Dropout (20%)
        - Hidden Layer 2: 32 neurons with ReLU activation + Dropout (20%)
        - Output: 1 neuron with Sigmoid activation (returns score 0-1)
        
        How it works:
        1. Input (6 features) → enters network
        2. Hidden Layer 1 (64 neurons) → learns complex patterns
        3. Hidden Layer 2 (32 neurons) → continues learning
        4. Output (1 neuron) → returns relevance score (0-1)
        
        Parameters:
        - input_dim: Number of input features (6)
        - hidden_units_1: Number of neurons in first hidden layer (64)
        - hidden_units_2: Number of neurons in second hidden layer (32)
        
        Returns:
        - keras.Model: Neural network model ready for training
        """
        if not NEURAL_NETWORK_AVAILABLE:
            print("Error: TensorFlow not available. Cannot build neural network model.")
            return None
        
        print("Building Neural Network ranking model...")
        print(f"   Input: {input_dim} features")
        print(f"   Hidden Layer 1: {hidden_units_1} neurons")
        print(f"   Hidden Layer 2: {hidden_units_2} neurons")
        print(f"   Output: 1 score (0-1)")
        
        # יצירת המודל
        model = keras.Sequential([
            # שכבה 1: קלט + שכבה נסתרת ראשונה
            # Dense = שכבה מלאה (כל נוירון מחובר לכל נוירון בשכבה הבאה)
            # input_dim = מספר התכונות בקלט (6)
            # units = מספר נוירונים בשכבה (64)
            # activation = פונקציית הפעלה (ReLU = Rectified Linear Unit)
            # ReLU אומר: אם הערך שלילי → 0, אם חיובי → הערך עצמו
            layers.Dense(units=hidden_units_1, activation='relu', input_dim=input_dim, name='hidden_layer_1'),
            
            # Dropout: "מכבה" 20% מהנוירונים באופן אקראי בזמן אימון
            # למה? כדי למנוע "שינון בעל פה" (overfitting)
            # זה עוזר לרשת ללמוד דפוסים כלליים יותר
            layers.Dropout(rate=0.2, name='dropout_1'),
            
            # שכבה 2: שכבה נסתרת שנייה
            # פחות נוירונים (32) → פחות מורכבות, יותר מהירות
            layers.Dense(units=hidden_units_2, activation='relu', name='hidden_layer_2'),
            
            # Dropout נוסף
            layers.Dropout(rate=0.2, name='dropout_2'),
            
            # שכבה 3: פלט
            # 1 נוירון → מחזיר ציון אחד
            # sigmoid = פונקציה שמחזירה ערך בין 0 ל-1
            # 0 = לא רלוונטי, 1 = מאוד רלוונטי
            layers.Dense(units=1, activation='sigmoid', name='output_layer')
        ])
        
        # קומפילציה של המודל
        # compile = הגדרת איך המודל ילמד
        # optimizer = איך המודל מתעדכן (adam = אלגוריתם חכם)
        # loss = איך המודל מודד שגיאות (binary_crossentropy = לבעיות כן/לא)
        # metrics = מה המודל יציג בזמן אימון (accuracy = דיוק)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("   Model built successfully!")
        print(f"   Total parameters: {model.count_params():,}")
        
        # שמירת המודל
        self.neural_ranking_model = model
        
        return model
    
    def train_neural_ranking_model(self, X_features, y_labels, epochs=10, batch_size=32, validation_split=0.2):
        """
        Trains the Neural Network ranking model
        
        What it does:
        1. Normalizes features using StandardScaler
        2. Splits data into train (80%) and validation (20%)
        3. Trains the model for specified number of epochs
        4. Stores the feature scaler for future predictions
        
        Parameters:
        - X_features: Array of features (num_samples, 6)
        - y_labels: Array of labels (1 = interaction, 0 = no interaction)
        - epochs: Number of training rounds (default: 10)
        - batch_size: Number of samples processed together (default: 32)
        - validation_split: Percentage of data for validation (default: 0.2 = 20%)
        
        Returns:
        - keras.History: Training history (loss, accuracy for each epoch)
        """
        if not NEURAL_NETWORK_AVAILABLE:
            print("Error: TensorFlow not available. Cannot train neural network model.")
            return None
        
        if self.neural_ranking_model is None:
            print("Error: Model not built. Call build_neural_ranking_model() first.")
            return None
        
        if X_features is None or y_labels is None:
            print("Error: Features or labels are None. Call prepare_neural_network_features() first.")
            return None
        
        print("="*60)
        print("Training Neural Network Ranking Model")
        print("="*60)
        
        # נרמול התכונות (חשוב לרשתות עצביות!)
        # למה? כי תכונות עם ערכים גדולים יותר לא ישתלטו על התכונות הקטנות
        # StandardScaler = מנרמל כל תכונה להיות בין -1 ל-1 בערך
        print("\nStep 1: Normalizing features...")
        scaler = StandardScaler()
        X_features_normalized = scaler.fit_transform(X_features)
        print(f"   Features normalized: shape {X_features_normalized.shape}")
        
        # חלוקה ל-Train ו-Validation
        # Train = נתונים שהמודל ילמד מהם
        # Validation = נתונים שנבדוק עליהם (המודל לא רואה אותם בזמן אימון)
        print(f"\nStep 2: Splitting data...")
        print(f"   Total samples: {len(X_features_normalized)}")
        print(f"   Train: {int(len(X_features_normalized) * (1 - validation_split))} samples ({(1-validation_split)*100:.0f}%)")
        print(f"   Validation: {int(len(X_features_normalized) * validation_split)} samples ({validation_split*100:.0f}%)")
        
        # אימון המודל
        # fit = הפונקציה שמאמנת את המודל
        # verbose = כמה פרטים להציג (1 = הצג כל epoch)
        print(f"\nStep 3: Training model...")
        print(f"   Epochs: {epochs} (הרשת תראה את הנתונים {epochs} פעמים)")
        print(f"   Batch size: {batch_size} (בודקים {batch_size} דוגמאות ביחד)")
        print(f"   This may take a few minutes...")
        print()
        
        # האימון עצמו
        history = self.neural_ranking_model.fit(
            X_features_normalized,  # תכונות (מנורמלות)
            y_labels,                # תוויות (1 או 0)
            epochs=epochs,           # מספר סיבובים
            batch_size=batch_size,   # גודל קבוצה
            validation_split=validation_split,  # 20% ל-Validation
            verbose=1                # הצג פרטים
        )
        
        # סיכום תוצאות
        print("\n" + "="*60)
        print("Training Completed!")
        print("="*60)
        
        # תוצאות אחרונות
        final_train_loss = history.history['loss'][-1]
        final_train_accuracy = history.history['accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_val_accuracy = history.history['val_accuracy'][-1]
        
        print(f"\nFinal Results:")
        print(f"   Train Loss: {final_train_loss:.4f} (כמה נמוך = כמה טוב)")
        print(f"   Train Accuracy: {final_train_accuracy:.4f} ({final_train_accuracy*100:.1f}%)")
        print(f"   Validation Loss: {final_val_loss:.4f}")
        print(f"   Validation Accuracy: {final_val_accuracy:.4f} ({final_val_accuracy*100:.1f}%)")
        
        # בדיקה אם יש overfitting
        # Overfitting = המודל למד "בעל פה" את Train אבל לא מבין Validation
        if final_train_accuracy > final_val_accuracy + 0.1:
            print(f"\n   Warning: Possible overfitting detected!")
            print(f"   (Train accuracy is much higher than Validation accuracy)")
        else:
            print(f"\n   Model looks good! (Train and Validation are similar)")
        
        # שמירת ה-scaler לשימוש בעתיד
        self.feature_scaler = scaler
        
        return history
    
    def check_and_retrain_neural_network(self, force_retrain=False):
        """
        Implements Continuous Learning - checks if retraining is needed and retrains if threshold met
        
        What it does:
        1. Checks how many new interactions have occurred
        2. If threshold met (100+ interactions) → retrains the model
        3. If below threshold → skips retraining (saves time)
        
        Why is this important?
        - Neural network learned from old data
        - New interactions provide new feedback
        - Retraining allows the model to learn from new feedback
        - This is Continuous Learning!
        
        How it works:
        1. Check: Counts new interactions (threshold: 100)
        2. Prepare features: Creates features from new interactions
        3. Retrain: Trains model on new data (3 epochs, smaller batch)
        4. Reset: Clears new_interactions list and counter
        
        Parameters:
        - force_retrain: If True, retrains even if below threshold
        
        Returns:
        - bool: True if retrained, False if skipped
        """
        if not NEURAL_NETWORK_AVAILABLE:
            print("TensorFlow not available. Cannot retrain neural network.")
            return False
        
        if self.neural_ranking_model is None:
            print("Neural network model not trained yet. Cannot retrain.")
            return False
        
        # שלב 1: בדיקה - כמה אינטראקציות חדשות יש?
        if not force_retrain and self.new_interactions_count < self.retrain_threshold:
            # יש פחות מ-100 אינטראקציות חדשות
            print(f"Not enough new interactions for retraining: {self.new_interactions_count}/{self.retrain_threshold}")
            print(f"   (Need at least {self.retrain_threshold} new interactions to retrain)")
            return False
        
        print("="*60)
        print("Continuous Learning: Retraining Neural Network")
        print("="*60)
        print(f"   Found {self.new_interactions_count} new interactions")
        print(f"   Threshold: {self.retrain_threshold}")
        print(f"   Retraining model with new data...")
        
        # שלב 2: איסוף אינטראקציות חדשות
        # לוקחים את כל האינטראקציות החדשות
        new_interactions = self.new_interactions.copy()
        
        if not new_interactions:
            print("   No new interactions to process.")
            return False
        
        print(f"   Processing {len(new_interactions)} new interactions...")
        
        # שלב 3: הכנת תכונות מהאינטראקציות החדשות
        # זה דומה ל-prepare_neural_network_features, אבל רק לאינטראקציות החדשות
        new_features_list = []
        new_labels_list = []
        
        for interaction in new_interactions:
            user_id = interaction['user_id']
            product_id = interaction['product_id']
            
            # תכונות משתמש (כמו ב-prepare_neural_network_features)
            user_cluster = 0
            if self.users_with_clusters is not None and 'user_id' in self.users_with_clusters.columns:
                if user_id in self.users_with_clusters['user_id'].values:
                    user_row = self.users_with_clusters[self.users_with_clusters['user_id'] == user_id]
                    if 'cluster' in user_row.columns:
                        user_cluster = user_row['cluster'].iloc[0]
            
            # סטטיסטיקות משתמש
            total_interactions = 0
            num_products = 0
            if user_id in self.user_id_to_index:
                user_idx = self.user_id_to_index[user_id]
                user_row = self.interaction_matrix.iloc[user_idx]
                total_interactions = user_row.sum()
                num_products = (user_row > 0).sum()
            
            # תכונות מוצר
            product_cluster = 0
            product_price = 0
            product_category = 0
            
            if self.products_df is not None and product_id in self.products_df['id'].values:
                product_row = self.products_df[self.products_df['id'] == product_id].iloc[0]
                product_price = product_row.get('price', 0)
                if 'main_category' in product_row:
                    category_str = str(product_row['main_category'])
                    product_category = hash(category_str) % 100
                elif 'category' in product_row:
                    category_str = str(product_row['category'])
                    product_category = hash(category_str) % 100
            
            if self.products_with_clusters is not None and product_id in self.products_with_clusters['id'].values:
                product_cluster_row = self.products_with_clusters[self.products_with_clusters['id'] == product_id]
                if 'ml_cluster' in product_cluster_row.columns:
                    product_cluster = product_cluster_row['ml_cluster'].iloc[0]
            
            # יצירת וקטור תכונות - וודא שכל הערכים הם מספרים
            feature_vector = [
                float(user_cluster),
                float(product_cluster),
                float(product_price),
                float(product_category),
                float(total_interactions),
                float(num_products)
            ]
            
            # תווית: יש אינטראקציה (כי זה מהאינטראקציות החדשות)
            label = 1
            
            new_features_list.append(feature_vector)
            new_labels_list.append(label)
        
        # המרה למערכים
        new_X_features = np.array(new_features_list)
        new_y_labels = np.array(new_labels_list)
        
        print(f"   Created {len(new_features_list)} new samples")
        
        # שלב 4: אימון מחדש
        # מאמנים את הרשת על הנתונים החדשים (עם פחות epochs כי זה עדכון קטן)
        print(f"   Retraining model with new data...")
        
        # נרמול התכונות החדשות (עם אותו scaler)
        if self.feature_scaler is not None:
            new_X_features_normalized = self.feature_scaler.transform(new_X_features)
        else:
            # אם אין scaler, ניצור אחד חדש
            scaler = StandardScaler()
            new_X_features_normalized = scaler.fit_transform(new_X_features)
            self.feature_scaler = scaler
        
        # אימון מחדש (עם פחות epochs כי זה עדכון קטן)
        history = self.neural_ranking_model.fit(
            new_X_features_normalized,
            new_y_labels,
            epochs=3,  # פחות epochs כי זה עדכון קטן
            batch_size=min(32, len(new_X_features)),
            verbose=1
        )
        
        # איפוס המונים (כי כבר אימנו על האינטראקציות האלה)
        self.new_interactions = []  # מנקים את הרשימה
        self.new_interactions_count = 0  # מאפסים את המונה
        
        print(f"\n   Retraining completed!")
        print(f"   Model updated with {len(new_features_list)} new interactions")
        print("="*60)
        
        return True
    
    def predict_product_score(self, user_id, product_id):
        """
        Uses Neural Network to predict relevance score for a user-product pair
        
        What it does:
        1. Extracts 6 features (user_cluster, product_cluster, price, category, total_interactions, num_products)
        2. Normalizes features using self.feature_scaler (from training)
        3. Sends to Neural Network for prediction
        4. Returns relevance score (0-1)
        
        Score interpretation:
        - 0.0 - 0.3: Not relevant (user probably won't like)
        - 0.3 - 0.6: Maybe relevant (could be interesting)
        - 0.6 - 0.8: Relevant (user probably will like)
        - 0.8 - 1.0: Very relevant (user probably will love)
        
        Parameters:
        - user_id: User ID
        - product_id: Product ID
        
        Returns:
        - float: Relevance score (0-1), or None if error
        """
        if not NEURAL_NETWORK_AVAILABLE:
            print("Error: TensorFlow not available.")
            return None
        
        if self.neural_ranking_model is None:
            print("Error: Neural network model not trained. Call train_neural_ranking_model() first.")
            return None
        
        if self.feature_scaler is None:
            print("Error: Feature scaler not found. Model needs to be trained first.")
            return None
        
        # שלב 1: הכנת תכונות
        # זה אותו קוד כמו ב-prepare_neural_network_features, אבל רק לזוג אחד
        
        # תכונות משתמש
        user_cluster = 0
        if self.users_with_clusters is not None and 'user_id' in self.users_with_clusters.columns:
            if user_id in self.users_with_clusters['user_id'].values:
                user_row = self.users_with_clusters[self.users_with_clusters['user_id'] == user_id]
                if 'cluster' in user_row.columns:
                    user_cluster = user_row['cluster'].iloc[0]
        
        # סטטיסטיקות משתמש
        total_interactions = 0
        num_products = 0
        if user_id in self.user_id_to_index:
            user_idx = self.user_id_to_index[user_id]
            user_row = self.interaction_matrix.iloc[user_idx]
            total_interactions = user_row.sum()
            num_products = (user_row > 0).sum()
        
        # תכונות מוצר
        product_cluster = 0
        product_price = 0
        product_category = 0
        
        if self.products_df is not None and product_id in self.products_df['id'].values:
            product_row = self.products_df[self.products_df['id'] == product_id].iloc[0]
            product_price = product_row.get('price', 0)
            # קטגוריה - נמיר למספר
            if 'main_category' in product_row:
                category_str = str(product_row['main_category'])
                product_category = hash(category_str) % 100
            elif 'category' in product_row:
                category_str = str(product_row['category'])
                product_category = hash(category_str) % 100
        
        if self.products_with_clusters is not None and product_id in self.products_with_clusters['id'].values:
            product_cluster_row = self.products_with_clusters[self.products_with_clusters['id'] == product_id]
            if 'ml_cluster' in product_cluster_row.columns:
                cluster_value = product_cluster_row['ml_cluster'].iloc[0]
                # Convert cluster to number (if it's a string category name, hash it)
                if isinstance(cluster_value, str):
                    product_cluster = hash(cluster_value) % 1000  # Convert string to number
                else:
                    product_cluster = float(cluster_value) if pd.notna(cluster_value) else 0
        
        # יצירת וקטור תכונות (בדיוק כמו ב-prepare_neural_network_features)
        feature_vector = np.array([[
            float(user_cluster),           # 0: אשכול משתמש
            float(product_cluster),        # 1: אשכול מוצר
            float(product_price),          # 2: מחיר מוצר
            float(product_category),       # 3: קטגוריה מוצר
            float(total_interactions),     # 4: סך אינטראקציות משתמש
            float(num_products)            # 5: מספר מוצרים שהמשתמש התקשר איתם
        ]])
        
        # שלב 2: נרמול (חשוב! המודל למד על תכונות מנורמלות)
        # משתמשים ב-self.feature_scaler שנוצר בזמן אימון
        feature_vector_normalized = self.feature_scaler.transform(feature_vector)
        
        # שלב 3: חיזוי עם הרשת העצבית
        # predict = הפונקציה שמחזירה ציון
        # [0][0] = לוקחים את הערך הראשון (יש רק אחד)
        score = self.neural_ranking_model.predict(feature_vector_normalized, verbose=0)[0][0]
        
        return float(score)
    
    def hybrid_recommendations_with_neural_ranking(self, user_id, n_recommendations=5, use_neural_ranking=True):
        """
        Hybrid recommendations combining Collaborative + Content-Based + Neural Network Ranking
        
        What it does:
        1. Gets base recommendations from hybrid_recommendations (Collaborative 70% + Content-Based 30%)
        2. Scores each recommendation using Neural Network (predict_product_score)
        3. Combines scores: 60% base score + 40% neural score
        4. Re-ranks and returns top recommendations
        
        How it works:
        1. Base recommendations: Gets recommendations from hybrid approach
        2. Neural scoring: Calculates neural network score for each product
        3. Score combination: final_score = (base_score × 0.6) + (neural_score × 0.4)
        4. Re-ranking: Sorts by final score and returns top N
        
        Why is this better?
        - Neural Network learns complex patterns
        - Can identify relationships classical methods miss
        - Combined approach gives more accurate recommendations
        
        Parameters:
        - user_id: User ID
        - n_recommendations: Number of recommendations to return
        - use_neural_ranking: If True, uses Neural Network (if False, uses base hybrid only)
        
        Returns:
        - List of recommended product_ids (ranked with Neural Network)
        """
        print(f"Preparing hybrid recommendations with Neural Network ranking for user {user_id}...")
        
        # שלב 1: המלצות בסיסיות (Collaborative + Content-Based)
        # זה הפונקציה הקיימת שלנו
        base_recommendations = self.hybrid_recommendations(user_id, n_recommendations=n_recommendations * 2)
        # n_recommendations * 2 = לוקחים יותר המלצות, נדרג אותם ונחזיר את הטובות
        
        if not base_recommendations:
            print("No base recommendations found.")
            return []
        
        # בדיקה אם יש רשת עצבית מאומנת
        if not use_neural_ranking or self.neural_ranking_model is None:
            print("Neural network not available. Using base recommendations only.")
            return base_recommendations[:n_recommendations]
        
        print(f"   Found {len(base_recommendations)} base recommendations")
        print(f"   Ranking with Neural Network...")
        
        # שלב 2: דירוג כל המלצה עם הרשת העצבית
        scored_recommendations = []
        
        for product_id in base_recommendations:
            # חישוב ציון בסיסי (מ-hybrid_recommendations)
            # הציון הבסיסי הוא המיקום ברשימה (הראשון = הכי טוב)
            base_score = 1.0 - (base_recommendations.index(product_id) / len(base_recommendations))
            # למשל: מוצר ראשון = 1.0, מוצר שני = 0.9, וכו'
            
            # חישוב ציון רשת עצבית
            neural_score = self.predict_product_score(user_id, product_id)
            
            if neural_score is None:
                # אם הרשת לא הצליחה, משתמשים רק בציון בסיסי
                neural_score = 0.5  # ציון בינוני
            
            # שלב 3: שילוב ציונים
            # משקלים: 50% בסיסי, 50% רשת עצבית (יותר משקל לרשת)
            final_score = (base_score * 0.5) + (neural_score * 0.5)
            
            scored_recommendations.append({
                'product_id': product_id,
                'base_score': base_score,
                'neural_score': neural_score,
                'final_score': final_score
            })
        
        # שלב 4: דירוג מחדש לפי הציון הסופי
        # מסדרים מהגבוה לנמוך
        scored_recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        
        # לוקחים את הטובות ביותר
        final_recommendations = [rec['product_id'] for rec in scored_recommendations[:n_recommendations]]
        
        print(f"   Top {n_recommendations} recommendations after Neural Network ranking:")
        for i, rec in enumerate(scored_recommendations[:n_recommendations], 1):
            print(f"      {i}. Product {rec['product_id']}: Final Score = {rec['final_score']:.3f} "
                  f"(Base: {rec['base_score']:.3f}, Neural: {rec['neural_score']:.3f})")
        
        return final_recommendations
        
    def recommend_for_new_user(self, user_interactions):
        """
        Recommends products for new users (< 3 interactions) using TF-IDF + Cosine Similarity
        
        What it does:
        - Finds products user has interacted with
        - Calculates average TF-IDF vector of interested products
        - Finds similar products using cosine similarity
        - Returns top 5 recommendations
        
        Parameters:
        - user_interactions: Dictionary {product_id: interaction_value}
        
        Returns:
        - List of 5 recommended product_ids
        """
        print("Preparing recommendations for new user...")
        
        if self.product_tfidf_matrix is None:
            self.prepare_tfidf_for_products()
        
        # מציאת מוצרים שהמשתמש התעניין בהם
        interested_products = []
        for product_id, interaction in user_interactions.items():
            if interaction > 0:
                interested_products.append(product_id - 1)  # התאמה לאינדקס
        
        if not interested_products:
            # אם אין אינטראקציות, החזרת מוצרים פופולריים
            popular_products = self.product_metadata_df.nlargest(5, 'clicks')['pid'].tolist()
            return popular_products
        
        # חישוב וקטור ממוצע של מוצרים מעניינים
        product_vectors = self.product_tfidf_matrix[interested_products]
        # המרה ל-array אם זה sparse matrix
        if hasattr(product_vectors, 'toarray'):
            product_vectors = product_vectors.toarray()
        avg_vector = np.mean(product_vectors, axis=0)
        
        # חישוב דמיון לכל המוצרים (avg_vector צריך להיות 2D)
        # המרה ל-array אם זה sparse matrix
        if hasattr(self.product_tfidf_matrix, 'toarray'):
            product_tfidf_array = self.product_tfidf_matrix.toarray()
        else:
            product_tfidf_array = np.asarray(self.product_tfidf_matrix)
        similarities = cosine_similarity(avg_vector.reshape(1, -1), product_tfidf_array).flatten()
        
        # דירוג והחזרת Top-K
        top_indices = np.argsort(similarities)[::-1][:5]
        recommendations = [idx + 1 for idx in top_indices if idx + 1 not in interested_products]
        
        return recommendations[:5]
    
    def recommend_for_old_user_collaborative(self, user_id, n_recommendations=5):
        """
        Recommends products for existing users using Collaborative Filtering
        
        What it does:
        - Finds similar users based on interaction patterns
        - Recommends products that similar users liked
        - Uses weighted scores based on user similarity
        
        How it works:
        1. Finds 3 most similar users
        2. For each similar user, finds products they liked
        3. Calculates weighted score: similarity × interaction_value
        4. Returns top N products
        
        Parameters:
        - user_id: User ID
        - n_recommendations: Number of recommendations to return
        
        Returns:
        - List of recommended product_ids
        """
        print(f"Preparing recommendations for user {user_id} (Collaborative Filtering)...")
        
        if user_id not in self.interaction_matrix.index:
            # אם המשתמש לא קיים במטריצה, מטפלים בו כמשווק חדש
            print(f"User {user_id} not found in matrix, switching to new user approach...")
            # איסוף אינטראקציות - Long format
            user_interactions = {}
            user_clicks = self.clicks_df[self.clicks_df['uid'] == user_id]
            for _, row in user_clicks.iterrows():
                product_id = row['product_id']
                if product_id not in user_interactions:
                    user_interactions[product_id] = 0
                user_interactions[product_id] += row['clicks']
            user_purchases = self.purchases_df[self.purchases_df['uid'] == user_id]
            for _, row in user_purchases.iterrows():
                product_id = row['product_id']
                if product_id not in user_interactions:
                    user_interactions[product_id] = 0
                user_interactions[product_id] += row['purchases']
            user_visits = self.visits_time_df[self.visits_time_df['uid'] == user_id]
            for _, row in user_visits.iterrows():
                product_id = row['product_id']
                if product_id not in user_interactions:
                    user_interactions[product_id] = 0
                user_interactions[product_id] += row['visit_time']
            return self.recommend_for_new_user(user_interactions)
        
        # מציאת משתמשים דומים
        user_idx = self.interaction_matrix.index.get_loc(user_id)
        user_similarities = self.user_similarity_matrix[user_idx]
        
        # דירוג משתמשים לפי דמיון
        similar_users = np.argsort(user_similarities)[::-1][1:4]  # 3 המשתמשים הכי דומים
        
        # חישוב ציון לכל מוצר
        user_ratings = self.interaction_matrix.loc[user_id]
        recommendations = {}
        
        for similar_user_idx in similar_users:
            similar_user_id = self.interaction_matrix.index[similar_user_idx]
            similar_user_ratings = self.interaction_matrix.loc[similar_user_id]
            similarity = user_similarities[similar_user_idx]
            
            # חישוב ציון משוקלל
            for product_id in similar_user_ratings.index:
                if user_ratings[product_id] == 0 and similar_user_ratings[product_id] > 0:
                    if product_id not in recommendations:
                        recommendations[product_id] = 0
                    recommendations[product_id] += similarity * similar_user_ratings[product_id]
        
        # דירוג ההמלצות
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        
        return [int(rec[0].split('_')[1]) for rec in sorted_recommendations[:n_recommendations]]
    
    def recommend_for_old_user_content_based(self, user_id, n_recommendations=5):
        """
        Recommends products for existing users using Content-Based Filtering
        
        What it does:
        - Finds products user has interacted with
        - Identifies user's preferred categories
        - Recommends similar products from same categories
        - Ranks by popularity (views)
        
        Parameters:
        - user_id: User ID
        - n_recommendations: Number of recommendations to return
        
        Returns:
        - List of recommended product_ids
        """
        print(f"Preparing recommendations for user {user_id} (Content-based Filtering)...")
        
        # מציאת מוצרים שהמשתמש כבר רכש/לחץ עליהם - Long format
        user_clicks = self.clicks_df[self.clicks_df['uid'] == user_id]
        user_purchases = self.purchases_df[self.purchases_df['uid'] == user_id]
        
        # מציאת מוצרים מעניינים
        interested_products = set()
        
        # מוצרים עם קליקים
        if len(user_clicks) > 0:
            interested_products.update(user_clicks[user_clicks['clicks'] > 0]['product_id'].tolist())
        
        # מוצרים עם רכישות
        if len(user_purchases) > 0:
            interested_products.update(user_purchases[user_purchases['purchases'] > 0]['product_id'].tolist())
        
        interested_products = list(interested_products)
        
        if not interested_products:
            return []
        
        # מציאת קטגוריות מועדפות
        user_categories = []
        for product_id in interested_products:
            product_row = self.products_df[self.products_df['id'] == product_id]
            if len(product_row) > 0:
                # Use main_category if available, otherwise fall back to category
                if 'main_category' in product_row.columns:
                    category = product_row.iloc[0]['main_category']
                elif 'category' in product_row.columns:
                    category = product_row.iloc[0]['category']
                else:
                    continue
                user_categories.append(category)
        
        if not user_categories:
            return []
        
        # מציאת מוצרים דומים בקטגוריות המועדפות
        # נשתמש בקטגוריות שהמשתמש רכש (לא רק לחץ) - זה יותר חשוב
        user_purchased_categories = []
        if len(user_purchases) > 0:
            purchased_product_ids = user_purchases[user_purchases['purchases'] > 0]['product_id'].tolist()
            for product_id in purchased_product_ids:
                product_row = self.products_df[self.products_df['id'] == product_id]
                if len(product_row) > 0:
                    if 'main_category' in product_row.columns:
                        user_purchased_categories.append(product_row.iloc[0]['main_category'])
                    elif 'category' in product_row.columns:
                        user_purchased_categories.append(product_row.iloc[0]['category'])
        
        # העדפה חזקה לקטגוריות רכישה - אם יש רכישות, נשתמש רק בהן
        if user_purchased_categories:
            categories_to_use = set(user_purchased_categories)
            # אם יש רק קטגוריה אחת, נוסיף גם קטגוריות מקליקים (אבל עם עדיפות נמוכה יותר)
            if len(categories_to_use) == 1 and len(user_categories) > 1:
                # נוסיף קטגוריה אחת נוספת מקליקים
                click_categories = set(user_categories) - categories_to_use
                if click_categories:
                    categories_to_use.add(list(click_categories)[0])
        else:
            categories_to_use = set(user_categories)
        
        recommendations = []
        category_scores = {}  # נשתמש בציונים כדי לדרג מוצרים
        
        for category in categories_to_use:
            # Use main_category if available, otherwise fall back to category
            if 'main_category' in self.products_df.columns:
                category_products = self.products_df[
                    (self.products_df['main_category'] == category) &
                    (~self.products_df['id'].isin(interested_products))
                ]
            else:
                category_products = self.products_df[
                    (self.products_df['category'] == category) &
                    (~self.products_df['id'].isin(interested_products))
                ]
            
            if len(category_products) > 0:
                # דירוג לפי פופולריות (views) - מוצרים פופולריים יותר מקבלים ציון גבוה יותר
                category_products = category_products.sort_values('views', ascending=False)
                
                # ציון לקטגוריה - קטגוריות רכישה מקבלות ציון גבוה יותר
                category_weight = 2.0 if category in user_purchased_categories else 1.0
                
                # הוספת מוצרים עם ציונים משוקללים
                for idx, (_, product_row) in enumerate(category_products.iterrows()):
                    product_id = product_row['id']
                    # ציון = משקל קטגוריה × (1 / מיקום) × פופולריות (views נורמלית)
                    views_score = min(product_row['views'] / 1000000, 1.0)  # נרמול views
                    position_score = 1.0 / (idx + 1)  # מיקום ברשימה
                    final_score = category_weight * position_score * (1 + views_score)
                    
                    if product_id not in category_scores:
                        category_scores[product_id] = 0
                    category_scores[product_id] += final_score
        
        # דירוג לפי ציון סופי
        sorted_products = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [product_id for product_id, _ in sorted_products[:n_recommendations]]
        
        return recommendations
    
    def hybrid_recommendations(self, user_id, n_recommendations=5):
        """
        Hybrid recommendations combining Collaborative Filtering (70%) + Content-Based (30%)
        
        What it does:
        - Checks if user is new (< 3 interactions) or old (3+ interactions)
        - For new users: Uses TF-IDF approach (recommend_for_new_user)
        - For old users: Combines Collaborative (70%) + Content-Based (30%)
        
        How it combines:
        - Collaborative recommendations get weight 0.7
        - Content-Based recommendations get weight 0.3
        - Products appearing in both get combined score
        - Returns top N by final score
        
        Parameters:
        - user_id: User ID
        - n_recommendations: Number of recommendations to return
        
        Returns:
        - List of recommended product_ids
        """
        print(f"Preparing hybrid recommendations for user {user_id}...")
        
        # בדיקה אם המשתמש חדש או ותיק - Long format
        user_interactions = {}
        
        # איסוף אינטראקציות מקליקים
        user_clicks = self.clicks_df[self.clicks_df['uid'] == user_id]
        for _, row in user_clicks.iterrows():
            product_id = row['product_id']
            if product_id not in user_interactions:
                user_interactions[product_id] = 0
            user_interactions[product_id] += row['clicks']
        
        # איסוף אינטראקציות מרכישות
        user_purchases = self.purchases_df[self.purchases_df['uid'] == user_id]
        for _, row in user_purchases.iterrows():
            product_id = row['product_id']
            if product_id not in user_interactions:
                user_interactions[product_id] = 0
            user_interactions[product_id] += row['purchases']
        
        # איסוף אינטראקציות מזמן ביקור
        user_visits = self.visits_time_df[self.visits_time_df['uid'] == user_id]
        for _, row in user_visits.iterrows():
            product_id = row['product_id']
            if product_id not in user_interactions:
                user_interactions[product_id] = 0
            user_interactions[product_id] += row['visit_time']
        
        # ספירת אינטראקציות
        total_interactions = sum(user_interactions.values())
        
        if total_interactions < 3:  # משתמש חדש
            print("New user - using TF-IDF")
            return self.recommend_for_new_user(user_interactions)
        else:  # משתמש ותיק
            print("Old user - using hybrid approach")
            
            # המלצות Collaborative
            cf_recommendations = self.recommend_for_old_user_collaborative(user_id, n_recommendations)
            
            # המלצות Content-based
            cb_recommendations = self.recommend_for_old_user_content_based(user_id, n_recommendations)
            
            # שילוב ההמלצות
            hybrid_recs = {}
            
            # שינוי משקלים - Content-Based מקבל משקל גבוה יותר (50%) כי הוא יותר מדויק לקטגוריות
            # הוספת המלצות Collaborative עם משקל בינוני
            for i, product_id in enumerate(cf_recommendations):
                hybrid_recs[product_id] = (len(cf_recommendations) - i) * 0.5
            
            # הוספת המלצות Content-based עם משקל גבוה יותר (50%)
            for i, product_id in enumerate(cb_recommendations):
                if product_id not in hybrid_recs:
                    hybrid_recs[product_id] = (len(cb_recommendations) - i) * 0.5
                else:
                    hybrid_recs[product_id] += (len(cb_recommendations) - i) * 0.5
            
            # דירוג סופי
            sorted_hybrid = sorted(hybrid_recs.items(), key=lambda x: x[1], reverse=True)
            
            return [rec[0] for rec in sorted_hybrid[:n_recommendations]]
    
    def evaluate_recommendations(self):
        """
        Evaluates recommendation quality using Precision@K metric
        
        What it does:
        - Tests recommendations on sample users
        - Calculates Precision@3 (category match)
        - Compares recommended categories with purchased categories
        - Returns evaluation results
        
        Metric:
        - Precision@3: Percentage of recommendations in same categories as user's purchases
        
        Returns:
        - List of evaluation results (one per test user)
        """
        print("\nEvaluating recommendation quality...")
        
        # מציאת כל המשתמשים שיש להם רכישות (לבדיקה)
        # משתמשים עם רכישות הם הטובים ביותר לבדיקה כי יש לנו "אמת" (מה הם באמת קנו)
        users_with_purchases = self.purchases_df[self.purchases_df['purchases'] > 0]['uid'].unique()
        
        # אם יש מעט משתמשים עם רכישות, נוסיף גם משתמשים עם קליקים
        if len(users_with_purchases) < 5:
            users_with_clicks = self.clicks_df[self.clicks_df['clicks'] > 0]['uid'].unique()
            test_users = sorted(list(set(list(users_with_purchases) + list(users_with_clicks[:10]))))
        else:
            test_users = sorted(list(users_with_purchases))
        
        # אפשרות: לבדוק את כל המשתמשים (5000) - אבל זה יקח הרבה זמן
        # לכן נבדוק רק משתמשים עם אינטראקציות (לפחות קליק אחד)
        # אם רוצים לבדוק את כל 5000, אפשר לשנות את זה
        all_users_with_interactions = set(self.clicks_df['uid'].unique()) | set(self.purchases_df['uid'].unique()) | set(self.visits_time_df['uid'].unique())
        
        print(f"Found {len(test_users)} users with purchases for testing:")
        print(f"  - Users with purchases: {len(users_with_purchases)}")
        print(f"  - Total users with any interaction: {len(all_users_with_interactions)}")
        print(f"  - Total users in system: {len(self.users_df)}")
        print(f"  - Test users: {test_users[:10]}{'...' if len(test_users) > 10 else ''}")
        print(f"\nNote: Testing only users with purchases (for ground truth).")
        print(f"      To test all {len(self.users_df)} users, modify evaluate_recommendations().")
        
        results = []
        
        for user_id in test_users:
            # בדיקה אם המשתמש קיים בטבלאות האינטראקציות
            user_exists = (user_id in self.clicks_df['uid'].values or 
                          user_id in self.purchases_df['uid'].values or 
                          user_id in self.visits_time_df['uid'].values)
            
            if user_exists:
                # המלצות - משתמשים ברשת נוירונים אם זמינה
                if NEURAL_NETWORK_AVAILABLE and self.neural_ranking_model is not None:
                    recommendations = self.hybrid_recommendations_with_neural_ranking(user_id, 3, use_neural_ranking=True)
                else:
                    recommendations = self.hybrid_recommendations(user_id, 3)
                
                # בדיקת רלוונטיות - Long format
                user_purchases = self.purchases_df[self.purchases_df['uid'] == user_id]
                purchased_products = user_purchases[user_purchases['purchases'] > 0]['product_id'].tolist()
                
                # חישוב Precision@3 - בדיקה לפי קטגוריות (מוצרים דומים)
                # מוצר רלוונטי = מוצר מהמלצות שנמצא באותה קטגוריה כמו מוצרים שהמשתמש רכש
                recommended_categories = []
                for rec_id in recommendations:
                    product_row = self.products_df[self.products_df['id'] == rec_id]
                    if len(product_row) > 0:
                        # Use main_category if available, otherwise fall back to category
                        if 'main_category' in product_row.columns:
                            recommended_categories.append(product_row.iloc[0]['main_category'])
                        elif 'category' in product_row.columns:
                            recommended_categories.append(product_row.iloc[0]['category'])
                
                purchased_categories = []
                for pur_id in purchased_products:
                    product_row = self.products_df[self.products_df['id'] == pur_id]
                    if len(product_row) > 0:
                        # Use main_category if available, otherwise fall back to category
                        if 'main_category' in product_row.columns:
                            purchased_categories.append(product_row.iloc[0]['main_category'])
                        elif 'category' in product_row.columns:
                            purchased_categories.append(product_row.iloc[0]['category'])
                
                # Precision = כמה מההמלצות באותן קטגוריות כמו הרכישות
                relevant_recommendations = len(set(recommended_categories) & set(purchased_categories))
                precision = relevant_recommendations / len(set(recommended_categories)) if recommended_categories else 0
                
                results.append({
                    'user_id': user_id,
                    'recommendations': recommendations,
                    'recommended_categories': list(set(recommended_categories)),
                    'purchased_products': purchased_products[:10],  # רק 10 ראשונים להצגה
                    'purchased_categories': list(set(purchased_categories)),
                    'precision@3': precision
                })
                
                print(f"User {user_id}: Recommendations {recommendations}")
                print(f"  Recommended categories: {list(set(recommended_categories))}")
                print(f"  Purchased categories: {list(set(purchased_categories))}")
                print(f"  Precision (category match): {precision:.2f}")
        
        if results:
            avg_precision = np.mean([r['precision@3'] for r in results])
            print(f"\nAverage Precision@3: {avg_precision:.2f}")
        else:
            print("No users found for testing")
        
        return results
    
    def run_phase2(self):
        """
        Runs the complete Phase 2 pipeline: Hybrid Recommendation System with Neural Network Ranking
        
        What it does (in order):
        1. Loads all data (products, users, interactions, clustering results)
        2. Prepares TF-IDF for product descriptions
        3. Creates user-product interaction matrix
        4. Calculates user similarity matrix
        5. Trains Neural Network for ranking (if TensorFlow available)
        6. Evaluates recommendation quality (using Neural Network if available)
        7. Saves evaluation results to CSV
        
        Note:
        - If TensorFlow is available, uses Neural Network ranking for better recommendations
        - If TensorFlow is not available, uses base hybrid recommendations (Collaborative + Content-Based)
        
        Returns:
        - List of evaluation results
        """
        print("="*80)
        print("Phase 2: Hybrid Recommendation System")
        print("="*80)
        
        # טעינת נתונים
        self.load_data()
        
        # הכנת מודלים
        self.prepare_tfidf_for_products()
        self.create_user_interaction_matrix()
        self.calculate_user_similarity()
        
        # אימון רשת נוירונים (אם זמין)
        if NEURAL_NETWORK_AVAILABLE:
            print("\n" + "="*80)
            print("Training Neural Network for Ranking")
            print("="*80)
            try:
                # הכנת תכונות
                X_features, y_labels = self.prepare_neural_network_features(sample_size=5000)
                if X_features is not None and len(X_features) > 0:
                    # בניית המודל
                    self.build_neural_ranking_model()
                    # אימון המודל - פחות epochs כדי למנוע overfitting
                    self.train_neural_ranking_model(X_features, y_labels, epochs=3, batch_size=16, validation_split=0.2)
                    print("Neural Network trained successfully!")
                else:
                    print("Warning: Could not prepare features for Neural Network. Using base recommendations.")
            except Exception as e:
                print(f"Warning: Could not train Neural Network: {e}")
                print("Continuing with base recommendations (Collaborative + Content-Based)")
        else:
            print("\nTensorFlow not available. Using base recommendations (Collaborative + Content-Based)")
        
        # הערכת ההמלצות
        evaluation_results = self.evaluate_recommendations()
        
        # שמירת תוצאות
        output_path = self.data_path / "datasets" / "ml_results"
        evaluation_df = pd.DataFrame(evaluation_results)
        evaluation_df.to_csv(output_path / "recommendation_evaluation.csv", index=False)
        
        print(f"\n" + "="*80)
        print("Phase 2 completed successfully!")
        print("="*80)
        print("Hybrid recommendation system is operational")
        print("Evaluation results saved")
        
        return evaluation_results

if __name__ == "__main__":
    rec_system = RecommendationSystem(r"C:\Users\Reuven\Desktop\ML")
    results = rec_system.run_phase2()



