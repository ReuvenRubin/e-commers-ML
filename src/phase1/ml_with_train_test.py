"""
ML Implementation with proper Train/Test split
מערכת למידת מכונה עם חלוקת train/test נכונה
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MLWithTrainTest:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.products_df = None
        self.users_df = None
        self.clicks_df = None
        self.purchases_df = None
        self.visits_time_df = None
        self.product_metadata_df = None
        
        # Train/Test splits
        self.products_train = None
        self.products_test = None
        self.users_train = None
        self.users_test = None
        
        # Models
        self.product_kmeans = None
        self.user_dbscan = None
        self.user_kmeans = None
        self.tfidf_vectorizer = None
        
        # Results
        self.product_clusters = None
        self.user_clusters = None
        
    def load_data(self):
        """
        טוען את הנתונים הנדרשים
        """
        print("="*80)
        print("Loading data for ML system with Train/Test")
        print("="*80)
        
        # טעינת נתונים מקוריים - רק 500 מוצרים ראשונים
        all_products = pd.read_csv(self.data_path / "datasets/original/products_5000.csv")
        self.products_df = all_products.head(500).copy()
        self.users_df = pd.read_csv(self.data_path / "datasets/original/users_5000.csv")
        
        # טעינת טבלאות האינטראקציות בפורמט Long
        self.clicks_df = pd.read_csv(self.data_path / "datasets/original/user_clicks_interactions_long.csv")
        self.purchases_df = pd.read_csv(self.data_path / "datasets/original/user_purchase_interactions_long.csv")
        self.visits_time_df = pd.read_csv(self.data_path / "datasets/original/user_visits_time_interactions_long.csv")
        self.product_metadata_df = pd.read_csv(self.data_path / "datasets/original/product_interaction_metadata_500.csv")
        
        print(f"Loaded {len(self.products_df)} products and {len(self.users_df)} users")
        print(f"Loaded {len(self.clicks_df)} click interactions, {len(self.purchases_df)} purchase interactions, {len(self.visits_time_df)} visit time interactions")
        
    def create_train_test_splits(self):
        """
        יוצר חלוקת train/test
        """
        print("\n" + "="*60)
        print("Creating Train/Test Split")
        print("="*60)
        
        # חלוקת מוצרים (80% train, 20% test)
        self.products_train, self.products_test = train_test_split(
            self.products_df, 
            test_size=0.2, 
            random_state=42,
            stratify=self.products_df['category']  # שמירה על פרופורציות קטגוריות
        )
        
        # חלוקת משתמשים (80% train, 20% test)
        self.users_train, self.users_test = train_test_split(
            self.users_df, 
            test_size=0.2, 
            random_state=42
        )
        
        print(f"Products - Train: {len(self.products_train)}, Test: {len(self.products_test)}")
        print(f"Users - Train: {len(self.users_train)}, Test: {len(self.users_test)}")
        
        # חלוקת אינטראקציות לפי משתמשים (Long format)
        train_user_ids = set(self.users_train['id'])
        test_user_ids = set(self.users_test['id'])
        
        self.clicks_train = self.clicks_df[self.clicks_df['uid'].isin(train_user_ids)]
        self.clicks_test = self.clicks_df[self.clicks_df['uid'].isin(test_user_ids)]
        
        self.purchases_train = self.purchases_df[self.purchases_df['uid'].isin(train_user_ids)]
        self.purchases_test = self.purchases_df[self.purchases_df['uid'].isin(test_user_ids)]
        
        # גם visits_time (אם קיים)
        self.visits_time_train = self.visits_time_df[self.visits_time_df['uid'].isin(train_user_ids)]
        self.visits_time_test = self.visits_time_df[self.visits_time_df['uid'].isin(test_user_ids)]
        
        print(f"Interactions - Train: {len(self.clicks_train)}, Test: {len(self.clicks_test)}")
        
    def prepare_product_features(self, data_type='train'):
        """
        מכין תכונות למוצרים
        """
        if data_type == 'train':
            products = self.products_train.copy()
        else:
            products = self.products_test.copy()
        
        # נרמול תכונות מספריות
        scaler = StandardScaler()
        numeric_features = ['price', 'quantity', 'views']
        products[numeric_features] = scaler.fit_transform(products[numeric_features])
        
        # קידוד קטגוריות
        le = LabelEncoder()
        products['category_encoded'] = le.fit_transform(products['category'])
        
        # TF-IDF על תיאורי מוצרים
        if data_type == 'train':
            self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            description_tfidf = self.tfidf_vectorizer.fit_transform(products['description'].fillna(''))
        else:
            # השתמש ב-vectorizer שהוכשר על train
            description_tfidf = self.tfidf_vectorizer.transform(products['description'].fillna(''))
        
        # המרה ל-DataFrame
        tfidf_df = pd.DataFrame(description_tfidf.toarray(), 
                               columns=[f'tfidf_{i}' for i in range(description_tfidf.shape[1])])
        
        # שילוב תכונות
        feature_columns = numeric_features + ['category_encoded']
        features = pd.concat([
            products[feature_columns].reset_index(drop=True),
            tfidf_df.reset_index(drop=True)
        ], axis=1)
        
        print(f"Created {features.shape[1]} product features ({data_type})")
        return features, products
    
    def train_product_categorization(self):
        """
        מאמן מודל קטגוריזציית מוצרים על train set
        """
        print("\n" + "="*60)
        print("Training Product Categorization - K-means")
        print("="*60)
        
        # הכנת תכונות train
        train_features, train_products = self.prepare_product_features('train')
        
        # מציאת מספר אשכולות אופטימלי
        print("Finding optimal number of clusters...")
        inertias = []
        silhouette_scores = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(train_features)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(train_features, cluster_labels))
        
        # בחירת מספר אשכולות אופטימלי
        optimal_k = 5
        print(f"Selected {optimal_k} clusters")
        
        # אימון המודל
        self.product_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        train_cluster_labels = self.product_kmeans.fit_predict(train_features)
        
        # הוספת אשכולות למוצרי train
        train_products['ml_cluster'] = train_cluster_labels
        
        # חישוב מדדי איכות
        silhouette_avg = silhouette_score(train_features, train_cluster_labels)
        
        print(f"Silhouette Score (Train): {silhouette_avg:.3f}")
        
        # ניתוח אשכולות
        cluster_analysis = train_products.groupby('ml_cluster').agg({
            'price': 'mean',
            'quantity': 'mean',
            'views': 'mean',
            'category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
            'id': 'count'
        }).round(2)
        
        cluster_analysis.columns = ['Avg_Price', 'Avg_Quantity', 'Avg_Views', 'Main_Category', 'Count']
        print("\nCluster Analysis (Train):")
        print(cluster_analysis)
        
        return train_products, silhouette_avg
    
    def test_product_categorization(self):
        """
        בודק מודל קטגוריזציית מוצרים על test set
        """
        print("\n" + "="*60)
        print("Testing Product Categorization - Test Set")
        print("="*60)
        
        # הכנת תכונות test
        test_features, test_products = self.prepare_product_features('test')
        
        # חיזוי על test set
        test_cluster_labels = self.product_kmeans.predict(test_features)
        test_products['ml_cluster'] = test_cluster_labels
        
        # חישוב מדדי איכות
        silhouette_avg = silhouette_score(test_features, test_cluster_labels)
        
        print(f"Silhouette Score (Test): {silhouette_avg:.3f}")
        
        # ניתוח אשכולות
        cluster_analysis = test_products.groupby('ml_cluster').agg({
            'price': 'mean',
            'quantity': 'mean',
            'views': 'mean',
            'category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
            'id': 'count'
        }).round(2)
        
        cluster_analysis.columns = ['Avg_Price', 'Avg_Quantity', 'Avg_Views', 'Main_Category', 'Count']
        print("\nCluster Analysis (Test):")
        print(cluster_analysis)
        
        return test_products, silhouette_avg
    
    def prepare_user_features(self, data_type='train'):
        """
        מכין תכונות למשתמשים
        """
        if data_type == 'train':
            users = self.users_train.copy()
            clicks = self.clicks_train
            purchases = self.purchases_train
        else:
            users = self.users_test.copy()
            clicks = self.clicks_test
            purchases = self.purchases_test
        
        # חישוב תכונות משתמשים
        user_features = []
        
        # קבלת product IDs (500 מוצרים)
        product_ids = set(self.products_df['id'].tolist())
        num_products = len(product_ids)
        
        for _, user in users.iterrows():
            user_id = user['id']
            
            # חישוב אינטראקציות (Long format)
            user_clicks = clicks[clicks['uid'] == user_id]
            user_purchases = purchases[purchases['uid'] == user_id]
            
            # סכום קליקים
            total_clicks = user_clicks['clicks'].sum() if not user_clicks.empty else 0
            
            # סכום רכישות
            total_purchases = user_purchases['purchases'].sum() if not user_purchases.empty else 0
            
            # חישוב זמן ביקור (Long format)
            if data_type == 'train':
                visits_time = self.visits_time_train[self.visits_time_train['uid'] == user_id]
            else:
                visits_time = self.visits_time_test[self.visits_time_test['uid'] == user_id]
            total_visit_time = visits_time['visit_time'].sum() if not visits_time.empty else 0
            
            # חישוב מוצרים ייחודיים
            user_products = set()
            if not user_clicks.empty:
                user_products.update(user_clicks['product_id'].unique())
            if not user_purchases.empty:
                user_products.update(user_purchases['product_id'].unique())
            unique_products = len([p for p in user_products if p in product_ids])
            
            # חישוב conversion rate
            conversion_rate = total_purchases / total_clicks if total_clicks > 0 else 0
            
            # חישוב category diversity - נרמול לפי מספר מוצרים (500)
            category_diversity = unique_products / num_products if num_products > 0 else 0
            
            user_features.append({
                'user_id': user_id,
                'total_clicks': total_clicks,
                'total_purchases': total_purchases,
                'total_visit_time': total_visit_time,
                'unique_products': unique_products,
                'conversion_rate': conversion_rate,
                'category_diversity': category_diversity
            })
        
        features_df = pd.DataFrame(user_features)
        
        # נרמול תכונות
        scaler = StandardScaler()
        feature_columns = ['total_clicks', 'total_purchases', 'total_visit_time', 
                          'unique_products', 'conversion_rate', 'category_diversity']
        features_df[feature_columns] = scaler.fit_transform(features_df[feature_columns])
        
        print(f"Created {features_df.shape[1]} user features ({data_type})")
        return features_df, users
    
    def train_user_categorization(self):
        """
        מאמן מודל קטגוריזציית משתמשים על train set
        """
        print("\n" + "="*60)
        print("Training User Categorization - DBSCAN -> K-means")
        print("="*60)
        
        # הכנת תכונות train
        train_features, train_users = self.prepare_user_features('train')
        
        # שלב 1: DBSCAN לזיהוי outliers
        print("Step 1: DBSCAN...")
        self.user_dbscan = DBSCAN(eps=0.5, min_samples=2)
        dbscan_labels = self.user_dbscan.fit_predict(train_features[['total_clicks', 'total_purchases', 'total_visit_time', 
                                                                    'unique_products', 'conversion_rate', 'category_diversity']])
        
        print(f"DBSCAN: {len(set(dbscan_labels))} clusters, {sum(dbscan_labels == -1)} noise points")
        
        # שלב 2: K-means על נקודות שאינן outliers
        print("Step 2: K-means on non-outlier points...")
        non_outlier_mask = dbscan_labels != -1
        non_outlier_features = train_features[non_outlier_mask]
        
        if len(non_outlier_features) > 4:  # צריך לפחות 5 נקודות ל-K-means
            n_clusters = min(4, len(non_outlier_features) - 1)  # לא יותר מנקודות-1
            self.user_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_labels = self.user_kmeans.fit_predict(non_outlier_features[['total_clicks', 'total_purchases', 'total_visit_time', 
                                                                              'unique_products', 'conversion_rate', 'category_diversity']])
            
            # שילוב תוויות
            final_labels = dbscan_labels.copy()
            final_labels[non_outlier_mask] = kmeans_labels
            
            # חישוב מדדי איכות
            if len(set(kmeans_labels)) > 1:  # רק אם יש יותר מאשכול אחד
                silhouette_avg = silhouette_score(non_outlier_features[['total_clicks', 'total_purchases', 'total_visit_time', 
                                                                       'unique_products', 'conversion_rate', 'category_diversity']], kmeans_labels)
            else:
                silhouette_avg = 0
            
            print(f"Silhouette Score (Train): {silhouette_avg:.3f}")
            
            # הוספת אשכולות למשתמשי train
            train_users['cluster'] = final_labels
            
            # ניתוח אשכולות
            cluster_analysis = train_users.groupby('cluster').agg({
                'id': 'count'
            })
            cluster_analysis.columns = ['Count']
            print("\nUser Cluster Analysis (Train):")
            print(cluster_analysis)
            
            return train_users, silhouette_avg
        else:
            print("Not enough non-outlier points to train K-means")
            # השתמש רק ב-DBSCAN
            train_users['cluster'] = dbscan_labels
            return train_users, 0
    
    def test_user_categorization(self):
        """
        בודק מודל קטגוריזציית משתמשים על test set
        """
        print("\n" + "="*60)
        print("Testing User Categorization - Test Set")
        print("="*60)
        
        # הכנת תכונות test
        test_features, test_users = self.prepare_user_features('test')
        
        # חיזוי על test set
        if self.user_kmeans is not None:
            test_cluster_labels = self.user_kmeans.predict(test_features[['total_clicks', 'total_purchases', 'total_visit_time', 
                                                                         'unique_products', 'conversion_rate', 'category_diversity']])
            test_users['cluster'] = test_cluster_labels
            
            # חישוב מדדי איכות
            if len(set(test_cluster_labels)) > 1:  # רק אם יש יותר מאשכול אחד
                silhouette_avg = silhouette_score(test_features[['total_clicks', 'total_purchases', 'total_visit_time', 
                                                               'unique_products', 'conversion_rate', 'category_diversity']], test_cluster_labels)
            else:
                silhouette_avg = 0
            
            print(f"Silhouette Score (Test): {silhouette_avg:.3f}")
            
            # ניתוח אשכולות
            cluster_analysis = test_users.groupby('cluster').agg({
                'id': 'count'
            })
            cluster_analysis.columns = ['Count']
            print("\nUser Cluster Analysis (Test):")
            print(cluster_analysis)
            
            return test_users, silhouette_avg
        else:
            print("K-means model not trained")
            # השתמש ב-DBSCAN על test set
            test_dbscan = DBSCAN(eps=0.5, min_samples=2)
            test_dbscan_labels = test_dbscan.fit_predict(test_features[['total_clicks', 'total_purchases', 'total_visit_time', 
                                                                       'unique_products', 'conversion_rate', 'category_diversity']])
            test_users['cluster'] = test_dbscan_labels
            return test_users, 0
    
    def run_ml_pipeline(self):
        """
        מריץ את כל pipeline של למידת המכונה
        """
        print("="*80)
        print("ML System with Train/Test Split")
        print("="*80)
        
        # טעינת נתונים
        self.load_data()
        
        # יצירת חלוקת train/test
        self.create_train_test_splits()
        
        # אימון ובדיקה של קטגוריזציית מוצרים
        print("\n" + "="*80)
        print("Product Categorization")
        print("="*80)
        
        train_products, train_silhouette = self.train_product_categorization()
        test_products, test_silhouette = self.test_product_categorization()
        
        # אימון ובדיקה של קטגוריזציית משתמשים
        print("\n" + "="*80)
        print("User Categorization")
        print("="*80)
        
        train_users, train_user_silhouette = self.train_user_categorization()
        test_users, test_user_silhouette = self.test_user_categorization()
        
        # שמירת תוצאות
        self.save_results(train_products, test_products, train_users, test_users)
        
        # סיכום
        print("\n" + "="*80)
        print("Results Summary")
        print("="*80)
        print(f"Products - Train Silhouette: {train_silhouette:.3f}, Test Silhouette: {test_silhouette:.3f}")
        print(f"Users - Train Silhouette: {train_user_silhouette:.3f}, Test Silhouette: {test_user_silhouette:.3f}")
        
        return {
            'train_products': train_products,
            'test_products': test_products,
            'train_users': train_users,
            'test_users': test_users,
            'product_silhouette': {'train': train_silhouette, 'test': test_silhouette},
            'user_silhouette': {'train': train_user_silhouette, 'test': test_user_silhouette}
        }
    
    def save_results(self, train_products, test_products, train_users, test_users):
        """
        שומר את התוצאות
        """
        output_path = self.data_path / "datasets" / "results"
        output_path.mkdir(exist_ok=True)
        
        # שמירת תוצאות מוצרים
        train_products.to_csv(output_path / "products_train_with_clusters.csv", index=False)
        test_products.to_csv(output_path / "products_test_with_clusters.csv", index=False)
        
        # שמירת תוצאות משתמשים
        train_users.to_csv(output_path / "users_train_with_clusters.csv", index=False)
        test_users.to_csv(output_path / "users_test_with_clusters.csv", index=False)
        
        print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    ml_system = MLWithTrainTest(r"C:\Users\Reuven\Desktop\ML")
    results = ml_system.run_ml_pipeline()
