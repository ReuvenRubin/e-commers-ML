"""
Phase 2: מערכת המלצות היברידית עם Train/Test Split
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RecommendationSystemWithTrainTest:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        
        # נתונים
        self.products_df = None
        self.users_df = None
        self.clicks_df = None
        self.purchases_df = None
        self.visits_time_df = None
        self.product_metadata_df = None
        
        # Train/Test splits
        self.users_train = None
        self.users_test = None
        self.clicks_train = None
        self.clicks_test = None
        self.purchases_train = None
        self.purchases_test = None
        self.visits_time_train = None
        self.visits_time_test = None
        
        # תוצאות קטגוריזציה
        self.products_with_clusters = None
        self.users_with_clusters = None
        
        # מודלים (מאומנים על Train)
        self.tfidf_vectorizer = None
        self.product_tfidf_matrix = None
        self.user_similarity_matrix = None
        self.interaction_matrix = None
        
    def load_data(self):
        """
        טוען את הנתונים
        """
        print("Loading data for recommendation system...")
        
        # נתונים מקוריים - רק 500 מוצרים ראשונים
        products_all = pd.read_csv(self.data_path / "datasets/original/products_5000.csv")
        self.products_df = products_all.head(500).copy()
        self.users_df = pd.read_csv(self.data_path / "datasets/original/users_5000.csv")
        
        # טבלאות אינטראקציות - Long format
        self.clicks_df = pd.read_csv(self.data_path / "datasets/original/user_clicks_interactions_long.csv")
        self.purchases_df = pd.read_csv(self.data_path / "datasets/original/user_purchase_interactions_long.csv")
        self.visits_time_df = pd.read_csv(self.data_path / "datasets/original/user_visits_time_interactions_long.csv")
        self.product_metadata_df = pd.read_csv(self.data_path / "datasets/original/product_interaction_metadata_500.csv")
        
        # תוצאות קטגוריזציה
        self.products_with_clusters = pd.read_csv(self.data_path / "datasets/ml_results/products_with_clusters.csv")
        self.users_with_clusters = pd.read_csv(self.data_path / "datasets/ml_results/users_with_clusters.csv")
        
        print("Data loaded successfully!")
    
    def create_train_test_splits(self):
        """
        יוצר חלוקת train/test
        """
        print("\n" + "="*60)
        print("Creating Train/Test Split")
        print("="*60)
        
        # חלוקת משתמשים (80% train, 20% test)
        self.users_train, self.users_test = train_test_split(
            self.users_df,
            test_size=0.2,
            random_state=42
        )
        
        train_user_ids = set(self.users_train['id'])
        test_user_ids = set(self.users_test['id'])
        
        # חלוקת אינטראקציות לפי משתמשים (Long format)
        self.clicks_train = self.clicks_df[self.clicks_df['uid'].isin(train_user_ids)]
        self.clicks_test = self.clicks_df[self.clicks_df['uid'].isin(test_user_ids)]
        
        self.purchases_train = self.purchases_df[self.purchases_df['uid'].isin(train_user_ids)]
        self.purchases_test = self.purchases_df[self.purchases_df['uid'].isin(test_user_ids)]
        
        self.visits_time_train = self.visits_time_df[self.visits_time_df['uid'].isin(train_user_ids)]
        self.visits_time_test = self.visits_time_df[self.visits_time_df['uid'].isin(test_user_ids)]
        
        print(f"Users - Train: {len(self.users_train)}, Test: {len(self.users_test)}")
        print(f"Clicks - Train: {len(self.clicks_train)}, Test: {len(self.clicks_test)}")
        print(f"Purchases - Train: {len(self.purchases_train)}, Test: {len(self.purchases_test)}")
        print(f"Visits Time - Train: {len(self.visits_time_train)}, Test: {len(self.visits_time_test)}")
    
    def prepare_tfidf_for_products(self):
        """
        מכין TF-IDF לתיאורי מוצרים (על כל הנתונים)
        """
        print("Preparing TF-IDF for product descriptions...")
        
        # TF-IDF על תיאורי מוצרים
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.product_tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.products_df['description'].fillna('')
        )
        
        print(f"Created TF-IDF matrix: {self.product_tfidf_matrix.shape}")
    
    def create_user_interaction_matrix(self, use_train=True):
        """
        יוצר מטריצת אינטראקציות משתמש-מוצר (על Train או Test)
        """
        if use_train:
            clicks_df = self.clicks_train
            purchases_df = self.purchases_train
            visits_time_df = self.visits_time_train
            print("Creating interaction matrix (Train)...")
        else:
            clicks_df = self.clicks_test
            purchases_df = self.purchases_test
            visits_time_df = self.visits_time_test
            print("Creating interaction matrix (Test)...")
        
        # זיהוי כל המשתמשים והמוצרים הייחודיים
        all_user_ids = set(clicks_df['uid'].unique()) | set(purchases_df['uid'].unique()) | set(visits_time_df['uid'].unique())
        all_product_ids = set(clicks_df['product_id'].unique()) | set(purchases_df['product_id'].unique()) | set(visits_time_df['product_id'].unique())
        
        all_user_ids = sorted(list(all_user_ids))
        all_product_ids = sorted(list(all_product_ids))
        
        num_users = len(all_user_ids)
        num_products = len(all_product_ids)
        
        # יצירת מיפוי product_id -> index
        product_id_to_index = {pid: idx for idx, pid in enumerate(all_product_ids)}
        
        # יצירת מטריצת אינטראקציות משוקללת
        interaction_matrix = np.zeros((num_users, num_products))
        
        # מילוי מטריצה מקליקים
        for _, row in clicks_df.iterrows():
            user_idx = all_user_ids.index(row['uid'])
            product_idx = product_id_to_index[row['product_id']]
            interaction_matrix[user_idx, product_idx] += row['clicks'] * 1.0
        
        # מילוי מטריצה מרכישות
        for _, row in purchases_df.iterrows():
            user_idx = all_user_ids.index(row['uid'])
            product_idx = product_id_to_index[row['product_id']]
            interaction_matrix[user_idx, product_idx] += row['purchases'] * 5.0
        
        # מילוי מטריצה מזמן ביקור
        for _, row in visits_time_df.iterrows():
            user_idx = all_user_ids.index(row['uid'])
            product_idx = product_id_to_index[row['product_id']]
            interaction_matrix[user_idx, product_idx] += row['visit_time'] * 0.1
        
        # המרה ל-DataFrame
        interaction_matrix_df = pd.DataFrame(
            interaction_matrix,
            index=all_user_ids,
            columns=[f'product_{pid}' for pid in all_product_ids]
        )
        
        print(f"Created interaction matrix: {interaction_matrix_df.shape}")
        return interaction_matrix_df
    
    def calculate_user_similarity(self, interaction_matrix):
        """
        מחשב דמיון בין משתמשים
        """
        print("Calculating user similarity...")
        
        # נרמול המטריצה
        scaler = StandardScaler()
        normalized_matrix = scaler.fit_transform(interaction_matrix)
        
        # חישוב דמיון קוסינוס
        user_similarity_matrix = cosine_similarity(normalized_matrix)
        
        print(f"Created user similarity matrix: {user_similarity_matrix.shape}")
        return user_similarity_matrix
    
    def recommend_for_new_user(self, user_interactions):
        """
        המלצות למשתמש חדש - TF-IDF + Cosine Similarity
        """
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
        avg_vector = np.mean(product_vectors, axis=0)
        
        # חישוב דמיון לכל המוצרים
        similarities = cosine_similarity(avg_vector, self.product_tfidf_matrix).flatten()
        
        # דירוג והחזרת Top-K
        top_indices = np.argsort(similarities)[::-1][:5]
        recommendations = [idx + 1 for idx in top_indices if idx + 1 not in interested_products]
        
        return recommendations[:5]
    
    def recommend_for_old_user_collaborative(self, user_id, interaction_matrix, user_similarity_matrix, clicks_df, purchases_df, visits_time_df, n_recommendations=5):
        """
        המלצות למשתמש ותיק - Collaborative Filtering
        """
        if user_id not in interaction_matrix.index:
            # אם המשתמש לא קיים במטריצה, מטפלים בו כמשווק חדש
            user_interactions = {}
            user_clicks = clicks_df[clicks_df['uid'] == user_id]
            for _, row in user_clicks.iterrows():
                product_id = row['product_id']
                if product_id not in user_interactions:
                    user_interactions[product_id] = 0
                user_interactions[product_id] += row['clicks']
            user_purchases = purchases_df[purchases_df['uid'] == user_id]
            for _, row in user_purchases.iterrows():
                product_id = row['product_id']
                if product_id not in user_interactions:
                    user_interactions[product_id] = 0
                user_interactions[product_id] += row['purchases']
            user_visits = visits_time_df[visits_time_df['uid'] == user_id]
            for _, row in user_visits.iterrows():
                product_id = row['product_id']
                if product_id not in user_interactions:
                    user_interactions[product_id] = 0
                user_interactions[product_id] += row['visit_time']
            return self.recommend_for_new_user(user_interactions)
        
        # מציאת משתמשים דומים
        user_idx = interaction_matrix.index.get_loc(user_id)
        user_similarities = user_similarity_matrix[user_idx]
        
        # דירוג משתמשים לפי דמיון
        similar_users = np.argsort(user_similarities)[::-1][1:4]  # 3 המשתמשים הכי דומים
        
        # חישוב ציון לכל מוצר
        user_ratings = interaction_matrix.loc[user_id]
        recommendations = {}
        
        for similar_user_idx in similar_users:
            similar_user_id = interaction_matrix.index[similar_user_idx]
            similar_user_ratings = interaction_matrix.loc[similar_user_id]
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
    
    def recommend_for_old_user_content_based(self, user_id, clicks_df, purchases_df, n_recommendations=5):
        """
        המלצות למשתמש ותיק - Content-based Filtering
        """
        # מציאת מוצרים שהמשתמש כבר רכש/לחץ עליהם - Long format
        user_clicks = clicks_df[clicks_df['uid'] == user_id]
        user_purchases = purchases_df[purchases_df['uid'] == user_id]
        
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
                category = product_row.iloc[0]['category']
                user_categories.append(category)
        
        if not user_categories:
            return []
        
        # מציאת מוצרים דומים בקטגוריות המועדפות
        recommendations = []
        for category in set(user_categories):
            category_products = self.products_df[
                (self.products_df['category'] == category) & 
                (~self.products_df['id'].isin(interested_products))
            ]
            
            # דירוג לפי פופולריות
            category_products = category_products.sort_values('views', ascending=False)
            recommendations.extend(category_products['id'].head(2).tolist())
        
        return recommendations[:n_recommendations]
    
    def hybrid_recommendations(self, user_id, interaction_matrix, user_similarity_matrix, clicks_df, purchases_df, visits_time_df, n_recommendations=5):
        """
        המלצות היברידיות - שילוב של Collaborative ו-Content-based
        """
        # בדיקה אם המשתמש חדש או ותיק - Long format
        user_interactions = {}
        
        # איסוף אינטראקציות מקליקים
        user_clicks = clicks_df[clicks_df['uid'] == user_id]
        for _, row in user_clicks.iterrows():
            product_id = row['product_id']
            if product_id not in user_interactions:
                user_interactions[product_id] = 0
            user_interactions[product_id] += row['clicks']
        
        # איסוף אינטראקציות מרכישות
        user_purchases = purchases_df[purchases_df['uid'] == user_id]
        for _, row in user_purchases.iterrows():
            product_id = row['product_id']
            if product_id not in user_interactions:
                user_interactions[product_id] = 0
            user_interactions[product_id] += row['purchases']
        
        # איסוף אינטראקציות מזמן ביקור
        user_visits = visits_time_df[visits_time_df['uid'] == user_id]
        for _, row in user_visits.iterrows():
            product_id = row['product_id']
            if product_id not in user_interactions:
                user_interactions[product_id] = 0
            user_interactions[product_id] += row['visit_time']
        
        # ספירת אינטראקציות
        total_interactions = sum(user_interactions.values())
        
        if total_interactions < 3:  # משתמש חדש
            return self.recommend_for_new_user(user_interactions)
        else:  # משתמש ותיק
            # המלצות Collaborative
            cf_recommendations = self.recommend_for_old_user_collaborative(
                user_id, interaction_matrix, user_similarity_matrix, 
                clicks_df, purchases_df, visits_time_df, n_recommendations
            )
            
            # המלצות Content-based
            cb_recommendations = self.recommend_for_old_user_content_based(
                user_id, clicks_df, purchases_df, n_recommendations
            )
            
            # שילוב ההמלצות
            hybrid_recs = {}
            
            # הוספת המלצות Collaborative עם משקל גבוה
            for i, product_id in enumerate(cf_recommendations):
                hybrid_recs[product_id] = (len(cf_recommendations) - i) * 0.7
            
            # הוספת המלצות Content-based עם משקל נמוך יותר
            for i, product_id in enumerate(cb_recommendations):
                if product_id not in hybrid_recs:
                    hybrid_recs[product_id] = (len(cb_recommendations) - i) * 0.3
                else:
                    hybrid_recs[product_id] += (len(cb_recommendations) - i) * 0.3
            
            # דירוג סופי
            sorted_hybrid = sorted(hybrid_recs.items(), key=lambda x: x[1], reverse=True)
            
            return [rec[0] for rec in sorted_hybrid[:n_recommendations]]
    
    def evaluate_on_test_set(self):
        """
        מעריך את המודל על Test Set
        """
        print("\n" + "="*80)
        print("Evaluating on Test Set")
        print("="*80)
        
        # יצירת מטריצת אינטראקציות ל-Test
        test_interaction_matrix = self.create_user_interaction_matrix(use_train=False)
        
        # חישוב דמיון משתמשים ל-Test
        test_user_similarity_matrix = self.calculate_user_similarity(test_interaction_matrix)
        
        # בחירת משתמשים לבדיקה (רק משתמשים עם אינטראקציות)
        test_user_ids = set(self.clicks_test['uid'].unique()) | set(self.purchases_test['uid'].unique()) | set(self.visits_time_test['uid'].unique())
        test_user_ids = sorted(list(test_user_ids))[:20]  # 20 משתמשים ראשונים לבדיקה
        
        results = []
        
        for user_id in test_user_ids:
            # המלצות
            recommendations = self.hybrid_recommendations(
                user_id, test_interaction_matrix, test_user_similarity_matrix,
                self.clicks_test, self.purchases_test, self.visits_time_test, 5
            )
            
            # בדיקת רלוונטיות - Long format
            user_purchases = self.purchases_test[self.purchases_test['uid'] == user_id]
            purchased_products = user_purchases[user_purchases['purchases'] > 0]['product_id'].tolist()
            
            # חישוב Precision@5 - בדיקה לפי קטגוריות
            recommended_categories = []
            for rec_id in recommendations:
                product_row = self.products_df[self.products_df['id'] == rec_id]
                if len(product_row) > 0:
                    recommended_categories.append(product_row.iloc[0]['category'])
            
            purchased_categories = []
            for pur_id in purchased_products:
                product_row = self.products_df[self.products_df['id'] == pur_id]
                if len(product_row) > 0:
                    purchased_categories.append(product_row.iloc[0]['category'])
            
            # Precision = כמה מההמלצות באותן קטגוריות כמו הרכישות
            relevant_recommendations = len(set(recommended_categories) & set(purchased_categories))
            precision = relevant_recommendations / len(set(recommended_categories)) if recommended_categories else 0
            
            # Recall = כמה מהרכישות נמצאו בהמלצות
            recall = relevant_recommendations / len(set(purchased_categories)) if purchased_categories else 0
            
            # F1 Score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'user_id': user_id,
                'recommendations': recommendations,
                'recommended_categories': list(set(recommended_categories)),
                'purchased_products': purchased_products[:10],
                'purchased_categories': list(set(purchased_categories)),
                'precision@5': precision,
                'recall@5': recall,
                'f1@5': f1
            })
        
        # סיכום
        if results:
            avg_precision = np.mean([r['precision@5'] for r in results])
            avg_recall = np.mean([r['recall@5'] for r in results])
            avg_f1 = np.mean([r['f1@5'] for r in results])
            
            print(f"\n" + "="*80)
            print("Evaluation Summary")
            print("="*80)
            print(f"Users tested: {len(results)}")
            print(f"Average Precision@5: {avg_precision:.3f}")
            print(f"Average Recall@5: {avg_recall:.3f}")
            print(f"Average F1@5: {avg_f1:.3f}")
        
        return results
    
    def run_phase2_with_train_test(self):
        """
        מריץ את Phase 2 עם Train/Test Split
        """
        print("="*80)
        print("Phase 2: Hybrid Recommendation System with Train/Test Split")
        print("="*80)
        
        # טעינת נתונים
        self.load_data()
        
        # יצירת חלוקת train/test
        self.create_train_test_splits()
        
        # הכנת מודלים על Train
        print("\n" + "="*80)
        print("Training Models on Train Set")
        print("="*80)
        
        self.prepare_tfidf_for_products()
        self.interaction_matrix = self.create_user_interaction_matrix(use_train=True)
        self.user_similarity_matrix = self.calculate_user_similarity(self.interaction_matrix)
        
        # הערכה על Test Set
        evaluation_results = self.evaluate_on_test_set()
        
        # שמירת תוצאות
        output_path = self.data_path / "datasets" / "ml_results"
        evaluation_df = pd.DataFrame(evaluation_results)
        evaluation_df.to_csv(output_path / "recommendation_evaluation_train_test.csv", index=False)
        
        print(f"\n" + "="*80)
        print("Phase 2 with Train/Test completed successfully!")
        print("="*80)
        print("Hybrid recommendation system evaluated on test set")
        print("Evaluation results saved")
        
        return evaluation_results

if __name__ == "__main__":
    rec_system = RecommendationSystemWithTrainTest(r"C:\Users\Reuven\Desktop\ML")
    results = rec_system.run_phase2_with_train_test()

