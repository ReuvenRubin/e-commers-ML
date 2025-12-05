"""
Phase 3: NLP לחיפוש מתקדם
לפי המסמך של המרצה
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class NLPSearchSystem:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        
        # נתונים
        self.products_df = None
        self.product_metadata_df = None
        
        # מודלים
        self.tfidf_vectorizer = None
        self.product_tfidf_matrix = None
        self.product_features_matrix = None
        
    def load_data(self):
        """
        טוען את הנתונים
        """
        print("טוען נתונים למערכת NLP...")
        
        # נתונים מקוריים - רק 500 מוצרים ראשונים
        all_products = pd.read_csv(self.data_path / "datasets/original/products_5000.csv")
        self.products_df = all_products.head(500).copy()
        self.product_metadata_df = pd.read_csv(self.data_path / "datasets/original/product_interaction_metadata_500.csv")
        
        print(f"נטענו {len(self.products_df)} מוצרים")
        
    def preprocess_text(self, text):
        """
        עיבוד טקסט לחיפוש
        """
        if pd.isna(text):
            return ""
        
        # המרה לאותיות קטנות
        text = str(text).lower()
        
        # הסרת תווים מיוחדים
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # הסרת מילים קצרות
        words = text.split()
        words = [word for word in words if len(word) > 2]
        
        return ' '.join(words)
    
    def prepare_search_features(self):
        """
        מכין תכונות לחיפוש
        """
        print("מכין תכונות לחיפוש...")
        
        # עיבוד טקסט
        processed_descriptions = self.products_df['description'].apply(self.preprocess_text)
        processed_names = self.products_df['product_name'].apply(self.preprocess_text)
        processed_categories = self.products_df['category'].apply(self.preprocess_text)
        
        # שילוב טקסטים
        combined_text = processed_names + " " + processed_descriptions + " " + processed_categories
        
        # TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words='english',
            ngram_range=(1, 2)  # unigrams ו-bigrams
        )
        
        self.product_tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_text)
        
        print(f"נוצרה מטריצת TF-IDF: {self.product_tfidf_matrix.shape}")
        
        # יצירת מטריצת תכונות נוספת (מחיר, קטגוריה, וכו')
        self.product_features_matrix = np.column_stack([
            self.products_df['price'].values,
            self.products_df['views'].values,
            pd.Categorical(self.products_df['category']).codes
        ])
        
        # נרמול תכונות
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.product_features_matrix = scaler.fit_transform(self.product_features_matrix)
        
        print(f"נוצרה מטריצת תכונות: {self.product_features_matrix.shape}")
        
    def cosine_similarity_search(self, query, top_k=5):
        """
        חיפוש עם Cosine Similarity
        """
        print(f"חיפוש עם Cosine Similarity: '{query}'")
        
        # עיבוד השאילתה
        processed_query = self.preprocess_text(query)
        
        # המרה ל-TF-IDF
        query_tfidf = self.tfidf_vectorizer.transform([processed_query])
        
        # חישוב דמיון
        similarities = cosine_similarity(query_tfidf, self.product_tfidf_matrix).flatten()
        
        # דירוג תוצאות
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # רק תוצאות עם דמיון > 0
                results.append({
                    'product_id': self.products_df.iloc[idx]['id'],
                    'product_name': self.products_df.iloc[idx]['product_name'],
                    'category': self.products_df.iloc[idx]['category'],
                    'price': self.products_df.iloc[idx]['price'],
                    'similarity_score': similarities[idx],
                    'method': 'Cosine Similarity'
                })
        
        return results
    
    def pearson_correlation_search(self, query, top_k=5):
        """
        חיפוש עם Pearson Correlation
        """
        print(f"חיפוש עם Pearson Correlation: '{query}'")
        
        # עיבוד השאילתה
        processed_query = self.preprocess_text(query)
        query_tfidf = self.tfidf_vectorizer.transform([processed_query]).toarray().flatten()
        
        # חישוב קורלציה לכל מוצר
        correlations = []
        for i in range(self.product_tfidf_matrix.shape[0]):
            product_vector = self.product_tfidf_matrix[i].toarray().flatten()
            if np.std(product_vector) > 0 and np.std(query_tfidf) > 0:
                corr, _ = pearsonr(query_tfidf, product_vector)
                correlations.append(corr if not np.isnan(corr) else 0)
            else:
                correlations.append(0)
        
        correlations = np.array(correlations)
        
        # דירוג תוצאות
        top_indices = np.argsort(correlations)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if correlations[idx] > 0:
                results.append({
                    'product_id': self.products_df.iloc[idx]['id'],
                    'product_name': self.products_df.iloc[idx]['product_name'],
                    'category': self.products_df.iloc[idx]['category'],
                    'price': self.products_df.iloc[idx]['price'],
                    'similarity_score': correlations[idx],
                    'method': 'Pearson Correlation'
                })
        
        return results
    
    def adjusted_cosine_similarity_search(self, query, top_k=5):
        """
        חיפוש עם Adjusted Cosine Similarity
        """
        print(f"חיפוש עם Adjusted Cosine Similarity: '{query}'")
        
        # עיבוד השאילתה
        processed_query = self.preprocess_text(query)
        query_tfidf = self.tfidf_vectorizer.transform([processed_query]).toarray().flatten()
        
        # חישוב ממוצע של כל המוצרים
        mean_vector = np.mean(self.product_tfidf_matrix.toarray(), axis=0)
        
        # Adjusted Cosine Similarity
        similarities = []
        for i in range(self.product_tfidf_matrix.shape[0]):
            product_vector = self.product_tfidf_matrix[i].toarray().flatten()
            
            # חישוב adjusted vectors
            adjusted_query = query_tfidf - mean_vector
            adjusted_product = product_vector - mean_vector
            
            # cosine similarity
            if np.linalg.norm(adjusted_query) > 0 and np.linalg.norm(adjusted_product) > 0:
                similarity = np.dot(adjusted_query, adjusted_product) / (
                    np.linalg.norm(adjusted_query) * np.linalg.norm(adjusted_product)
                )
                similarities.append(similarity)
            else:
                similarities.append(0)
        
        similarities = np.array(similarities)
        
        # דירוג תוצאות
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    'product_id': self.products_df.iloc[idx]['id'],
                    'product_name': self.products_df.iloc[idx]['product_name'],
                    'category': self.products_df.iloc[idx]['category'],
                    'price': self.products_df.iloc[idx]['price'],
                    'similarity_score': similarities[idx],
                    'method': 'Adjusted Cosine Similarity'
                })
        
        return results
    
    def hybrid_search(self, query, top_k=5):
        """
        חיפוש היברידי - שילוב של כל השיטות
        """
        print(f"חיפוש היברידי: '{query}'")
        
        # חיפוש עם כל השיטות
        cosine_results = self.cosine_similarity_search(query, top_k*2)
        pearson_results = self.pearson_correlation_search(query, top_k*2)
        adjusted_results = self.adjusted_cosine_similarity_search(query, top_k*2)
        
        # שילוב תוצאות
        all_results = {}
        
        # הוספת תוצאות עם משקלים
        for result in cosine_results:
            product_id = result['product_id']
            if product_id not in all_results:
                all_results[product_id] = {
                    'product_id': product_id,
                    'product_name': result['product_name'],
                    'category': result['category'],
                    'price': result['price'],
                    'combined_score': 0
                }
            all_results[product_id]['combined_score'] += result['similarity_score'] * 0.5
        
        for result in pearson_results:
            product_id = result['product_id']
            if product_id not in all_results:
                all_results[product_id] = {
                    'product_id': product_id,
                    'product_name': result['product_name'],
                    'category': result['category'],
                    'price': result['price'],
                    'combined_score': 0
                }
            all_results[product_id]['combined_score'] += result['similarity_score'] * 0.3
        
        for result in adjusted_results:
            product_id = result['product_id']
            if product_id not in all_results:
                all_results[product_id] = {
                    'product_id': product_id,
                    'product_name': result['product_name'],
                    'category': result['category'],
                    'price': result['price'],
                    'combined_score': 0
                }
            all_results[product_id]['combined_score'] += result['similarity_score'] * 0.2
        
        # דירוג לפי ציון משולב
        sorted_results = sorted(all_results.values(), key=lambda x: x['combined_score'], reverse=True)
        
        return sorted_results[:top_k]
    
    def evaluate_search_methods(self):
        """
        הערכת שיטות החיפוש
        """
        print("\nמעריך שיטות חיפוש...")
        
        # שאילתות בדיקה
        test_queries = [
            "electronics gadget",
            "clothing fashion",
            "book education",
            "sports fitness",
            "automotive car"
        ]
        
        results = []
        
        for query in test_queries:
            print(f"\nבדיקת שאילתה: '{query}'")
            
            # חיפוש עם כל השיטות
            cosine_results = self.cosine_similarity_search(query, 3)
            pearson_results = self.pearson_correlation_search(query, 3)
            adjusted_results = self.adjusted_cosine_similarity_search(query, 3)
            hybrid_results = self.hybrid_search(query, 3)
            
            results.append({
                'query': query,
                'cosine_results': cosine_results,
                'pearson_results': pearson_results,
                'adjusted_results': adjusted_results,
                'hybrid_results': hybrid_results
            })
            
            # הצגת תוצאות
            print("Cosine Similarity:")
            for result in cosine_results:
                print(f"  {result['product_name']} ({result['category']}) - {result['similarity_score']:.3f}")
            
            print("Pearson Correlation:")
            for result in pearson_results:
                print(f"  {result['product_name']} ({result['category']}) - {result['similarity_score']:.3f}")
            
            print("Adjusted Cosine Similarity:")
            for result in adjusted_results:
                print(f"  {result['product_name']} ({result['category']}) - {result['similarity_score']:.3f}")
            
            print("Hybrid Search:")
            for result in hybrid_results:
                print(f"  {result['product_name']} ({result['category']}) - {result['combined_score']:.3f}")
        
        return results
    
    def run_phase3(self):
        """
        מריץ את Phase 3 - NLP לחיפוש
        """
        print("="*80)
        print("Phase 3: NLP לחיפוש מתקדם")
        print("="*80)
        
        # טעינת נתונים
        self.load_data()
        
        # הכנת תכונות
        self.prepare_search_features()
        
        # הערכת שיטות חיפוש
        evaluation_results = self.evaluate_search_methods()
        
        # שמירת תוצאות
        output_path = self.data_path / "datasets" / "ml_results"
        
        # המרת תוצאות לפורמט JSON
        import json
        with open(output_path / "nlp_search_evaluation.json", 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n" + "="*80)
        print("Phase 3 הושלם בהצלחה!")
        print("="*80)
        print("מערכת NLP לחיפוש פועלת")
        print("תוצאות הערכה נשמרו")
        
        return evaluation_results

if __name__ == "__main__":
    nlp_system = NLPSearchSystem(r"C:\Users\Reuven\Desktop\ML")
    results = nlp_system.run_phase3()



