"""
E-Commerce Recommendation System - Phase 1: Product Categorization
מערכת קטגוריזציה של מוצרים באמצעות Logistic Regression
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ProductCategorization:
    def __init__(self, data_path):
        """
        Initializes the ProductCategorization class
        
        Parameters:
        - data_path: Path to the data directory containing datasets
        """
        self.data_path = Path(data_path)
        self.products_df = None
        self.model = None
        self.tfidf_vectorizer = None
        self.price_scaler = None
        self.products_with_categories = None
        
    def load_data(self):
        """
        Loads product data from CSV file
        
        What it loads:
        - products_10000.csv: All products
        
        Returns:
        - None (data is stored in self.products_df)
        """
        print("Loading product data...")
        
        products_path = self.data_path / "datasets" / "raw" / "products_10000.csv"
        
        if not products_path.exists():
            raise FileNotFoundError(
                f"Product data file not found: {products_path}\n"
                f"Please ensure products_10000.csv is in: {self.data_path / 'datasets' / 'raw'}"
            )
        
        try:
            self.products_df = pd.read_csv(products_path)
            if self.products_df.empty:
                raise ValueError("products_10000.csv is empty")
            print(f"  Loaded {len(self.products_df)} products")
        except Exception as e:
            raise ValueError(f"Error loading products file: {e}")
    
    def clean_data(self):
        """
        Cleans the product data
        
        What it does:
        - Fills missing text fields with empty strings
        - Fills missing categories with 'Unknown'
        - Removes products with no name or description
        """
        print("\nCleaning product data...")
        
        # Fill missing text fields with empty strings
        self.products_df['product_name'] = self.products_df['product_name'].fillna('')
        self.products_df['description'] = self.products_df['description'].fillna('')
        
        # Fill missing categories with 'Unknown'
        self.products_df['main_category'] = self.products_df['main_category'].fillna('Unknown')
        self.products_df['sub_category'] = self.products_df['sub_category'].fillna('Unknown')
        
        # Remove completely empty products
        initial_size = len(self.products_df)
        self.products_df = self.products_df[~((self.products_df['product_name'].str.strip() == '') & 
                                               (self.products_df['description'].str.strip() == ''))]
        removed = initial_size - len(self.products_df)
        if removed > 0:
            print(f"  Removed {removed} empty products")
        print(f"  Final dataset size: {len(self.products_df)}")
    
    def prepare_features(self):
        """
        Prepares features for machine learning
        
        What it does:
        - Combines product_name and description into combined_text
        - Creates combined_category (main_category || sub_category)
        - Separates features (X) and target (y)
        
        Returns:
        - X_text: Text features (product names and descriptions)
        - X_price: Price features
        - y: Target categories
        """
        print("\nPreparing features for ML...")
        
        # Combine text features
        self.products_df['combined_text'] = (self.products_df['product_name'] + ' ' + 
                                            self.products_df['description'] + ' ' + 
                                            self.products_df['description'])
        
        # Create combined target feature
        self.products_df['combined_category'] = (self.products_df['main_category'] + ' || ' + 
                                                 self.products_df['sub_category'])
        
        # Separate features and target
        X_text = self.products_df['combined_text']
        X_price = self.products_df['price']
        y = self.products_df['combined_category']
        
        print(f"  Text features: {X_text.shape}")
        print(f"  Price features: {X_price.shape}")
        print(f"  Target categories: {y.nunique()} unique categories")
        
        return X_text, X_price, y
    
    def train_model(self):
        """
        Trains Logistic Regression model for product categorization
        
        What it does:
        1. Prepares features
        2. Splits data into train/test
        3. Converts text to numbers using TF-IDF
        4. Trains Logistic Regression model
        5. Evaluates the model
        
        Returns:
        - Dictionary with accuracy metrics
        """
        print("\n" + "="*60)
        print("Product Categorization - Logistic Regression")
        print("="*60)
        
        # Prepare features
        X_text, X_price, y = self.prepare_features()
        
        # Remove rare categories for stratification
        category_counts = y.value_counts()
        rare_categories = category_counts[category_counts < 2]
        if len(rare_categories) > 0:
            print(f"\nRemoving {len(rare_categories)} rare category combinations...")
            mask = ~y.isin(rare_categories.index)
            X_text = X_text[mask]
            X_price = X_price[mask]
            y = y[mask]
            self.products_df = self.products_df[mask].reset_index(drop=True)
        
        # Combine for splitting
        X_combined = pd.DataFrame({'text': X_text, 'price': X_price})
        
        # Split data
        print("\nSplitting data into train/test sets...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )
            print("  Successfully stratified by combined category")
        except ValueError:
            print("  Could not stratify by combined category, using main_category...")
            main_cat = self.products_df['main_category']
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y,
                test_size=0.20,
                random_state=42,
                stratify=main_cat
            )
        
        print(f"  Training set: {len(X_train)} products")
        print(f"  Test set: {len(X_test)} products")
        
        # Feature engineering - TF-IDF
        print("\nConverting text to numbers using TF-IDF...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1500,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.90,
            stop_words='english'
        )
        
        X_train_text_tfidf = self.tfidf_vectorizer.fit_transform(X_train['text']).toarray()
        X_test_text_tfidf = self.tfidf_vectorizer.transform(X_test['text']).toarray()
        
        # Scale price
        print("Scaling price feature...")
        self.price_scaler = StandardScaler()
        X_train_price_scaled = self.price_scaler.fit_transform(X_train['price'].values.reshape(-1, 1))
        X_test_price_scaled = self.price_scaler.transform(X_test['price'].values.reshape(-1, 1))
        
        # Combine features
        X_train_combined = np.hstack([X_train_text_tfidf, X_train_price_scaled])
        X_test_combined = np.hstack([X_test_text_tfidf, X_test_price_scaled])
        
        # Train model
        print("\nTraining Logistic Regression model...")
        self.model = LogisticRegression(
            max_iter=2000,
            multi_class='auto',
            n_jobs=-1
        )
        self.model.fit(X_train_combined, y_train)
        
        # Evaluate
        print("\nEvaluating model...")
        y_pred = self.model.predict(X_test_combined)
        accuracy_combined = accuracy_score(y_test, y_pred)
        
        # Split predictions for separate evaluation
        y_test_main = y_test.str.split(' || ').str[0]
        y_test_sub = y_test.str.split(' || ').str[1]
        y_pred_main = pd.Series(y_pred).str.split(' || ').str[0]
        y_pred_sub = pd.Series(y_pred).str.split(' || ').str[1]
        
        accuracy_main = accuracy_score(y_test_main, y_pred_main)
        accuracy_sub = accuracy_score(y_test_sub, y_pred_sub)
        
        print(f"\nFinal Metrics:")
        print(f"  Combined Category Accuracy: {accuracy_combined:.4f} ({accuracy_combined*100:.2f}%)")
        print(f"  Main Category Accuracy: {accuracy_main:.4f} ({accuracy_main*100:.2f}%)")
        print(f"  Sub Category Accuracy: {accuracy_sub:.4f} ({accuracy_sub*100:.2f}%)")
        
        return {
            'accuracy_combined': accuracy_combined,
            'accuracy_main': accuracy_main,
            'accuracy_sub': accuracy_sub
        }
    
    def categorize_all_products(self):
        """
        Categorizes all products using the trained model
        
        Returns:
        - DataFrame with products and predicted categories
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print("\nCategorizing all products...")
        
        # Prepare features for all products
        X_text = self.products_df['combined_text']
        X_price = self.products_df['price']
        
        # Transform
        X_text_tfidf = self.tfidf_vectorizer.transform(X_text).toarray()
        X_price_scaled = self.price_scaler.transform(X_price.values.reshape(-1, 1))
        X_combined = np.hstack([X_text_tfidf, X_price_scaled])
        
        # Predict
        y_pred = self.model.predict(X_combined)
        
        # Add predictions to DataFrame
        self.products_df['predicted_category'] = y_pred
        self.products_df['predicted_main_category'] = pd.Series(y_pred).str.split(' || ').str[0]
        self.products_df['predicted_sub_category'] = pd.Series(y_pred).str.split(' || ').str[1]
        
        self.products_with_categories = self.products_df.copy()
        
        print(f"  Categorized {len(self.products_df)} products")
        
        return self.products_with_categories
    
    def get_tfidf_matrix_for_descriptions(self, max_features=100):
        """
        Creates TF-IDF matrix for product descriptions (for content-based recommendations)
        
        This is separate from the categorization TF-IDF, optimized for recommendations:
        - Uses only descriptions (not combined text)
        - Fewer features for faster similarity calculations
        - Optimized parameters for recommendation use case
        
        Parameters:
        - max_features: Maximum number of features (default: 100)
        
        Returns:
        - TF-IDF matrix (sparse matrix) for product descriptions
        - TfidfVectorizer instance used
        """
        if self.products_df is None:
            raise ValueError("products_df is not loaded. Call load_data() first.")
        
        if 'description' not in self.products_df.columns:
            raise ValueError("products_df must contain 'description' column")
        
        print("Creating TF-IDF matrix for product descriptions (for recommendations)...")
        
        # Create a separate TF-IDF vectorizer optimized for recommendations
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        recommendation_tfidf = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # unigrams and bigrams
            min_df=2,  # word must appear in at least 2 products
            max_df=0.95  # word must not appear in more than 95% of products
        )
        
        # Fill missing descriptions with empty string
        descriptions = self.products_df['description'].fillna('')
        
        # Create TF-IDF matrix
        tfidf_matrix = recommendation_tfidf.fit_transform(descriptions)
        
        print(f"Created TF-IDF matrix: {tfidf_matrix.shape}")
        print(f"  - {tfidf_matrix.shape[0]} products")
        print(f"  - {tfidf_matrix.shape[1]} features (words/phrases)")
        
        return tfidf_matrix, recommendation_tfidf
    
    def save_results(self):
        """
        Saves product categorization results to CSV files
        
        Returns:
        - output_path: Path where results were saved
        """
        print("\nSaving product categorization results...")
        
        output_path = self.data_path / "datasets" / "results" / "phase1"
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.products_with_categories is None:
            raise ValueError("No categorized products. Call categorize_all_products() first.")
        
        # Save products with categories
        products_file = output_path / "products_with_categories.csv"
        self.products_with_categories.to_csv(products_file, index=False)
        print(f"  Saved: {products_file.name}")
        
        return output_path
    
    def run_product_categorization(self):
        """
        Runs the complete product categorization pipeline
        
        Returns:
        - Dictionary with results and metrics
        """
        print("="*80)
        print("Phase 1: Product Categorization")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Clean data
        self.clean_data()
        
        # Train model
        metrics = self.train_model()
        
        # Categorize all products
        products_with_categories = self.categorize_all_products()
        
        # Save results
        output_path = self.save_results()
        
        print(f"\n" + "="*80)
        print("Product Categorization completed successfully!")
        print("="*80)
        print(f"Products: {len(products_with_categories)} categorized")
        print(f"Accuracy: {metrics['accuracy_combined']:.4f} ({metrics['accuracy_combined']*100:.2f}%)")
        print(f"Results saved to: {output_path}")
        print("="*80)
        
        return {
            'products_with_categories': products_with_categories,
            'metrics': metrics,
            'output_path': output_path
        }

if __name__ == "__main__":
    import os
    # Get the project root directory (parent of src)
    project_root = Path(__file__).parent.parent.parent
    pc = ProductCategorization(str(project_root))
    results = pc.run_product_categorization()

