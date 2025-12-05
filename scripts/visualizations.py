"""
Visualizations for E-Commerce Recommendation System
ויזואליזציות למערכת ההמלצות
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# הגדרת עברית
plt.rcParams['font.family'] = 'Arial Unicode MS'  # או כל פונט שתומך בעברית
sns.set_style("whitegrid")

class Visualizations:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.output_path = self.data_path / "datasets" / "ml_results"
        self.output_path.mkdir(exist_ok=True)
    
    def visualize_product_clusters(self):
        """מציג גרף של אשכולות מוצרים"""
        try:
            products_df = pd.read_csv(self.output_path / "products_with_clusters.csv")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Product Clusters Analysis', fontsize=16, fontweight='bold')
            
            # 1. התפלגות אשכולות
            cluster_counts = products_df['ml_cluster'].value_counts().sort_index()
            axes[0, 0].bar(cluster_counts.index, cluster_counts.values, color='skyblue')
            axes[0, 0].set_title('Number of Products per Cluster')
            axes[0, 0].set_xlabel('Cluster')
            axes[0, 0].set_ylabel('Number of Products')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. מחיר ממוצע לפי אשכול
            price_by_cluster = products_df.groupby('ml_cluster')['price'].mean()
            axes[0, 1].bar(price_by_cluster.index, price_by_cluster.values, color='lightgreen')
            axes[0, 1].set_title('Average Price by Cluster')
            axes[0, 1].set_xlabel('Cluster')
            axes[0, 1].set_ylabel('Average Price')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. צפיות ממוצעות לפי אשכול
            views_by_cluster = products_df.groupby('ml_cluster')['views'].mean()
            axes[1, 0].bar(views_by_cluster.index, views_by_cluster.values, color='coral')
            axes[1, 0].set_title('Average Views by Cluster')
            axes[1, 0].set_xlabel('Cluster')
            axes[1, 0].set_ylabel('Average Views')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. קטגוריות לפי אשכול
            category_by_cluster = products_df.groupby('ml_cluster')['category'].apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown')
            axes[1, 1].bar(range(len(category_by_cluster)), range(len(category_by_cluster)), color='plum')
            axes[1, 1].set_title('Main Category by Cluster')
            axes[1, 1].set_xlabel('Cluster')
            axes[1, 1].set_ylabel('Category Index')
            axes[1, 1].set_xticks(range(len(category_by_cluster)))
            axes[1, 1].set_xticklabels(category_by_cluster.values, rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_path / "product_clusters_analysis.png", dpi=300, bbox_inches='tight')
            print(f"Product clusters visualization saved to: {self.output_path / 'product_clusters_analysis.png'}")
            plt.close()
        except Exception as e:
            print(f"Error creating product clusters visualization: {e}")
    
    def visualize_user_clusters(self):
        """מציג גרף של אשכולות משתמשים"""
        try:
            users_df = pd.read_csv(self.output_path / "users_with_clusters.csv")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('User Clusters Analysis', fontsize=16, fontweight='bold')
            
            # 1. התפלגות אשכולות
            cluster_counts = users_df['cluster'].value_counts().sort_index()
            axes[0, 0].bar(cluster_counts.index.astype(str), cluster_counts.values, color='skyblue')
            axes[0, 0].set_title('Number of Users per Cluster')
            axes[0, 0].set_xlabel('Cluster')
            axes[0, 0].set_ylabel('Number of Users')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. קליקים ממוצעים לפי אשכול
            if 'total_clicks' in users_df.columns:
                clicks_by_cluster = users_df.groupby('cluster')['total_clicks'].mean()
                axes[0, 1].bar(clicks_by_cluster.index.astype(str), clicks_by_cluster.values, color='lightgreen')
                axes[0, 1].set_title('Average Clicks by Cluster')
                axes[0, 1].set_xlabel('Cluster')
                axes[0, 1].set_ylabel('Average Clicks')
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. רכישות ממוצעות לפי אשכול
            if 'total_purchases' in users_df.columns:
                purchases_by_cluster = users_df.groupby('cluster')['total_purchases'].mean()
                axes[1, 0].bar(purchases_by_cluster.index.astype(str), purchases_by_cluster.values, color='coral')
                axes[1, 0].set_title('Average Purchases by Cluster')
                axes[1, 0].set_xlabel('Cluster')
                axes[1, 0].set_ylabel('Average Purchases')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. שיעור המרה ממוצע לפי אשכול
            if 'conversion_rate' in users_df.columns:
                conversion_by_cluster = users_df.groupby('cluster')['conversion_rate'].mean()
                axes[1, 1].bar(conversion_by_cluster.index.astype(str), conversion_by_cluster.values, color='plum')
                axes[1, 1].set_title('Average Conversion Rate by Cluster')
                axes[1, 1].set_xlabel('Cluster')
                axes[1, 1].set_ylabel('Average Conversion Rate')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_path / "user_clusters_analysis.png", dpi=300, bbox_inches='tight')
            print(f"User clusters visualization saved to: {self.output_path / 'user_clusters_analysis.png'}")
            plt.close()
        except Exception as e:
            print(f"Error creating user clusters visualization: {e}")
    
    def visualize_recommendation_evaluation(self):
        """מציג גרף של הערכת ההמלצות"""
        try:
            eval_df = pd.read_csv(self.output_path / "recommendation_evaluation.csv")
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Recommendation System Evaluation', fontsize=16, fontweight='bold')
            
            # 1. Precision@3 לפי משתמש
            axes[0].bar(range(len(eval_df)), eval_df['precision@3'], color='skyblue')
            axes[0].set_title('Precision@3 by User')
            axes[0].set_xlabel('User Index')
            axes[0].set_ylabel('Precision@3')
            axes[0].axhline(y=eval_df['precision@3'].mean(), color='r', linestyle='--', label=f'Average: {eval_df["precision@3"].mean():.2f}')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 2. התפלגות Precision@3
            axes[1].hist(eval_df['precision@3'], bins=10, color='lightgreen', edgecolor='black')
            axes[1].set_title('Precision@3 Distribution')
            axes[1].set_xlabel('Precision@3')
            axes[1].set_ylabel('Frequency')
            axes[1].axvline(x=eval_df['precision@3'].mean(), color='r', linestyle='--', label=f'Mean: {eval_df["precision@3"].mean():.2f}')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_path / "recommendation_evaluation.png", dpi=300, bbox_inches='tight')
            print(f"Recommendation evaluation visualization saved to: {self.output_path / 'recommendation_evaluation.png'}")
            plt.close()
        except Exception as e:
            print(f"Error creating recommendation evaluation visualization: {e}")
    
    def visualize_train_test_comparison(self):
        """מציג השוואה בין Train ו-Test"""
        try:
            # נסה לטעון תוצאות train/test
            train_products = pd.read_csv(self.output_path / "products_train_with_clusters.csv")
            test_products = pd.read_csv(self.output_path / "products_test_with_clusters.csv")
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Train vs Test Comparison', fontsize=16, fontweight='bold')
            
            # 1. התפלגות אשכולות - Train vs Test
            train_clusters = train_products['ml_cluster'].value_counts().sort_index()
            test_clusters = test_products['ml_cluster'].value_counts().sort_index()
            
            x = np.arange(len(train_clusters))
            width = 0.35
            
            axes[0].bar(x - width/2, train_clusters.values, width, label='Train', color='skyblue')
            axes[0].bar(x + width/2, test_clusters.values, width, label='Test', color='coral')
            axes[0].set_title('Cluster Distribution: Train vs Test')
            axes[0].set_xlabel('Cluster')
            axes[0].set_ylabel('Number of Products')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(train_clusters.index)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 2. מחיר ממוצע - Train vs Test
            train_price = train_products['price'].mean()
            test_price = test_products['price'].mean()
            
            axes[1].bar(['Train', 'Test'], [train_price, test_price], color=['skyblue', 'coral'])
            axes[1].set_title('Average Price: Train vs Test')
            axes[1].set_ylabel('Average Price')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_path / "train_test_comparison.png", dpi=300, bbox_inches='tight')
            print(f"Train/Test comparison visualization saved to: {self.output_path / 'train_test_comparison.png'}")
            plt.close()
        except Exception as e:
            print(f"Error creating train/test comparison visualization: {e}")
    
    def create_all_visualizations(self):
        """יוצר את כל הויזואליזציות"""
        print("="*80)
        print("Creating Visualizations")
        print("="*80)
        
        self.visualize_product_clusters()
        self.visualize_user_clusters()
        self.visualize_recommendation_evaluation()
        self.visualize_train_test_comparison()
        
        print("\n" + "="*80)
        print("All visualizations created successfully!")
        print("="*80)

if __name__ == "__main__":
    viz = Visualizations(r"C:\Users\Reuven\Desktop\ML")
    viz.create_all_visualizations()

