import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ECommerceAnalyzer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.original_data = None
        self.integrated_data = None
        self.wide_format_data = None
        
    def load_all_datasets(self):
        """
        טוען את כל הנתונים
        """
        print("טוען את כל הנתונים...")
        
        # טעינת נתונים מקוריים - רק 500 מוצרים ראשונים
        original_path = self.data_path / "datasets" / "original"
        all_products = pd.read_csv(original_path / "products_5000.csv")
        self.original_data = {
            'users': pd.read_csv(original_path / "users_5000.csv"),
            'products': all_products.head(500).copy(),
            'purchases': pd.read_csv(original_path / "user_purchase_history.csv"),
            'wishlist': pd.read_csv(original_path / "user_wishlist.csv"),
            'search_history': pd.read_csv(original_path / "user_search_history.csv")
        }
        
        # טעינת נתונים משולבים
        integrated_path = self.data_path / "datasets" / "integrated"
        self.integrated_data = {
            'interactions': pd.read_csv(integrated_path / "user_interactions.csv"),
            'interaction_matrix': pd.read_csv(integrated_path / "interaction_matrix.csv", index_col=0),
            'enhanced_products': pd.read_csv(integrated_path / "enhanced_products.csv")
        }
        
        # טעינת נתונים בפורמט wide
        wide_path = self.data_path / "datasets" / "wide"
        self.wide_format_data = {
            'clicks_wide': pd.read_csv(wide_path / "clicks_wide.csv"),
            'purchases_wide': pd.read_csv(wide_path / "purchases_wide.csv"),
            'visits_time_wide': pd.read_csv(wide_path / "visits_time_wide.csv"),
            'product_metadata': pd.read_csv(wide_path / "product_metadata.csv"),
            'interactions_long': pd.read_csv(wide_path / "interactions_long.csv"),
            'interaction_matrix_weighted': pd.read_csv(wide_path / "interaction_matrix_weighted.csv", index_col=0)
        }
        
        print("כל הנתונים נטענו בהצלחה!")
        
    def analyze_user_behavior_patterns(self):
        """
        מנתח דפוסי התנהגות משתמשים
        """
        print("\n=== ניתוח דפוסי התנהגות משתמשים ===")
        
        # ניתוח אינטראקציות
        interactions = self.integrated_data['interactions']
        
        # סטטיסטיקות כלליות
        print(f"סה\"כ אינטראקציות: {len(interactions)}")
        print(f"מספר משתמשים ייחודיים: {interactions['user_id'].nunique()}")
        print(f"מספר מוצרים ייחודיים: {interactions['product_id'].nunique()}")
        
        # ניתוח לפי סוג אינטראקציה
        interaction_counts = interactions['interaction_type'].value_counts()
        print("\nהתפלגות סוגי אינטראקציות:")
        for interaction_type, count in interaction_counts.items():
            percentage = (count / len(interactions)) * 100
            print(f"  {interaction_type}: {count} ({percentage:.1f}%)")
        
        # ניתוח משתמשים פעילים
        user_activity = interactions.groupby('user_id').size().sort_values(ascending=False)
        print(f"\nהמשתמש הפעיל ביותר: {user_activity.index[0]} עם {user_activity.iloc[0]} אינטראקציות")
        print(f"המשתמש הפחות פעיל: {user_activity.index[-1]} עם {user_activity.iloc[-1]} אינטראקציות")
        
        return user_activity
    
    def analyze_product_popularity(self):
        """
        מנתח פופולריות מוצרים
        """
        print("\n=== ניתוח פופולריות מוצרים ===")
        
        # ניתוח מטבלת המוצרים המשופרת
        products = self.integrated_data['enhanced_products']
        
        # מוצרים פופולריים ביותר
        top_products = products.nlargest(10, 'views')[['id', 'product_name', 'views', 'category', 'popularity_score']]
        print("10 המוצרים הפופולריים ביותר (לפי צפיות):")
        for _, product in top_products.iterrows():
            print(f"  {product['id']}: {product['product_name']} - {product['views']} צפיות ({product['category']})")
        
        # ניתוח לפי קטגוריות
        category_stats = products.groupby('category').agg({
            'views': 'sum',
            'price': 'mean',
            'popularity_score': 'mean'
        }).round(2)
        
        print("\nסטטיסטיקות לפי קטגוריות:")
        print(category_stats)
        
        return top_products, category_stats
    
    def analyze_conversion_rates(self):
        """
        מנתח שיעורי המרה
        """
        print("\n=== ניתוח שיעורי המרה ===")
        
        # ניתוח מטבלת המוצרים בפורמט wide
        clicks = self.wide_format_data['clicks_wide']
        purchases = self.wide_format_data['purchases_wide']
        
        # חישוב שיעורי המרה לכל מוצר
        conversion_rates = {}
        
        for col in clicks.columns:
            if col.startswith('pid'):
                product_id = col.replace('pid', '')
                total_clicks = clicks[col].sum()
                total_purchases = purchases[col].sum()
                
                if total_clicks > 0:
                    conversion_rate = (total_purchases / total_clicks) * 100
                    conversion_rates[product_id] = {
                        'clicks': total_clicks,
                        'purchases': total_purchases,
                        'conversion_rate': conversion_rate
                    }
        
        # דירוג לפי שיעור המרה
        sorted_conversions = sorted(conversion_rates.items(), 
                                  key=lambda x: x[1]['conversion_rate'], 
                                  reverse=True)
        
        print("שיעורי המרה לפי מוצר:")
        for product_id, stats in sorted_conversions:
            print(f"  מוצר {product_id}: {stats['conversion_rate']:.1f}% ({stats['purchases']}/{stats['clicks']})")
        
        return conversion_rates
    
    def analyze_user_segments(self):
        """
        מנתח קטעי משתמשים
        """
        print("\n=== ניתוח קטעי משתמשים ===")
        
        # חישוב מדדי פעילות לכל משתמש
        interactions = self.integrated_data['interactions']
        user_stats = interactions.groupby('user_id').agg({
            'value': ['sum', 'count', 'mean'],
            'interaction_type': lambda x: x.value_counts().to_dict()
        }).round(2)
        
        # פישוט העמודות
        user_stats.columns = ['total_value', 'interaction_count', 'avg_value', 'interaction_types']
        
        # קטעי משתמשים
        high_activity = user_stats[user_stats['interaction_count'] >= 5]
        medium_activity = user_stats[(user_stats['interaction_count'] >= 2) & (user_stats['interaction_count'] < 5)]
        low_activity = user_stats[user_stats['interaction_count'] < 2]
        
        print(f"משתמשים פעילים מאוד (5+ אינטראקציות): {len(high_activity)}")
        print(f"משתמשים פעילים בינונית (2-4 אינטראקציות): {len(medium_activity)}")
        print(f"משתמשים פעילים מעט (פחות מ-2 אינטראקציות): {len(low_activity)}")
        
        return user_stats
    
    def generate_insights(self):
        """
        מפיק תובנות עסקיות
        """
        print("\n=== תובנות עסקיות ===")
        
        # ניתוח כל הנתונים
        user_activity = self.analyze_user_behavior_patterns()
        top_products, category_stats = self.analyze_product_popularity()
        conversion_rates = self.analyze_conversion_rates()
        user_segments = self.analyze_user_segments()
        
        # תובנות מרכזיות
        print("\nתובנות מרכזיות:")
        
        # 1. מוצר עם שיעור המרה הגבוה ביותר
        best_conversion = max(conversion_rates.items(), key=lambda x: x[1]['conversion_rate'])
        print(f"1. מוצר {best_conversion[0]} יש שיעור המרה הגבוה ביותר: {best_conversion[1]['conversion_rate']:.1f}%")
        
        # 2. קטגוריה הפופולרית ביותר
        most_popular_category = category_stats.loc[category_stats['views'].idxmax()]
        print(f"2. הקטגוריה הפופולרית ביותר: {most_popular_category.name} עם {most_popular_category['views']} צפיות")
        
        # 3. משתמש הפעיל ביותר
        most_active_user = user_activity.index[0]
        print(f"3. המשתמש הפעיל ביותר: {most_active_user} עם {user_activity.iloc[0]} אינטראקציות")
        
        # 4. המלצות עסקיות
        print("\nהמלצות עסקיות:")
        print("• התמקד במוצרים עם שיעור המרה גבוה")
        print("• שפר את החשיפה לקטגוריות הפופולריות")
        print("• צור תוכניות נאמנות למשתמשים הפעילים")
        print("• שפר את חוויית המשתמש למוצרים עם שיעור המרה נמוך")
        
        return {
            'user_activity': user_activity,
            'top_products': top_products,
            'category_stats': category_stats,
            'conversion_rates': conversion_rates,
            'user_segments': user_segments
        }
    
    def create_summary_report(self):
        """
        יוצר דוח סיכום
        """
        print("\n=== דוח סיכום ===")
        
        # סטטיסטיקות כלליות
        total_users = len(self.original_data['users'])
        total_products = len(self.original_data['products'])
        total_interactions = len(self.integrated_data['interactions'])
        
        print(f"סה\"כ משתמשים במסד: {total_users}")
        print(f"סה\"כ מוצרים במסד: {total_products}")
        print(f"סה\"כ אינטראקציות: {total_interactions}")
        
        # ניתוח קטגוריות
        categories = self.original_data['products']['category'].value_counts()
        print(f"\nהתפלגות קטגוריות:")
        for category, count in categories.head(5).items():
            print(f"  {category}: {count} מוצרים")
        
        # ניתוח מחירים
        price_stats = self.original_data['products']['price'].describe()
        print(f"\nסטטיסטיקות מחירים:")
        print(f"  ממוצע: ₪{price_stats['mean']:.2f}")
        print(f"  חציון: ₪{price_stats['50%']:.2f}")
        print(f"  מינימום: ₪{price_stats['min']:.2f}")
        print(f"  מקסימום: ₪{price_stats['max']:.2f}")
        
    def run_full_analysis(self):
        """
        מריץ ניתוח מלא
        """
        print("=== ניתוח מקיף של מסד הנתונים E-Commerce ===")
        
        # טעינת נתונים
        self.load_all_datasets()
        
        # יצירת דוח סיכום
        self.create_summary_report()
        
        # הפקת תובנות
        insights = self.generate_insights()
        
        print("\nהניתוח הושלם בהצלחה!")
        
        return insights

if __name__ == "__main__":
    # הפעלת הניתוח המקיף
    analyzer = ECommerceAnalyzer(r"C:\Users\Reuven\Desktop\ML")
    insights = analyzer.run_full_analysis()
