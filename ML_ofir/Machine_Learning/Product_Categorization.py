import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load data - resolve path relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '..', 'datasets', 'products_10000.csv')
df = pd.read_csv(csv_path)

# ============================================================================
# STEP 1: SEE WHAT WE HAVE
# ============================================================================
print("\n" + "="*50)
print("Step 1: Understanding our data")
print("="*50)

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# ============================================================================
# STEP 2: LOOK AT THE FEATURES WE'LL USE
# ============================================================================
print("\n" + "="*50)
print("Step 2: Looking at features for ML")
print("="*50)

# Show what products look like
print("\nSample of product data:")
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Don't limit width
pd.set_option('display.max_colwidth', 80)  # Limit each column width to 80 chars
print(df[['product_name', 'description', 'price', 'main_category', 'sub_category']].head())

# Check for missing values - ML models don't like missing data!
print("\nMissing values in our features:")
columns_to_check = ['product_name', 'description', 'price', 'main_category', 'sub_category']
for col in columns_to_check:
    missing_count = df[col].isnull().sum()
    print(f"  {col}: {missing_count} missing")

# ============================================================================
# STEP 3: CLEAN THE DATA
# ============================================================================
print("\n" + "="*50)
print("Step 3: Cleaning the data")
print("="*50)

# Fill missing text fields with empty strings
print("\nFilling missing text fields...")
df['product_name'] = df['product_name'].fillna('')
df['description'] = df['description'].fillna('')

# Fill missing categories with 'Unknown'
print("\nFilling missing categories...")
df['main_category'] = df['main_category'].fillna('Unknown')
df['sub_category'] = df['sub_category'].fillna('Unknown')

# Check if we have any completely empty products (both name and description empty)
empty_products = df[(df['product_name'].str.strip() == '') & 
                    (df['description'].str.strip() == '')]
print(f"Found {len(empty_products)} products with no name or description")

# Remove completely empty products
initial_size = len(df)
df = df[~((df['product_name'].str.strip() == '') & 
          (df['description'].str.strip() == ''))]
print(f"Removed {initial_size - len(df)} empty products")
print(f"Final dataset size: {len(df)}")

# ============================================================================
# STEP 4: PREPARE FEATURES FOR MACHINE LEARNING
# ============================================================================
print("\n" + "="*50)
print("Step 4: Preparing features for ML")
print("="*50)

# Combine text features - ML models work better with combined text
print("\nCombining product_name and description into one text field...")
df['combined_text'] = df['product_name'] + ' ' + df['description']
print(f"Example combined text: {df['combined_text'].iloc[0][:100]}...")

print("\nCreating combined target feature (main_category || sub_category)...")
# Use || as separator since main_category already contains dashes (e.g., "Food - Beverages")
df['combined_category'] = df['main_category'] + ' || ' + df['sub_category']
print(f"Example combined category: {df['combined_category'].iloc[0][:100]}...")

# Separate features (X) and target (y)
# X = what we use to predict (features)
# y = what we want to predict (category)
print("\nSeparating features and target...")
X_text = df['combined_text']  # Text feature
X_price = df['price']  # Numeric feature
y = df['combined_category']  # Target variable (what we want to predict)

print(f"X_text shape: {X_text.shape}")
print(f"X_price shape: {X_price.shape}")
print(f"y shape: {y.shape}")
print(f"\nTarget categories: {y.unique()}")

# ============================================================================
# STEP 5: SPLIT DATA INTO TRAIN AND TEST SETS
# ============================================================================
print("\n" + "="*50)
print("Step 5: Splitting data into train and test sets")
print("="*50)

# Why split? 
# - Train set: Used to TEACH the model (model learns patterns from this)
# - Test set: Used to TEST the model (model has never seen this data)
# - This tells us if the model can predict NEW products it hasn't seen before

# Combine X_text and X_price for splitting
# We need to split them together so train/test have matching rows
X_combined = pd.DataFrame({
    'text': X_text,
    'price': X_price
})

# Split: 80% for training, 20% for testing
# Try to stratify by combined category, but if some categories are too rare, stratify by main_category
print("\nChecking category distribution for stratification...")
category_counts = y.value_counts()
rare_categories = category_counts[category_counts < 2]

if len(rare_categories) > 0:
    print(f"Warning: {len(rare_categories)} category combinations have less than 2 samples.")
    print("These will be removed to allow proper stratification.")
    # Remove products with rare category combinations
    df_filtered = df[~df['combined_category'].isin(rare_categories.index)]
    print(f"Removed {len(df) - len(df_filtered)} products with rare categories")
    df = df_filtered
    # Recreate X and y after filtering
    X_text = df['combined_text']
    X_price = df['price']
    y = df['combined_category']
    X_combined = pd.DataFrame({'text': X_text, 'price': X_price})

# Try stratify by combined category, but fall back to main_category if it fails
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, 
        test_size=0.20,
        random_state=21,
        stratify=y  # Try to stratify by combined category
    )
    print("Successfully stratified by combined category")

except ValueError as e:
    print(f"Could not stratify by combined category: {e}")
    print("Stratifying by main_category instead...")
    # Fall back to stratifying by main_category only
    main_cat_for_split = df['main_category']
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y,
        test_size=0.20,
        random_state=42,
        stratify=main_cat_for_split  # Stratify by main category only
    )

print(f"\nTraining set: {len(X_train)} products")
print(f"Test set: {len(X_test)} products")
print(f"\nTraining set categories distribution:")
print(y_train.value_counts())
print(f"\nTest set categories distribution:")
print(y_test.value_counts())

# Separate text and price again for easier use
X_train_text = X_train['text']
X_train_price = X_train['price']
X_test_text = X_test['text']
X_test_price = X_test['price']

# ============================================================================
# STEP 6: FEATURE ENGINEERING - CONVERT TEXT TO NUMBERS
# ============================================================================
print("\n" + "="*50)
print("Step 6: Feature Engineering - Converting text to numbers")
print("="*50)

# ML models need numbers, not text. TF-IDF converts text into numerical features
print("\nCreating TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,    # Only use the top 1000 most important words
    ngram_range=(1, 2),   # Use unigrams AND bigrams for richer context
    min_df=2,             # Word must appear in at least 2 products (filters typos)
    max_df=0.95,          # Word must not appear in more than 95% of products (filters common words)
    stop_words='english'  # Remove common English words like "the", "a", "and"
)

# Fit TF-IDF on training data - learn vocabulary from training data only
print("Fitting TF-IDF on training data...")
X_train_text_tfidf = tfidf_vectorizer.fit_transform(X_train_text)

# Transform test data - use the same vocabulary learned from training
print("Transforming test data...")
X_test_text_tfidf = tfidf_vectorizer.transform(X_test_text)

print(f"\nText features shape (sparse matrix):")
print(f"  Training: {X_train_text_tfidf.shape}")
print(f"  Test: {X_test_text_tfidf.shape}")

# Convert from sparse matrix to dense array
# Sparse matrix only stores non-zero values, dense array stores all values
# We convert to dense so we can combine with price features later
X_train_text_tfidf = X_train_text_tfidf.toarray()
X_test_text_tfidf = X_test_text_tfidf.toarray()

print(f"\nAfter converting to dense arrays:")
print(f"  Training: {X_train_text_tfidf.shape}")
print(f"  Test: {X_test_text_tfidf.shape}")

# ============================================================================
# STEP 7: COMBINE TEXT AND PRICE FEATURES
# ============================================================================
print("\n" + "="*50)
print("Step 7: Combining text and price features")
print("="*50)

# Scale the price feature so it matches the TF-IDF scale
print("\nScaling price feature...")
price_scaler = StandardScaler()
X_train_price_scaled = price_scaler.fit_transform(X_train_price.values.reshape(-1, 1))
X_test_price_scaled = price_scaler.transform(X_test_price.values.reshape(-1, 1))
print(f"  Training price shape (scaled): {X_train_price_scaled.shape}")
print(f"  Test price shape (scaled): {X_test_price_scaled.shape}")

# Combine text features (1000 columns) with price (1 column)
print("\nCombining text and price features...")
X_train_combined = np.hstack([X_train_text_tfidf, X_train_price_scaled])
X_test_combined = np.hstack([X_test_text_tfidf, X_test_price_scaled])

print(f"\nFinal feature shapes:")
print(f"  Training: {X_train_combined.shape}")
print(f"  Test: {X_test_combined.shape}")

# ============================================================================
# STEP 8: TRAIN A CLASSIFICATION MODEL
# ============================================================================
print("\n" + "="*50)
print("Step 8: Training a classification model")
print("="*50)

# Create the model
print("\nCreating Logistic Regression model...")
model = LogisticRegression(
    max_iter=5000,        # allow more iterations so it can converge
    multi_class='auto',   # automatically handle multi-class
    n_jobs=-1             # use all CPU cores if available
)

# Train the model on training data
print("Training the model...")
model.fit(X_train_combined, y_train)

# Evaluate on the test set
print("\nEvaluating on the test set...")
y_pred = model.predict(X_test_combined)

# Calculate accuracy for combined category
accuracy_combined = accuracy_score(y_test, y_pred)
print(f"\nCombined Category Accuracy (main - sub): {accuracy_combined:.4f}")

# Split predictions back to main and sub categories for separate evaluation
# Use || as separator (same as when combining)
y_test_main = y_test.str.split(' || ').str[0]
y_test_sub = y_test.str.split(' || ').str[1]
y_pred_main = pd.Series(y_pred).str.split(' || ').str[0]
y_pred_sub = pd.Series(y_pred).str.split(' || ').str[1]

# Calculate accuracy for main category only
accuracy_main = accuracy_score(y_test_main, y_pred_main)
print(f"Main Category Accuracy: {accuracy_main:.4f}")

# Calculate accuracy for sub category only
accuracy_sub = accuracy_score(y_test_sub, y_pred_sub)
print(f"Sub Category Accuracy: {accuracy_sub:.4f}")

print("\n" + "="*50)
print("Combined Category Classification Report:")
print("="*50)
print(classification_report(y_test, y_pred))