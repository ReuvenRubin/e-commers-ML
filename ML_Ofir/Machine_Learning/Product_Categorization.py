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

#print("\nCreating combined target feature (main_category - sub_category)...")
#df['combined_category'] = df['main_category'] + ' - ' + df['sub_category']

# Separate features (X) and target (y)
# X = what we use to predict (features)
# y = what we want to predict (category)
print("\nSeparating features and target...")
X_text = df['combined_text']  # Text feature
X_price = df['price']  # Numeric feature
y = df['main_category']  # Target variable (what we want to predict)

#y = df['combined_category']  # Target variable (what we want to predict)

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

# Split: 85% for training, 15% for testing
# stratify=y means each category appears in same proportion in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, 
    test_size=0.20,        # 15% for testing
    random_state=21,      # Makes split reproducible (same split every time)
    stratify=y            # Keeps same category distribution in train and test
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
    ngram_range=(1, 3),   # Use unigrams AND bigrams for richer context
    min_df=2,             # Word must appear in at least 2 products (filters typos)
    max_df=0.90,          # Word must not appear in more than 95% of products (filters common words)
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
    max_iter=1000,        # allow more iterations so it can converge
    multi_class='auto',   # automatically handle multi-class
    n_jobs=-1             # use all CPU cores if available
)

# Train the model on training data
print("Training the model...")
model.fit(X_train_combined, y_train)

# Evaluate on the test set
print("\nEvaluating on the test set...")
y_pred = model.predict(X_test_combined)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred))