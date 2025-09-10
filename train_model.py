import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from urllib.parse import urlparse
import tldextract

# --- 1. Feature Extraction Function ---
def extract_features(url):
    """Extracts features from a single URL and returns them as a dictionary."""
    features = {}
    
    # Parse the URL to get its components
    parsed_url = urlparse(url)
    domain_info = tldextract.extract(url)
    
    # URL-based features
    features['url_length'] = len(url)
    features['hostname_length'] = len(parsed_url.netloc)
    features['path_length'] = len(parsed_url.path)
    features['fd_length'] = len(parsed_url.path.split('/')[-1])
    features['count-'] = url.count('-')
    features['count@'] = url.count('@')
    features['count?'] = url.count('?')
    features['count%'] = url.count('%')
    features['count.'] = url.count('.')
    features['count='] = url.count('=')
    features['count-http'] = url.count('http')
    features['count-https'] = url.count('https')
    features['count-www'] = url.count('www')
    features['digits_count'] = sum(c.isdigit() for c in url)
    features['letters_count'] = sum(c.isalpha() for c in url)
    
    # Domain-based features
    features['is_ip'] = 1 if domain_info.domain.replace('.', '').isdigit() else 0
    features['abnormal_subdomain'] = 1 if url.count('.') > 3 and "www" not in url.lower() else 0
    
    return features

print("Starting model training process...")

# --- 2. Load and Prepare Data ---
print("Loading dataset...")
# Make sure your CSV file is named 'phishing_data.csv'
df = pd.read_csv('phishing_data.csv')

print("Extracting features from URLs... This may take a while.")
# Using the lowercase 'url' column from your file
features_list = df['url'].apply(extract_features)
features_df = pd.DataFrame(features_list.tolist())

# Assign features (X) and labels (y)
X = features_df
# Using the lowercase 'label' column from your file
y = df['label']

# --- 3. Train the Machine Learning Model ---
print("Splitting data and training the model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
# n_jobs=-1 uses all available CPU cores to speed up training
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# --- 4. Evaluate the Model ---
print("Evaluating the model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# --- 5. Save the Trained Model ---
print("Saving the trained model to 'phishing_model.pkl'...")
joblib.dump(model, 'phishing_model.pkl')

print("Model training and saving complete! 🎉")