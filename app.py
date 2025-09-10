from flask import Flask, request, render_template
import joblib
import pandas as pd
from urllib.parse import urlparse
import tldextract

app = Flask(__name__)

# --- Load the Trained Model ---
# The model expects features in a specific order. Let's define it.
FEATURE_NAMES = [
    'url_length', 'hostname_length', 'path_length', 'fd_length', 'count-', 'count@', 
    'count?', 'count%', 'count.', 'count=', 'count-http', 'count-https', 'count-www', 
    'digits_count', 'letters_count', 'is_ip', 'abnormal_subdomain'
]

# Load the model from the file
model = joblib.load('phishing_model.pkl')

# --- Feature Extraction Function (Must be IDENTICAL to the one in train_model.py) ---
def extract_features(url):
    """Extracts features from a single URL."""
    features = {}
    
    parsed_url = urlparse(url)
    domain_info = tldextract.extract(url)
    
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
    features['is_ip'] = 1 if domain_info.domain.replace('.', '').isdigit() else 0
    features['abnormal_subdomain'] = 1 if url.count('.') > 3 and "www" not in url.lower() else 0
    
    # Create a DataFrame with the correct feature order
    return pd.DataFrame([features], columns=FEATURE_NAMES)

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    url_input = ""
    is_phishing = None

    if request.method == 'POST':
        url_input = request.form.get('url')
        if url_input:
            # 1. Extract features from the input URL
            features_df = extract_features(url_input)
            
            # 2. Use the trained model to predict
            prediction = model.predict(features_df)
            
            # 3. Interpret the prediction
            is_phishing = (prediction[0] == 1)
            result = "🚨 This URL is likely a PHISHING attempt!" if is_phishing else "✅ This URL seems SAFE."
            
    return render_template('index.html', result=result, url=url_input, is_phishing=is_phishing)

if __name__ == '__main__':
    app.run(debug=True)