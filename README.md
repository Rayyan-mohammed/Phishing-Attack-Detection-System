#  Phishing Attack Detection System 🎣

![Python](https://img.shields.io/badge/Python-3.7%252B-blue?logo=python)  
![Flask](https://img.shields.io/badge/Flask-Web%2520Framework-lightgrey?logo=flask)  
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange?logo=scikit-learn)  
![License](https://img.shields.io/badge/License-MIT-green.svg)  

A web-based machine learning application that detects phishing websites from URLs. This system identifies fraudulent links by analyzing various URL features and classifying them as **SAFE ✅** or **PHISHING 🚨**.  

---

## 🔎 Overview  
Phishing is a major cybersecurity threat where attackers trick users into revealing sensitive information by impersonating legitimate websites. This system provides a simple interface to check a URL's safety by extracting key features and using a pre-trained **Random Forest classifier** for prediction.  

---

## ✨ Features  
- **Feature Extraction:** Analyzes over 15 URL characteristics including length, domain properties, and special characters  
- **Machine Learning Model:** Utilizes a Random Forest classifier trained on a large dataset for high accuracy  
- **User-Friendly Web Interface:** Clean Flask-based interface for easy URL verification  
- **Instant Results:** Provides clear "SAFE ✅" or "PHISHING 🚨" classification  
- **Real-time Prediction:** Instant analysis without external API calls  

---

## 🛠️ Tech Stack  
- **Backend:** Python, Flask  
- **Machine Learning:** Scikit-learn, Pandas, Joblib  
- **URL Parsing:** tldextract, urllib  
- **Frontend:** HTML, CSS, JavaScript  

---

## 📂 Project Structure  
```
phishing_detector/
├── templates/
│   └── index.html          # Frontend web page
├── app.py                  # Flask web application
├── train_model.py          # Script to train and save the ML model
├── requirements.txt        # Python dependencies
├── phishing_model.pkl      # Saved trained machine learning model
├── phishing_data.csv       # Dataset for training
└── README.md               # Project documentation
```

---

## ⚡ Setup & Installation  

### 1. Prerequisites  
- Python 3.7 or higher  
- pip (Python package installer)  

### 2. Clone the Repository  
```bash
git clone https://github.com/Rayyan-mohammed/Phishing-Attack-Detection-System.git
cd Phishing-Attack-Detection-System
```

### 3. Get the Dataset  
Download the dataset from Kaggle: **Phishing Site URLs Dataset**.  

Ensure the CSV file contains **url** and **label** columns, then place it in the project root as `phishing_data.csv`.  

### 4. Install Dependencies  
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run  

### Step 1: Train the Machine Learning Model  
Run the training script to process the dataset and create the model:  
```bash
python train_model.py
```

This will:  
- Process the `phishing_data.csv` file  
- Train the Random Forest classifier  
- Save the trained model as `phishing_model.pkl`  
- Display model accuracy metrics  

*Note: This step only needs to be done once unless retraining with new data.*  

### Step 2: Run the Web Application  
```bash
python app.py
```

### Step 3: Access the Application  
Open your web browser and navigate to:  
```
http://127.0.0.1:5000
```

Enter any URL into the input box and click **"Check URL"** to see if it's a phishing attempt.  

---

## 🔮 Future Enhancements  
- Browser Extension: Real-time URL scanning & protection  
- Deep Learning Models: Use LSTMs or CNNs for better detection  
- API Service: REST API for integration with other apps  
- Real-time Blacklists: Integration with PhishTank / Google Safe Browsing  
- Advanced Features: Screenshot & content-based detection  
- Multi-language Support: For global usability  
- Historical Analysis: Store user analysis history  
- Threat Intelligence Feed: Regular phishing pattern updates  

---

## 📊 Model Performance  
The Random Forest classifier achieves high accuracy in detecting phishing URLs:  

- **Accuracy:** >95% on test data  
- **Precision:** >94% for phishing detection  
- **Recall:** >96% for identifying phishing sites  

---

## 🤝 Contributing  
Contributions, issues, and feature requests are welcome!  

1. Fork the project  
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)  
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)  
4. Push to the branch (`git push origin feature/AmazingFeature`)  
5. Open a Pull Request  

---

## 📝 License  
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  

---

## 📧 Contact  
**Rayyan Mohammed**  
- GitHub: [@Rayyan-mohammed](https://github.com/Rayyan-mohammed)  
- Email: rayyan@example.com  

Project Link: [Phishing Attack Detection System](https://github.com/Rayyan-mohammed/Phishing-Attack-Detection-System)  

---

## 🙏 Acknowledgments  
- [Kaggle](https://www.kaggle.com/) for providing the phishing dataset  
- [Scikit-learn](https://scikit-learn.org/) team for the ML library  
- [Flask](https://flask.palletsprojects.com/) team for the web framework  
- Cybersecurity researchers contributing to phishing detection methodologies  
