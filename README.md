# 🌾 Fertilizer and Crop Recommendation System

## 📌 Project Overview
This is a Flask-based web application that predicts the best crop and fertilizer based on soil and environmental conditions. It utilizes machine learning techniques to provide accurate recommendations.

---
## 🚀 Features
- 🌱 Predicts suitable **crop** and **fertilizer** based on user inputs.
- 🔄 Uses **Random Forest Classifier** inside a **MultiOutputClassifier**.
- 📊 Preprocesses data using **Label Encoding** and **Standard Scaling**.
- 🌐 Built with **Flask, Pandas, NumPy, and Scikit-Learn**.

---
## 🛠 Installation Guide
### 1️⃣ Clone the repository:
```bash
git clone https://github.com/safiyashaik123/Fertilizer_Crop_Recommendation.git
cd Fertilizer_Crop_Recommendation
```

### 2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application:
```bash
python app.py
```
The app will be available at **http://127.0.0.1:5000/**.

---
## 📂 Project Structure
```
📁 Fertilizer_Crop_Recommendation
│── app.py             # Main Flask application
│── templates/
│   └── index.html     # Frontend template
│── fertilizer_recommends.csv  # Dataset used for training
│── requirements.txt   # Dependencies
│── README.md          # Project documentation
```

---
## 🏗 How It Works
1. Loads and encodes the dataset (`fertilizer_recommends.csv`).
2. Trains a **MultiOutputClassifier(RandomForestClassifier)** model.
3. Takes user inputs through a web form.
4. Predicts the **best crop and fertilizer** based on inputs.

---
## ✨ Tech Stack
- **Python** (Flask, Pandas, NumPy, Scikit-Learn)
- **Machine Learning** (Random Forest Classifier, MultiOutputClassifier)
- **Web Technologies** (HTML, CSS, Flask)

---
## 🤝 Contributing
Pull requests are welcome! If you want to contribute, feel free to fork the repository and submit a PR.

---
## 📜 License
This project is open-source and available under the **MIT License**.

---
## 📧 Contact
If you have any questions or suggestions, feel free to reach out via **GitHub Issues**.

