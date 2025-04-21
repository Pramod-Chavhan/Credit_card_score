# 💳 Credit Score Classification  
### 🎯 Machine Learning with Random Forest Classifier

<p align="center">
  <img src="https://img.shields.io/badge/Model-RandomForest-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
</p>

---
 ## Live Demo : https://creditcard-score.onrender.com/
## 📌 Overview

This project classifies individuals' credit scores into categories like **Good**, **Average**, and **Poor** using a **Random Forest Classifier**. It processes financial and demographic data to assist in credit risk evaluation, and is built to support real-world applications such as lending systems and financial dashboards.

---

## 🧰 Features

- 🔄 **Automated data preprocessing**
- 🌲 **Random Forest classification model**
- 📊 **Model evaluation using multiple metrics**
- 📈 **Insightful visualizations**
- 💾 **Model saving for deployment**

---

## 🧱 Project Structure

credit-score-classification/ ├── data/ # Raw and cleaned data │ ├── raw/ # Original input datasets │ └── processed/ # Cleaned and preprocessed data │ ├── models/ # Trained ML models │ └── random_forest_model.pkl │ ├── notebooks/ # Jupyter notebooks for EDA & experiments │ └── eda.ipynb │ ├── src/ # Source code │ ├── init.py │ ├── config.py # Configuration and constants │ ├── preprocessing.py # Data cleaning & transformation │ ├── train.py # Training the Random Forest model │ ├── evaluate.py # Model performance evaluation │ └── utils.py # Helper functions │ ├── outputs/ # Evaluation reports & plots │ ├── plots/ │ └── evaluation_report.json │ ├── tests/ # Unit tests │ └── test_preprocessing.py │ ├── requirements.txt # Python dependencies ├── README.md # Project documentation ├── .gitignore # Git ignore file └── LICENSE # Project license

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/credit-score-classification.git
cd credit-score-classification
2. Create & Activate Virtual Environment
bash

python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
3. Install Dependencies
bash

pip install -r requirements.txt
4. Run Training Script
bash

python src/train.py
📊 Model Evaluation
The model is evaluated on multiple classification metrics:

✅ Accuracy

🎯 Precision

🔁 Recall

🧠 F1-Score

Results are printed in the console and optionally saved to /outputs/evaluation_report.json.

🧪 Example Results (Optional)

Accuracy: 91.3%
Precision: 90.8%
Recall: 89.6%
F1 Score: 90.2%
📦 Tech Stack
Language: Python 3.8+

Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

📄 License
This project is licensed under the MIT License.

👨‍💻 Author
Pramod
💬 Feel free to reach out for collaboration or improvement suggestions.

🙌 Acknowledgments
This project was built as a practical application of classification techniques in real-world credit systems. It focuses on accuracy, explainability, and maintainable structure for further development or integration.

