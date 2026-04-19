# 🫁 Lung Cancer Detection using LightGBM

Early detection of lung cancer saves lives. This project builds a machine learning model that predicts whether a person is at risk of lung cancer based on simple symptom-based survey data — no medical imaging or lab tests required.

---

## 📌 Project Overview

Lung cancer is the **leading cause of cancer-related deaths worldwide**. Most cases are detected too late because early symptoms are easy to ignore or misattribute. This project tackles that problem by training a **LightGBM (Light Gradient Boosting Machine)** classifier on a dataset of 284 patient records, each described by 15 common symptoms and risk factors.

The model learns patterns from this data and can predict lung cancer risk with **~95% accuracy**, making it a low-cost, accessible screening tool that could support early medical intervention.

---

## 👥 Project Details

| Field | Details |
|---|---|
| **University** | GITAM (Deemed to be University) |
| **Program** | B.Tech — CSE (AI & ML) |
| **Course** | Artificial Neural Networks |
| **Instructor** | Mr. Sankara Rao |
| **Academic Year** | 2021-22 |
| **Team** | Ch.Tulasi Latha · T.Vamshi · D.Anjaniputra Varma · V.Surya Teja |

---

## 📊 Dataset

- **Source:** Kaggle — Survey Lung Cancer Dataset
- **Link:** https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer
- **Records:** 284 patient survey responses
- **Attributes:** 16 (15 symptom-based features + 1 class label)
- **Class Label:** LUNG_CANCER → YES / NO

### Features Used

| # | Feature | Description |
|---|---|---|
| 1 | GENDER | Male / Female |
| 2 | AGE | Age of the patient |
| 3 | SMOKING | Smoker or not |
| 4 | YELLOW_FINGERS | Presence of yellow fingers |
| 5 | ANXIETY | Suffers from anxiety |
| 6 | PEER_PRESSURE | Influenced by peer pressure |
| 7 | CHRONIC DISEASE | Has a chronic illness |
| 8 | FATIGUE | Experiences fatigue |
| 9 | ALLERGY | Has allergies |
| 10 | WHEEZING | Experiences wheezing |
| 11 | ALCOHOL CONSUMING | Consumes alcohol |
| 12 | COUGHING | Persistent coughing |
| 13 | SHORTNESS OF BREATH | Difficulty breathing |
| 14 | SWALLOWING DIFFICULTY | Trouble swallowing |
| 15 | CHEST PAIN | Chest pain or discomfort |

---

## ⚙️ How It Works

**1. Data Collection**
Survey data is loaded from a CSV file containing patient responses.

**2. Pre-processing**
- Duplicate records are removed
- Text labels (M/F, YES/NO) are converted to numbers using Label Encoding
- Binary symptom values are normalized from (1, 2) to (0, 1)

**3. Handling Class Imbalance**
The dataset has far more lung cancer cases than non-cases. Random Oversampling is applied to balance both classes so the model does not become biased.

**4. Train / Test Split**
Data is split 75% for training and 25% for testing, using stratified sampling to preserve class balance.

**5. Feature Scaling**
The AGE column is standardized using StandardScaler so it does not dominate other features during training.

**6. Model Training**
A **LightGBM Classifier** is trained on the processed data. LightGBM is a fast, efficient gradient boosting algorithm that performs exceptionally well on tabular datasets.

**7. Evaluation**
The model is tested on unseen data and evaluated using a confusion matrix, classification report, and accuracy score.

**8. Prediction**
The trained model can take a new patient's symptom inputs and instantly predict their lung cancer risk.

---

## 📈 Results

| Metric | No Cancer | Cancer |
|---|---|---|
| **Precision** | 0.91 | 1.00 |
| **Recall** | 1.00 | 0.90 |
| **F1-Score** | 0.95 | 0.95 |
| **Support** | 60 | 59 |

**Overall Accuracy: 94.95%**

---

## 💡 Why This Project Matters

- **Accessible:** Uses only survey responses — no expensive scans or lab work needed
- **Fast:** Prediction is instant once the model is trained
- **Scalable:** Can be deployed as a web app or mobile tool for mass screening
- **Impactful:** Early detection dramatically improves survival rates for lung cancer patients

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| LightGBM | Machine learning classifier |
| Pandas & NumPy | Data processing |
| Scikit-learn | Preprocessing, splitting, evaluation |
| Matplotlib & Seaborn | Data visualization |
| imbalanced-learn | Handling class imbalance |

---

## 🚀 How to Run

\`\`\`bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the model
python lung_cancer_lgbm.py
\`\`\`

---

## 📁 Project Structure

\`\`\`
Assignment/
│
├── lung_cancer_lgbm.py        # Full pipeline: training, evaluation, prediction
├── Lung_Cancer_Dataset.csv    # Input dataset
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
\`\`\`

---

## 📚 References

- Kaggle Dataset: https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer
- LightGBM Docs: https://lightgbm.readthedocs.io/
- Scikit-learn Docs: https://scikit-learn.org/
