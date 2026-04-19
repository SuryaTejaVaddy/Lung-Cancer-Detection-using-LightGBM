  # Lung Cancer Detection using LightGBM

  Early detection of lung cancer saves lives. This project applies a **LightGBM classifier** on a real-world survey dataset to predict whether a patient is
  at risk of lung cancer — using only symptom-based responses, no lab tests or imaging required.

  ---

  ## Project Details

  | Field | Details |
  |---|---|
  | **University** | GITAM (Deemed to be University) |
  | **Program** | B.Tech — CSE (AI & ML) |
  | **Course** | Artificial Neural Networks |
  | **Instructor** | Mr. Sankara Rao |
  | **Academic Year** | 2021-22 |
  | **Team** | V.Surya Teja  · T.Vamshi · D.Anjaniputra Varma ·  Ch.Tulasi Latha |

  ---

  ## Dataset

  - **Source:** Kaggle — Survey Lung Cancer Dataset
  - **Link:** https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer
  - **Records:** 284 patient survey responses
  - **Attributes:** 16 (15 symptom-based features + 1 class label)
  - **Class Label:** LUNG_CANCER → YES (268) / NO (16)

  ### Features Used

  | # | Feature | Type |
  |---|---|---|
  | 1 | GENDER | Categorical (M/F) |
  | 2 | AGE | Continuous |
  | 3 | SMOKING | Binary |
  | 4 | YELLOW_FINGERS | Binary |
  | 5 | ANXIETY | Binary |
  | 6 | PEER_PRESSURE | Binary |
  | 7 | CHRONIC DISEASE | Binary |
  | 8 | FATIGUE | Binary |
  | 9 | ALLERGY | Binary |
  | 10 | WHEEZING | Binary |
  | 11 | ALCOHOL CONSUMING | Binary |
  | 12 | COUGHING | Binary |
  | 13 | SHORTNESS OF BREATH | Binary |
  | 14 | SWALLOWING DIFFICULTY | Binary |
  | 15 | CHEST PAIN | Binary |

  ---

  ## Pipeline

  **1. Load Data**
  Survey responses are read from `survey_lung_cancer_dataset.csv` using Pandas.

  **2. Clean Data**
  - Duplicate records are removed
  - No missing values found across all 16 columns

  **3. Encode & Normalize**
  - GENDER and LUNG_CANCER labels are encoded using `LabelEncoder`
  - Binary symptom values are rescaled from (1, 2) to (0, 1)

  **4. Handle Class Imbalance**
  The dataset is heavily skewed (268 YES vs 16 NO). `RandomOverSampler` from imbalanced-learn balances both classes before training.

  **5. Split**
  75% training / 25% testing with stratified sampling to preserve class ratios.

  **6. Scale**
  `StandardScaler` is applied to the AGE column only, so it does not disproportionately influence the model.

  **7. Train**
  A `LGBMClassifier` is trained on the balanced, scaled training data.

  **8. Evaluate**
  Model is evaluated on the held-out test set using a confusion matrix, classification report, and accuracy score.

  ---

  ## Results

  | Metric | No Cancer | Cancer |
  |---|---|---|
  | **Precision** | 0.91 | 1.00 |
  | **Recall** | 1.00 | 0.90 |
  | **F1-Score** | 0.95 | 0.95 |
  | **Support** | 60 | 59 |

  **Overall Accuracy: 94.96%**

  ---

  ## Tech Stack

  | Tool | Purpose |
  |---|---|
  | Python 3.12 | Core programming language |
  | LightGBM 4.6 | Gradient boosting classifier |
  | Pandas & NumPy | Data loading and processing |
  | Scikit-learn | Encoding, scaling, splitting, evaluation |
  | Matplotlib & Seaborn | EDA and visualization |
  | imbalanced-learn | Random oversampling for class balance |

  ---

  ## How to Run

  1. Install dependencies — `pip install -r requirements.txt`
  2. Run the model — `python lung_cancer_lgbm.py`

  ---

  ## Project Structure

  Assignment/
  │
  ├── lung_cancer_lgbm.py              # Full pipeline: load, process, train, evaluate, predict
  ├── survey_lung_cancer_dataset.csv   # Survey dataset (284 records, 16 columns)
  ├── requirements.txt                 # Python dependencies
  └── README.md                        # Project documentation

  ---

  ## References

  - Kaggle Dataset: https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer
  - LightGBM Docs: https://lightgbm.readthedocs.io/
  - Scikit-learn Docs: https://scikit-learn.org/