import pandas as pd
import numpy as np
import matplotlib.pyplot as mp
import seaborn as sb
import sklearn
import lightgbm as lgb

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler

mp.style.use('ggplot')
colors = sb.color_palette('Paired', 5)
sb.set_palette(colors)

print("=" * 60)
print("    LUNG CANCER DETECTION USING LGBM CLASSIFIER")
print("=" * 60)

df = pd.read_csv("Lung_Cancer_Dataset.csv")
print(f"\n[INFO] Dataset Shape     : {df.shape}")
print(f"[INFO] Missing Values    : {df.isnull().sum().sum()}")
print(f"[INFO] Duplicates        : {df.duplicated().sum()}")
print(f"\n[INFO] Class Distribution:\n{df['LUNG_CANCER'].value_counts()}")

fig, axes = mp.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Lung Cancer Dataset - EDA", fontsize=16)
sb.countplot(x='LUNG_CANCER', data=df, ax=axes[0, 0])
axes[0, 0].set_title("Class Distribution")
sb.histplot(df['AGE'], kde=True, ax=axes[0, 1], color='steelblue')
axes[0, 1].set_title("Age Distribution")
sb.countplot(x='GENDER', hue='LUNG_CANCER', data=df, ax=axes[1, 0])
axes[1, 0].set_title("Gender vs Lung Cancer")
sb.countplot(x='SMOKING', hue='LUNG_CANCER', data=df, ax=axes[1, 1])
axes[1, 1].set_title("Smoking vs Lung Cancer")
mp.tight_layout()
mp.savefig("eda_plots.png", dpi=120)
print("\n[INFO] EDA plots saved as eda_plots.png")

print("\n" + "=" * 60)
print("                  PRE-PROCESSING")
print("=" * 60)

df.drop_duplicates(inplace=True)
label_encoder = preprocessing.LabelEncoder()
df['GENDER']      = label_encoder.fit_transform(df['GENDER'])
df['LUNG_CANCER'] = label_encoder.fit_transform(df['LUNG_CANCER'])
df.to_csv("Modified.csv", index=False)
mdf = pd.read_csv("Modified.csv")
print(f"[INFO] After encoding:\n{mdf.head()}")

X = mdf.drop(['LUNG_CANCER'], axis=1)
y = mdf['LUNG_CANCER']

symptom_cols = [c for c in X.columns if c not in ['GENDER', 'AGE']]
for col in symptom_cols:
    X[col] = X[col].apply(lambda v: v - 1)

ros = RandomOverSampler(random_state=42)
X_over, y_over = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_over, y_over, random_state=42, stratify=y_over)
print(f"\n[INFO] Train : {X_train.shape} | Test : {X_test.shape}")

scaler = StandardScaler()
X_train = X_train.copy()
X_test  = X_test.copy()
X_train['AGE'] = scaler.fit_transform(X_train[['AGE']])
X_test['AGE']  = scaler.transform(X_test[['AGE']])

print("\n" + "=" * 60)
print("                  MODEL TRAINING")
print("=" * 60)
model = lgb.LGBMClassifier(random_state=42, verbose=-1)
model.fit(X_train, y_train)
print("[INFO] Model trained successfully!")

print("\n" + "=" * 60)
print("                  MODEL EVALUATION")
print("=" * 60)
y_pred = model.predict(X_test)

confusion = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{confusion}")

mp.figure(figsize=(8, 6))
sb.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
           xticklabels=['No Cancer', 'Cancer'],
           yticklabels=['No Cancer', 'Cancer'])
mp.xlabel("Predicted")
mp.ylabel("Actual")
mp.title("Confusion Matrix - LightGBM Classifier")
mp.tight_layout()
mp.savefig("confusion_matrix.png", dpi=120)

print("\n" + "-" * 55)
print(classification_report(y_test, y_pred, target_names=['No Cancer', 'Cancer']))
print("-" * 55)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy using LGBM Classifier is : {acc:.4f}  ({acc*100:.2f}%)")
print("-" * 55)

print("\n" + "=" * 60)
print("                  SINGLE PREDICTION")
print("=" * 60)
sample_input = (1, 62, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0)
input_df = pd.DataFrame([sample_input], columns=X_train.columns)
input_df['AGE'] = scaler.transform(input_df[['AGE']])
prediction = model.predict(input_df)
print(f"\nInput  : {sample_input}")
print(f"Result : {'The Person HAS Lung Cancer' if prediction[0]==1 else 'The Person does NOT have Lung Cancer'}")
print("\n" + "=" * 60)
print("                       DONE")
print("=" * 60)
