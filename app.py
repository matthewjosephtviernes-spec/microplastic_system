import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

st.title("Microplastic Risk Prediction System")

# ========================================================
# STEP 3: LOAD DATA + PREPROCESSING
# ========================================================
@st.cache_data
def load_data():
    url = "YOUR_GITHUB_RAW_LINK"
    data = pd.read_csv(url)
    return data

df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df)

# Preprocessing
df = df.dropna()
st.write("✅ Data cleaned: Missing values removed")

# Encode your columns here later (Risk_Level etc.)


# ========================================================
# STEP 4: MODELING
# ========================================================
st.header("⚙️ Model Training and Evaluation")

X = df.drop("Risk_Level", axis=1)
y = df["Risk_Level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='macro'),
        "Recall": recall_score(y_test, y_pred, average='macro'),
        "F1 Score": f1_score(y_test, y_pred, average='macro')
    }

st.table(pd.DataFrame(results).T)


# ========================================================
# STEP 5: K-FOLD VALIDATION
# ========================================================
st.header("🔁 10-Fold Cross Validation")

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = {}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    cv_scores[name] = scores.mean()

st.write(cv_scores)


# ========================================================
# STEP 6: VISUALIZATIONS
# ========================================================
st.header("📊 Data Visualization")

plt.figure(figsize=(6,4))
df["Risk_Level"].value_counts().plot(kind='bar')
plt.title("Distribution of Risk Levels")
st.pyplot()
