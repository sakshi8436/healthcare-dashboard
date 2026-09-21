
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Disease Prediction Dashboard",
    page_icon="🩺",
    layout="wide"
)

# -------------------------------
# TITLE
# -------------------------------

st.title("🩺 Random Forest Based Disease Prediction")
st.markdown(
    "### Healthcare Disease Prediction Dashboard"
)

st.divider()

# -------------------------------
# LOAD DATASET
# -------------------------------

df = pd.read_csv("patients - patients.csv")

# -------------------------------
# DATA CLEANING
# -------------------------------

df["Age"] = df["Age"].fillna(
    df["Age"].median()
)

df["Weight"] = df["Weight"].fillna(
    df["Weight"].median()
)

# Handle unrealistic age
valid_age_median = df.loc[
    df["Age"] <= 100,
    "Age"
].median()

df.loc[
    df["Age"] > 100,
    "Age"
] = valid_age_median

# -------------------------------
# FEATURES & TARGET
# -------------------------------

X = df[
    ["Age", "Gender", "Weight"]
]

y = df["Disease"]

X = pd.get_dummies(
    X,
    columns=["Gender"],
    drop_first=True
)

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -------------------------------
# RANDOM FOREST MODEL
# -------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

# -------------------------------
# DASHBOARD METRICS
# -------------------------------

st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Patients",
    len(df)
)

col2.metric(
    "Total Features",
    3
)

col3.metric(
    "Total Diseases",
    df["Disease"].nunique()
)

col4.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)

# -------------------------------
# DATASET PREVIEW
# -------------------------------

st.subheader("📋 Dataset Preview")

st.dataframe(
    df,
    use_container_width=True
)

# -------------------------------
# EDA
# -------------------------------

st.header("📈 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:

    fig1, ax1 = plt.subplots()

    sns.countplot(
        data=df,
        x="Disease",
        ax=ax1
    )

    ax1.set_title(
        "Disease Distribution"
    )

    ax1.tick_params(
        axis="x",
        rotation=30
    )

    st.pyplot(fig1)

with col2:

    fig2, ax2 = plt.subplots()

    sns.histplot(
        data=df,
        x="Age",
        bins=10,
        kde=True,
        ax=ax2
    )

    ax2.set_title(
        "Age Distribution"
    )

    st.pyplot(fig2)

# -------------------------------
# GENDER DISTRIBUTION
# -------------------------------

col1, col2 = st.columns(2)

with col1:

    fig3, ax3 = plt.subplots()

    sns.countplot(
        data=df,
        x="Gender",
        ax=ax3
    )

    ax3.set_title(
        "Gender Distribution"
    )

    st.pyplot(fig3)

with col2:

    fig4, ax4 = plt.subplots()

    sns.histplot(
        data=df,
        x="Weight",
        bins=10,
        kde=True,
        ax=ax4
    )

    ax4.set_title(
        "Weight Distribution"
    )

    st.pyplot(fig4)

# -------------------------------
# RANDOM FOREST PERFORMANCE
# -------------------------------

st.header("🤖 Random Forest Model")

st.metric(
    "Random Forest Accuracy",
    f"{accuracy * 100:.2f}%"
)

# Confusion Matrix

st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

fig5, ax5 = plt.subplots(
    figsize=(8, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_,
    ax=ax5
)

ax5.set_xlabel(
    "Predicted Disease"
)

ax5.set_ylabel(
    "Actual Disease"
)

st.pyplot(fig5)

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------

st.subheader("🔍 Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

fig6, ax6 = plt.subplots(
    figsize=(8, 5)
)

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature",
    ax=ax6
)

ax6.set_title(
    "Random Forest Feature Importance"
)

st.pyplot(fig6)

# -------------------------------
# DISEASE PREDICTION
# -------------------------------

st.header("🔮 Disease Prediction")

st.write(
    "Enter patient information to predict the possible disease."
)

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30
    )

with col2:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col3:

    weight = st.number_input(
        "Weight (kg)",
        min_value=1.0,
        max_value=200.0,
        value=60.0
    )

# -------------------------------
# PREDICT
# -------------------------------

if st.button(
    "🔍 Predict Disease",
    use_container_width=True
):

    patient = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Weight": [weight]
    })

    patient = pd.get_dummies(
        patient,
        columns=["Gender"],
        drop_first=True
    )

    patient = patient.reindex(
        columns=X.columns,
        fill_value=0
    )

    prediction = model.predict(
        patient
    )[0]

    probabilities = model.predict_proba(
        patient
    )[0]

    confidence = max(
        probabilities
    )

    st.success(
        f"🩺 Predicted Disease: {prediction}"
    )

    st.info(
        f"Prediction Probability: "
        f"{confidence * 100:.2f}%"
    )

    # Probability chart

    probability_df = pd.DataFrame({
        "Disease": model.classes_,
        "Probability": probabilities * 100
    })

    probability_df = probability_df.sort_values(
        "Probability",
        ascending=False
    )

    fig7, ax7 = plt.subplots(
        figsize=(8, 5)
    )

    sns.barplot(
        data=probability_df,
        x="Probability",
        y="Disease",
        ax=ax7
    )

    ax7.set_xlabel(
        "Probability (%)"
    )

    ax7.set_title(
        "Disease Prediction Probability"
    )

    st.pyplot(fig7)

# -------------------------------
# FOOTER
# -------------------------------

st.divider()

st.caption(
    "Random Forest Disease Prediction | "
    "Academic Machine Learning Project"
)
