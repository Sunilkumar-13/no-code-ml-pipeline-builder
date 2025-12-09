import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="No-Code ML Pipeline Builder", layout="wide")

st.title("🧩 No-Code ML Pipeline Builder")
st.markdown("""
This app lets you build a simple Machine Learning pipeline **without writing code**.

**Flow:**  
1️⃣ Upload Dataset → 2️⃣ Preprocess → 3️⃣ Train–Test Split → 4️⃣ Model Selection → 5️⃣ Results  
""")

# ---------- STEP 1: DATASET UPLOAD ----------
st.header("1. Upload Dataset")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

df = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ File uploaded successfully!")
        st.write(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")
        st.write("**Column Names:**", list(df.columns))
        st.subheader("Data Preview")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Could not read the file: {e}")
        df = None
else:
    st.info("Please upload a CSV or Excel file to continue.")

if df is not None:

    # ---------- STEP 2: DATA PREPROCESSING ----------
    st.header("2. Data Preprocessing")

    numeric_cols = df.select_dtypes(include=["int64", "float64", "float32", "int32"]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns detected. Preprocessing will be skipped for scaling.")
        df_processed = df.copy()
    else:
        st.write("**Numeric columns detected:** ", numeric_cols)
        cols_to_scale = st.multiselect(
            "Select numeric columns to apply scaling to:",
            numeric_cols,
            default=numeric_cols
        )

        scaler_option = st.radio(
            "Choose preprocessing method:",
            ["None", "Standardization (StandardScaler)", "Normalization (MinMaxScaler)"],
            index=0
        )

        df_processed = df.copy()
        if scaler_option != "None" and cols_to_scale:
            if scaler_option == "Standardization (StandardScaler)":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()

            df_processed[cols_to_scale] = scaler.fit_transform(df_processed[cols_to_scale])
            st.success(f"✅ {scaler_option} applied to: {cols_to_scale}")
        else:
            st.info("No scaling applied.")

    st.subheader("Processed Data Preview")
    st.dataframe(df_processed.head())

    # ---------- STEP 3: TRAIN–TEST SPLIT ----------
    st.header("3. Train–Test Split")

    target_col = st.selectbox("Select the target column (label):", options=df_processed.columns)

    test_size_percent = st.slider(
        "Select test size percentage:",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
        help="Example: 20 means 80% train, 20% test."
    )

    # ---------- STEP 4: MODEL SELECTION ----------
    st.header("4. Model Selection")

    model_name = st.radio(
        "Choose a model:",
        ["Logistic Regression", "Decision Tree Classifier"]
    )

    run_pipeline = st.button("🚀 Run Pipeline")

    # ---------- STEP 5: MODEL TRAINING & RESULTS ----------
    if run_pipeline:
        if target_col is None:
            st.error("Please select a target column.")
        else:
            X = df_processed.drop(columns=[target_col])
            y = df_processed[target_col]

            # One-hot encode categorical features
            X = pd.get_dummies(X, drop_first=True)

            from pandas.api.types import is_numeric_dtype

            # Encode target if non-numeric
            if not is_numeric_dtype(y):
                y_encoded, class_names = pd.factorize(y)
            else:
                y_encoded = y.values
                class_names = np.unique(y_encoded).astype(str)

            test_size = test_size_percent / 100.0

try:
    stratify_arg = y_encoded if len(np.unique(y_encoded)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=stratify_arg
    )
except ValueError:
    # Fallback if stratified split fails due to very small classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=None
    )
                if model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                else:
                    model = DecisionTreeClassifier(random_state=42)

                with st.spinner("Training model..."):
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)

                st.header("5. Model Output & Results")
                st.success("✅ Model training completed successfully!")
                st.write(f"**Selected Model:** {model_name}")
                st.write(f"**Accuracy:** `{acc:.2f}`")

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=class_names,
                    yticklabels=class_names,
                    ax=ax
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)

                st.info("You have completed the full pipeline: Data → Preprocessing → Split → Model → Results ✅")
            except Exception as e:
                st.error(f"❌ Something went wrong while training the model: {e}")
