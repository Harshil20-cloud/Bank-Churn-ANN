# model.py

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# =========================
# LOAD & PREPROCESS DATA
# =========================
def load_and_preprocess_data(file_path):

    # Load dataset
    data = pd.read_csv(file_path)

    # Features and target
    X = data.iloc[:, 3:-1].values
    y = data.iloc[:, -1].values

    # Encode Gender (Male/Female → 1/0)
    le = LabelEncoder()
    X[:, 2] = le.fit_transform(X[:, 2])

    # One-hot encode Geography (France/Germany/Spain)
    ct = ColumnTransformer(
        transformers=[("geo", OneHotEncoder(drop="first", handle_unknown="ignore"), [1])],
        remainder="passthrough"
    )

    X = ct.fit_transform(X)
    X = np.array(X, dtype=np.float64)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test, sc, ct


# =========================
# BUILD MODEL
# =========================
def build_model(input_dim):

    model = Sequential()

    model.add(Dense(units=6, activation="sigmoid", input_dim=input_dim))
    model.add(Dense(units=6, activation="sigmoid"))
    model.add(Dense(units=1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =========================
# TRAIN MODEL
# =========================
def train_model(model, X_train, y_train):

    model.fit(X_train, y_train, batch_size=25, epochs=10, verbose=1)

    return model


# =========================
# EVALUATE MODEL
# =========================
def evaluate_model(model, X_test, y_test):

    y_pred = (model.predict(X_test) > 0.5).astype(int)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nAccuracy:")
    print(accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    try:
        file_path = "Artificial_Neural_Network_Case_Study_data (1).csv"

        # Load data
        X_train, X_test, y_train, y_test, sc, ct = load_and_preprocess_data(file_path)

        # Build model
        model = build_model(X_train.shape[1])

        # Train model
        model = train_model(model, X_train, y_train)

        # Evaluate
        evaluate_model(model, X_test, y_test)

        # Save everything
        model.save("churn_model.h5")
        joblib.dump(sc, "scaler.pkl")
        joblib.dump(ct, "encoder.pkl")

        print("\n✅ SUCCESS: Model, scaler, encoder saved!")

    except FileNotFoundError:
        print("❌ ERROR: CSV file not found. Check file name.")

    except Exception as e:
        print(f"❌ ERROR: {e}")