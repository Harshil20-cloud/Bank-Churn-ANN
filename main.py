# main.py

from model import (
    load_and_preprocess_data,
    build_model,
    train_model,
    evaluate_model
)

import joblib


def main():
    try:
        # ✅ Correct CSV name (make sure it matches your file)
        file_path = "Artificial_Neural_Network_Case_Study_data (1).csv"

        # Load and preprocess data
        X_train, X_test, y_train, y_test, sc, ct = load_and_preprocess_data(file_path)
        print("✅ Data preprocessing completed.")

        # Build model
        model = build_model(X_train.shape[1])
        print("✅ Model built successfully.")

        # Train model
        model = train_model(model, X_train, y_train)
        print("✅ Model training completed.")

        # Evaluate model
        print("\n📊 Model Evaluation:")
        evaluate_model(model, X_test, y_test)

        # ✅ Save model + scaler + encoder (VERY IMPORTANT)
        model.save("churn_model.h5")
        joblib.dump(sc, "scaler.pkl")
        joblib.dump(ct, "encoder.pkl")

        print("\n✅ SUCCESS: All files saved (model, scaler, encoder)")

    except FileNotFoundError:
        print("❌ ERROR: CSV file not found. Check file name and location.")

    except Exception as e:
        print(f"❌ ERROR: {e}")


if __name__ == "__main__":
    main()