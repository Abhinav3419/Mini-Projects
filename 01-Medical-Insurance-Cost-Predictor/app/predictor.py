"""
predictor.py — The prediction pipeline.

WHY THIS FILE EXISTS (Interview concept: "Separation of concerns")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
In production ML, you NEVER put model loading and prediction logic directly
inside your API route handlers. You separate them because:

1. The model loads ONCE when the server starts (expensive: ~2-5 seconds).
   API requests happen thousands of times per second (cheap: ~5ms each).
   If you reload the model on every request, you're wasting 99.9% of time on loading.

2. You can test the prediction logic independently of the web framework.
   If you switch from FastAPI to Flask tomorrow, this file doesn't change.

3. Different team members can work on the model vs the API without conflicts.

INTERVIEW QUESTION: "How do you serve an ML model in production?"
ANSWER: "Load the model once at startup into memory, then each API request
         runs inference on the already-loaded model. The prediction pipeline
         handles feature engineering, scaling, and prediction in a single call."
"""

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf


class InsurancePredictor:
    """
    Encapsulates the full prediction pipeline:
    raw input → one-hot encode → feature engineer → scale → predict → return.

    WHY A CLASS? (Interview concept: "Stateful vs stateless")
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    The model and scaler are STATEFUL objects — they hold learned weights/parameters.
    A class holds them as instance attributes, loaded once at __init__.
    Every prediction call after that is stateless — same input always gives same output.
    """

    # The exact columns that one-hot encoding produces.
    # This MUST match what the model was trained on.
    # If the training data had these columns, the deployed model needs them too.
    ORIGINAL_COLUMNS = [
        'age', 'bmi', 'children',
        'sex_female', 'sex_male',
        'smoker_no', 'smoker_yes',
        'region_northeast', 'region_northwest',
        'region_southeast', 'region_southwest'
    ]

    # Valid input categories (for validation)
    VALID_SEX = ['male', 'female']
    VALID_SMOKER = ['yes', 'no']
    VALID_REGION = ['northeast', 'northwest', 'southeast', 'southwest']

    def __init__(self, model_dir: str = "model"):
        """
        Load all artifacts at initialization time.

        WHY LOAD HERE? (Interview concept: "Cold start vs warm start")
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Loading a TF model from disk takes 2-5 seconds (cold start).
        Once loaded into RAM, each prediction takes ~5ms (warm).
        By loading in __init__, the cold start happens ONCE when the
        server boots. All subsequent requests are warm.

        In AWS Lambda, cold starts are a real problem — the first
        request after idle timeout is slow. In Docker/EC2, the server
        stays running so cold start only happens at deployment.
        """
        model_path = os.path.join(model_dir, "best_insurance_model.h5")
        scaler_path = os.path.join(model_dir, "feature_scaler.pkl")

        # Load model (compile=False avoids deserialization issues with custom losses)
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model.compile(loss='mae', optimizer='adam', metrics=['mae'])

        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Scaler loaded from {scaler_path}")
        print(f"  Input features expected: {self.model.input_shape[1]}")

    def _validate_input(self, data: dict) -> list:
        """
        Validate raw input and return list of errors (empty = valid).

        WHY VALIDATE? (Interview concept: "Defensive programming")
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        In production, users WILL send garbage: missing fields, wrong types,
        negative ages, BMI of 500. Your model will happily predict on garbage
        and return a confident number — that's worse than crashing, because
        the user trusts a wrong answer.

        Always validate BEFORE prediction. Return clear error messages.
        """
        errors = []

        # Required fields
        required = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: '{field}'")

        if errors:
            return errors  # Can't validate further without required fields

        # Type and range checks
        try:
            age = int(data['age'])
            if age < 18 or age > 100:
                errors.append(f"Age must be 18-100, got {age}")
        except (ValueError, TypeError):
            errors.append(f"Age must be an integer, got '{data['age']}'")

        try:
            bmi = float(data['bmi'])
            if bmi < 10 or bmi > 70:
                errors.append(f"BMI must be 10-70, got {bmi}")
        except (ValueError, TypeError):
            errors.append(f"BMI must be a number, got '{data['bmi']}'")

        try:
            children = int(data['children'])
            if children < 0 or children > 10:
                errors.append(f"Children must be 0-10, got {children}")
        except (ValueError, TypeError):
            errors.append(f"Children must be an integer, got '{data['children']}'")

        # Categorical checks
        if data.get('sex', '').lower() not in self.VALID_SEX:
            errors.append(f"Sex must be 'male' or 'female', got '{data.get('sex')}'")

        if data.get('smoker', '').lower() not in self.VALID_SMOKER:
            errors.append(f"Smoker must be 'yes' or 'no', got '{data.get('smoker')}'")

        if data.get('region', '').lower() not in self.VALID_REGION:
            errors.append(f"Region must be one of {self.VALID_REGION}, got '{data.get('region')}'")

        return errors

    def _one_hot_encode(self, data: dict) -> pd.DataFrame:
        """
        Convert raw input dict to one-hot encoded DataFrame.

        WHY NOT USE pd.get_dummies()? (Interview concept: "Train-serve skew")
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        pd.get_dummies() on a single row will only create columns for the
        categories PRESENT in that row. If a patient is male, you get
        'sex_male=1' but NO 'sex_female' column at all. The model expects
        BOTH columns — it was trained with both.

        So we manually construct ALL one-hot columns, ensuring the exact
        same column set regardless of input values. This is called
        "schema enforcement" and it's a top-3 production ML bug source.
        """
        row = {col: 0 for col in self.ORIGINAL_COLUMNS}  # Initialize all to 0

        # Numeric features
        row['age'] = int(data['age'])
        row['bmi'] = float(data['bmi'])
        row['children'] = int(data['children'])

        # One-hot: sex
        sex = data['sex'].lower()
        row[f'sex_{sex}'] = 1

        # One-hot: smoker
        smoker = data['smoker'].lower()
        row[f'smoker_{smoker}'] = 1

        # One-hot: region
        region = data['region'].lower()
        row[f'region_{region}'] = 1

        return pd.DataFrame([row])[self.ORIGINAL_COLUMNS]

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the EXACT same feature engineering used during training.

        WHY "EXACT SAME"? (Interview concept: "Training-serving skew")
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        If you engineer features differently at inference time than at
        training time, your model gets inputs it's never seen — and
        predictions become garbage. This is the #1 bug in deployed ML
        systems, and it's silent (no crash, just wrong answers).

        The rule: if you compute 'age_squared = age ** 2' during training,
        you MUST compute 'age_squared = age ** 2' at inference. Same formula,
        same column order, same dtype.
        """
        X = df.copy()

        # Group 1: Smoker × BMI interactions
        X['smoker_bmi']         = X['smoker_yes'] * X['bmi']
        X['is_obese']           = (X['bmi'] >= 30).astype(int)
        X['smoker_obese']       = X['smoker_yes'] * X['is_obese']
        X['smoker_bmi_above30'] = X['smoker_yes'] * np.maximum(X['bmi'] - 30, 0)

        # Group 2: Age interactions
        X['age_smoker']         = X['age'] * X['smoker_yes']
        X['age_squared']        = X['age'] ** 2
        X['age_bmi']            = X['age'] * X['bmi']

        # Group 3: BMI transformations
        X['bmi_deviation']      = np.abs(X['bmi'] - 24.9)
        X['bmi_squared']        = X['bmi'] ** 2

        # Group 4: Children
        X['has_children']       = (X['children'] > 0).astype(int)

        return X

    def predict(self, data: dict) -> dict:
        """
        Full prediction pipeline: validate → encode → engineer → scale → predict.

        Returns dict with prediction, confidence info, and metadata.
        """
        # Step 1: Validate
        errors = self._validate_input(data)
        if errors:
            return {"success": False, "errors": errors}

        # Step 2: One-hot encode
        df_encoded = self._one_hot_encode(data)

        # Step 3: Engineer features
        df_engineered = self._engineer_features(df_encoded)

        # Step 4: Scale (using the SAME scaler fitted on training data)
        X_scaled = self.scaler.transform(df_engineered)

        # Step 5: Predict
        prediction = self.model.predict(X_scaled, verbose=0).flatten()[0]

        # Step 6: Determine segment (for interpretability)
        is_smoker = data['smoker'].lower() == 'yes'
        bmi = float(data['bmi'])
        if not is_smoker:
            segment = "Non-smoker"
        elif bmi >= 30:
            segment = "Smoker (BMI≥30) — highest cost group"
        else:
            segment = "Smoker (BMI<30)"

        return {
            "success": True,
            "predicted_annual_charge": round(float(prediction), 2),
            "segment": segment,
            "model_info": {
                "name": "Insurance Cost Predictor v1.0",
                "architecture": "128-64-32 Swish Neural Network",
                "test_mae": 1650,
                "test_r2": 0.82,
                "disclaimer": (
                    "This is an estimate with ~$1,650 average error. "
                    "Actual insurance costs depend on factors not in this model "
                    "(pre-existing conditions, medication history, etc.)."
                )
            },
            "input_received": data
        }
