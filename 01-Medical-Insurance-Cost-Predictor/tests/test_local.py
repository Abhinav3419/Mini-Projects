"""
test_local.py — Quick smoke test you can run before deploying.

WHY TEST BEFORE DEPLOY? (Interview concept: "Shift left testing")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Finding a bug locally costs you 5 minutes.
Finding the same bug after deploying to AWS costs you 2 hours
(deploy, wait, check logs, fix, redeploy, wait, check again).

Always test locally BEFORE containerizing. Test in Docker BEFORE pushing to AWS.
This is called "shift left" — catch errors as early as possible.

Run this with: python tests/test_local.py
"""

import sys
import os

# Add parent directory to path so we can import app/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.predictor import InsurancePredictor


def test_predictor():
    """Test the prediction pipeline end-to-end."""
    print("=" * 60)
    print("SMOKE TEST — Local prediction pipeline")
    print("=" * 60)

    # Load predictor
    predictor = InsurancePredictor(model_dir="model")
    print()

    # Test 1: Valid prediction — non-smoker
    print("Test 1: Non-smoker, healthy BMI")
    result = predictor.predict({
        "age": 25, "sex": "male", "bmi": 22.0,
        "children": 0, "smoker": "no", "region": "northwest"
    })
    assert result["success"], f"Failed: {result}"
    charge = result["predicted_annual_charge"]
    print(f"  Prediction: ${charge:,.2f}")
    assert 1000 < charge < 15000, f"Prediction out of range: {charge}"
    print("  ✓ PASSED")

    # Test 2: Valid prediction — smoker, obese
    print("\nTest 2: Smoker, obese (high-cost segment)")
    result = predictor.predict({
        "age": 55, "sex": "female", "bmi": 40.0,
        "children": 0, "smoker": "yes", "region": "southwest"
    })
    assert result["success"], f"Failed: {result}"
    charge = result["predicted_annual_charge"]
    print(f"  Prediction: ${charge:,.2f}")
    assert 30000 < charge < 70000, f"Prediction out of range: {charge}"
    print("  ✓ PASSED")

    # Test 3: Valid prediction — smoker, lean
    print("\nTest 3: Smoker, lean BMI")
    result = predictor.predict({
        "age": 30, "sex": "male", "bmi": 24.0,
        "children": 1, "smoker": "yes", "region": "northeast"
    })
    assert result["success"], f"Failed: {result}"
    charge = result["predicted_annual_charge"]
    print(f"  Prediction: ${charge:,.2f}")
    assert 15000 < charge < 35000, f"Prediction out of range: {charge}"
    print("  ✓ PASSED")

    # Test 4: Missing field
    print("\nTest 4: Missing 'age' field")
    result = predictor.predict({
        "sex": "male", "bmi": 22.0,
        "children": 0, "smoker": "no", "region": "northwest"
    })
    assert not result["success"], "Should have failed"
    assert any("age" in e for e in result["errors"])
    print(f"  Errors: {result['errors']}")
    print("  ✓ PASSED (correctly rejected)")

    # Test 5: Invalid BMI
    print("\nTest 5: BMI out of range (999)")
    result = predictor.predict({
        "age": 30, "sex": "male", "bmi": 999,
        "children": 0, "smoker": "no", "region": "northwest"
    })
    assert not result["success"], "Should have failed"
    print(f"  Errors: {result['errors']}")
    print("  ✓ PASSED (correctly rejected)")

    # Test 6: Invalid smoker value
    print("\nTest 6: Invalid smoker value ('maybe')")
    result = predictor.predict({
        "age": 30, "sex": "male", "bmi": 25.0,
        "children": 0, "smoker": "maybe", "region": "northwest"
    })
    assert not result["success"], "Should have failed"
    print(f"  Errors: {result['errors']}")
    print("  ✓ PASSED (correctly rejected)")

    # Test 7: Segment classification
    print("\nTest 7: Segment classification check")
    for smoker, bmi, expected_seg in [
        ("no",  25.0, "Non-smoker"),
        ("yes", 25.0, "Smoker (BMI<30)"),
        ("yes", 35.0, "Smoker (BMI≥30)"),
    ]:
        result = predictor.predict({
            "age": 30, "sex": "male", "bmi": bmi,
            "children": 0, "smoker": smoker, "region": "northwest"
        })
        assert expected_seg in result["segment"], \
            f"Expected '{expected_seg}' in '{result['segment']}'"
    print("  All segments classified correctly")
    print("  ✓ PASSED")

    print("\n" + "=" * 60)
    print("ALL 7 TESTS PASSED ✓")
    print("=" * 60)
    print("\nYour prediction pipeline is working correctly.")
    print("Next step: start the FastAPI server with:")
    print("  uvicorn app.main:app --reload")


if __name__ == "__main__":
    test_predictor()
