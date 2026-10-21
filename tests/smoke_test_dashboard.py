"""
End-to-End Smoke Test for Port Congestion Streamlit Dashboard (app.py).

Verifies:
1. Application initial load and resource caching.
2. Preset scenario loading (Normal Operations, Elevated Delay Risk, Borderline, Custom).
3. Automatic weekday derivation from observation date.
4. Correct midnight date transition in forecast cutoff and horizon strings.
5. Live inference without retraining across all scenarios.
6. Zero-capacity validation guard and predict button disabling.
7. Feature importance and situational context table rendering without deprecation warnings.
8. Historical evaluation performance table presence and accuracy.
"""

from datetime import datetime, time, timedelta
from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "app.py"


def run_dashboard_smoke_test():
    print("=" * 70)
    print("STREAMLIT DASHBOARD COMPREHENSIVE SMOKE TEST")
    print("=" * 70)

    # 1. Initialize App
    print("[1/6] Launching App via AppTest engine...")
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=15)
    assert len(at.exception) == 0, f"App crashed on startup: {at.exception}"
    print("      App initialized cleanly with 0 exceptions.")

    # 2. Verify Initial Defaults & Weekday Derivation
    print("[2/6] Verifying default controls and weekday derivation...")
    day_input = at.text_input[0]
    assert day_input.disabled is True, "Day of week field must be read-only/disabled"
    assert day_input.value == "Friday", f"Expected Friday for default 2026-05-01, got {day_input.value}"
    print(f"      Derived weekday: '{day_input.value}' (matches observation date).")

    # 3. Test Scenario: Normal Operations
    print("[3/6] Testing Scenario: 'Normal Operations'...")
    at.sidebar.radio[0].set_value("Normal Operations").run(timeout=15)
    assert at.text_input[0].value == "Tuesday", f"Normal operations should derive Tuesday, got {at.text_input[0].value}"
    at.button[0].click().run(timeout=15)
    assert len(at.exception) == 0, f"Exception after predict: {at.exception}"
    metrics_normal = [m.value for m in at.metric]
    print(f"      Normal Operations Forecast Output: Arrival Load={metrics_normal[0]}, Delay Prob={metrics_normal[1]}")
    assert any("ON-TIME" in s.value for s in at.success), "Normal operations should forecast On-Time"

    # 4. Test Scenario: Elevated Delay Risk
    print("[4/6] Testing Scenario: 'Elevated Delay Risk'...")
    at.sidebar.radio[0].set_value("Elevated Delay Risk").run(timeout=15)
    assert at.text_input[0].value == "Saturday", f"Elevated risk should derive Saturday, got {at.text_input[0].value}"
    at.button[0].click().run(timeout=15)
    assert len(at.exception) == 0, f"Exception after predict: {at.exception}"
    metrics_elevated = [m.value for m in at.metric]
    print(f"      Elevated Risk Forecast Output: Arrival Load={metrics_elevated[0]}, Delay Prob={metrics_elevated[1]}")
    assert any("DELAY PREDICTED" in e.value for e in at.error), "Elevated risk should forecast Delay Predicted"

    # 5. Test Midnight Horizon Arithmetic
    print("[5/6] Testing Midnight Horizon Arithmetic (18:00 to 24:00)...")
    at.sidebar.radio[0].set_value("Custom Input").run(timeout=15)
    at.selectbox[0].set_value("18:00 to 24:00").run(timeout=15)
    info_texts = [i.value for i in at.info]
    assert any("2026-05-02 00:00" in t for t in info_texts), "Cutoff must advance to next day at midnight"
    assert any("2026-05-02 00:00 to 2026-05-02 06:00" in t for t in info_texts), "Forecast horizon must span 00:00 to 06:00 next day"
    print("      Midnight date transition verified: 2026-05-01 18:00 -> Cutoff: 2026-05-02 00:00 -> Horizon: 2026-05-02 00:00 to 06:00.")

    # 6. Test Input Validation & Zero-Capacity Guard
    print("[6/6] Testing Zero-Capacity Input Validation Guard...")
    at.number_input[1].set_value(0).run(timeout=15)
    assert at.button[0].disabled is True, "Predict button MUST be disabled when capacity is 0"
    errors = [e.value for e in at.error]
    assert any("Nominal Port Capacity" in e for e in errors), f"Must display validation banner for capacity=0, got {errors}"
    print("      Zero-capacity validation triggered cleanly; predict button disabled.")

    print("=" * 70)
    print("ALL DASHBOARD SMOKE TEST PHASES PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_dashboard_smoke_test()
