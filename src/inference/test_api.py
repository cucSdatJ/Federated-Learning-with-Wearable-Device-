import argparse
import json
import requests


def pretty_print(title: str, response: requests.Response):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print("Status code:", response.status_code)

    try:
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except Exception:
        print(response.text)


def test_health(base_url: str):
    url = f"{base_url}/health"
    response = requests.get(url, timeout=10)
    pretty_print("GET /health", response)


def test_labels(base_url: str):
    url = f"{base_url}/labels"
    response = requests.get(url, timeout=10)
    pretty_print("GET /labels", response)


def test_device_activity_map(base_url: str):
    url = f"{base_url}/device-activity-map"
    response = requests.get(url, timeout=10)
    pretty_print("GET /device-activity-map", response)


def test_predict(base_url: str):
    url = f"{base_url}/predict"

    payload = {
        "heart_rate": 128.0,
        "hr_rolling_mean": 125.0,
        "hr_rolling_std": 4.2,
        "acc_magnitude": 10.5,
        "act_rest": 0,
        "act_walk": 0,
        "act_brisk": 1,
        "act_run": 0
    }

    response = requests.post(url, json=payload, timeout=10)
    pretty_print("POST /predict", response)


def test_predict_from_sensor_group(base_url: str):
    url = f"{base_url}/predict-from-sensor"

    payload = {
        "heart_rate": 132.0,
        "hr_window": [126.0, 128.0, 130.0, 131.0, 132.0],
        "acc_x": 0.8,
        "acc_y": 9.6,
        "acc_z": 1.5,
        "activity_group": "brisk",
        "hour_of_day": 14.5
    }

    response = requests.post(url, json=payload, timeout=10)
    pretty_print("POST /predict-from-sensor (activity_group)", response)


def test_predict_from_sensor_activity_id(base_url: str):
    url = f"{base_url}/predict-from-sensor"

    payload = {
        "heart_rate": 112.0,
        "hr_window": [108.0, 110.0, 111.0, 112.0, 113.0],
        "acc_x": 0.4,
        "acc_y": 9.7,
        "acc_z": 0.9,
        "activity_id": 4,
        "hour_of_day": 10.0
    }

    response = requests.post(url, json=payload, timeout=10)
    pretty_print("POST /predict-from-sensor (activity_id)", response)


def test_predict_from_sensor_device_code(base_url: str):
    url = f"{base_url}/predict-from-sensor"

    payload = {
        "heart_rate": 138.0,
        "hr_window": [130.0, 133.0, 135.0, 137.0, 138.0],
        "acc_x": 0.9,
        "acc_y": 9.8,
        "acc_z": 1.8,
        "device_activity_code": 2,
        "hour_of_day": 15.0
    }

    response = requests.post(url, json=payload, timeout=10)
    pretty_print("POST /predict-from-sensor (device_activity_code)", response)


def test_invalid_predict(base_url: str):
    """
    Test lỗi one-hot không hợp lệ: nhiều hơn 1 activity bật cùng lúc.
    """
    url = f"{base_url}/predict"

    payload = {
        "heart_rate": 120.0,
        "hr_rolling_mean": 118.0,
        "hr_rolling_std": 3.5,
        "acc_magnitude": 10.1,
        "act_rest": 0,
        "act_walk": 1,
        "act_brisk": 1,  # invalid
        "act_run": 0
    }

    response = requests.post(url, json=payload, timeout=10)
    pretty_print("POST /predict (invalid onehot)", response)

def test_predict_from_sensor_ok_case(base_url: str):
    url = f"{base_url}/predict-from-sensor"

    payload = {
        "heart_rate": 82.0,
        "hr_window": [78.0, 80.0, 81.0, 82.0, 83.0],
        "acc_x": 0.2,
        "acc_y": 9.7,
        "acc_z": 0.3,
        "activity_group": "rest",
        "hour_of_day": 9.0
    }

    response = requests.post(url, json=payload, timeout=10)
    pretty_print("POST /predict-from-sensor (OK case)", response)


def test_predict_from_sensor_high_case(base_url: str):
    url = f"{base_url}/predict-from-sensor"

    payload = {
        "heart_rate": 180.0,
        "hr_window": [170.0, 174.0, 176.0, 178.0, 180.0],
        "acc_x": 2.5,
        "acc_y": 11.5,
        "acc_z": 3.0,
        "activity_group": "run",
        "hour_of_day": 17.0
    }

    response = requests.post(url, json=payload, timeout=10)
    pretty_print("POST /predict-from-sensor (HIGH case)", response)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL of FastAPI server",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    print(f"[INFO] Testing API at: {base_url}")

    try:
        test_health(base_url)
        test_labels(base_url)
        test_device_activity_map(base_url)

        test_predict(base_url)
        test_predict_from_sensor_group(base_url)
        test_predict_from_sensor_activity_id(base_url)
        test_predict_from_sensor_device_code(base_url)
        test_predict_from_sensor_ok_case(base_url)
        test_predict_from_sensor_high_case(base_url)
        # optional negative test
        test_invalid_predict(base_url)

        print("\n[DONE] API test finished.")

    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Cannot connect to API server.")
        print("Please start the FastAPI server first:")
        print("uvicorn src.inference.api_server:app --reload")

    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")


if __name__ == "__main__":
    main()