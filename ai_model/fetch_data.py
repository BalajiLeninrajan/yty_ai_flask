import pandas as pd
import requests
import datetime

def _call_pi_api(date: datetime.date) -> requests.Response:
    return requests.post(
        "https://ytygroup.app/pidetail/api/listofPI.php",
        json={
            "dt": date.strftime("%Y-%m-%d")
        }
    )

def fetch_pi_raw_data(dates: pd.DatetimeIndex) -> list[dict[str, str]]:
    data: list[dict[str, str]] = []

    date: pd.Timestamp
    for date in dates:
        response: requests.Response = _call_pi_api(date.date())
        data.extend(response.json())

    return data

def _call_insp_api(date: datetime.date) -> requests.Response:
    return requests.post(
        "https://ytygroup.app/pidetail/api/listofInsp.php",
        json={
            "dt": date.strftime("%Y-%m-%d")
        }
    )

def fetch_insp_raw_data(dates: pd.DatetimeIndex) -> list[dict[str, str]]:
    data: list[dict[str, str]] = []

    date: pd.Timestamp
    for date in dates:
        response: requests.Response = _call_pi_api(date.date())
        data.extend(response.json())

    return data

def _call_qa_api(date: datetime.date) -> requests.Response:
    return requests.post(
        "https://ytygroup.app/pidetail/api/listofQA.php",
        json={
            "dt": date.strftime("%Y-%m-%d")
        }
    )

def fetch_qa_raw_data(dates: pd.DatetimeIndex) -> list[dict[str, str]]:
    data: list[dict[str, str]] = []

    date: pd.Timestamp
    for date in dates:
        response: requests.Response = _call_pi_api(date.date())
        data.extend(response.json())

    return data

def preprocess_data(raw_data: list[dict[str, str]]) -> list[str]:
    data: list[str] = []
    for data_point in raw_data:
        data.append(", ".join([f"{key.replace("_", " ")}: {value}" for key, value in data_point.items()]))
    return data
