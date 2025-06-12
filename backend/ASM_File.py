import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import joblib
import firebase_admin
from firebase_admin import credentials, db
import os
from dotenv import load_dotenv
from pathlib import Path


'''Initialising ML model'''
model = joblib.load('battery_model.pkl')
x_scaler = joblib.load('x_scaler.pkl')
y_scaler = joblib.load('y_scaler.pkl')

'''Initialising Firebase '''
# Load .env from the same directory as this script
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

firebase_config = {
    "type": os.getenv("FIREBASE_TYPE"),
    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL"),
    "universe_domain": "googleapis.com"
}

cred = credentials.Certificate(firebase_config)
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://flutter-with-firebase-te-1f81e-default-rtdb.asia-southeast1.firebasedatabase.app"
})

'''Handle Raw data'''
def write_raw_data(filename):
    try:
        df = pd.read_excel(filename)

        voltage = df['voltage_v'].tolist()
        current = df['current_a'].tolist()
        time = df['time_s'].tolist()

        # Create structured data for logs
        data_log = {
            '0_timestamp': datetime.now().isoformat(),
            'time_s': time,
            'voltage_v': voltage,
            'current_a': current
        }

        # Upload to /raw_data (overwrite old data)
        db.reference('/raw_data/time_s').set(time)
        db.reference('/raw_data/current_a').set(current)
        db.reference('/raw_data/voltage_v').set(voltage)

        # Upload to /raw_data_logs with current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        db.reference(f'/raw_data_logs/{timestamp}').set(data_log)

        return {
            "status": "success",
            "message": f"{filename} uploaded to /raw_data and /raw_data_logs/{timestamp}",
            "log_id": timestamp
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to upload data from {filename}: {str(e)}"
        }

### Delete the timestamps from the raw_data_logs at Firebase
def delete_raw_data_log(log_id: str) -> dict:
    try:
        ref = db.reference(f"/raw_data_logs/{log_id}")
        if ref.get() is None:
            return {
                "status": "error",
                "message": f"Log ID '{log_id}' does not exist."
            }

        ref.delete()
        return {
            "status": "success",
            "message": f"Log ID '{log_id}' has been deleted successfully."
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"An error occurred while deleting '{log_id}': {str(e)}"
        }

### Get all the timestamps from the raw_data_logs at Firebase
def get_all_log_id():
    log_ref = db.reference('/raw_data_logs')
    log_snapshot = log_ref.get()

    if not log_snapshot:
        return {
            "status": "error",
            "message": "No logs found."
        }

    log_id = list(log_snapshot.keys())
    return {
        "status": "success",
        "log_count": len(log_id),
        "timestamps": sorted(log_id, reverse=True)
    }

### Only display the raw data by passing timestamp of the data (now: recent raw data, timestamp: raw data logs)
def read_raw_data(log_id: str = "now", key: str = None):

    data_ref = db.reference('/raw_data') if log_id == "now" else db.reference(f'/raw_data_logs/{log_id}')
    data = data_ref.get()
    
    if not data:
        return {
            "log_id": log_id,
            "status": "error",
            "message": "No data found",
            "available_keys": []
        }

    available_keys = list(data.keys())

    # If specific key is requested, return only that key
    if key:
        if key in data:
            return {
                "log_id": log_id,
                "status": "success",
                "message": (f"Returning only key: {key}"),
                "available_keys": available_keys,
                "data": {key: data[key]}
            }
        else:
            return {
                "log_id": log_id,
                "status": "error",
                "message": f"Key '{key}' not found",
                "available_keys": available_keys
            }

    # If no key is requested, return time, voltage, and current
    try:
        return {
            "log_id": log_id,
            "status": "success",
            "message": f"Returning all key:",
            "available_keys": available_keys,
            "data": {
                "time_s": data["time_s"],
                "voltage_v": data["voltage_v"],
                "current_a": data["current_a"]
            }
        }
    
    except KeyError as e:
        return {
            "log_id": log_id,
            "status": "error",
            "message": f"Missing key: {str(e)}",
            "available_keys": available_keys
        }


'''Pre processing raw data; collecting important features'''
def analyze_battery_features(data: dict, nominal_capacity: float):
    ### Pre-process battery data from Firebase: Compute important battery features.
    summary_data = []

    # Validate keys
    required_keys = ['time_s', 'voltage_v', 'current_a']
    if not all(key in data for key in required_keys):
        raise ValueError("Missing required keys in Firebase data.")

    # Convert to NumPy-friendly format
    data_dict = {
        'time': data['time_s'],
        'voltage': data['voltage_v'],
        'current': data['current_a'],
    }

    # Calculate features
    (   capacity, c_rate, 
        max_V, min_V, mean_V, 
        max_I, min_I, mean_I, 
        droprate_V
    )= calc_features(data_dict, nominal_capacity)

    summary_data.append({
        'Capacity (mAh)': capacity,
        'Avg C-rate': c_rate,
        'Max Voltage': max_V,
        'Min Voltage': min_V,
        'Mean Voltage': mean_V,
        'Max Current': max_I,
        'Min Current': min_I,
        'Mean Current': mean_I,
        'Voltage drop rate per hour': droprate_V
    })

    return pd.DataFrame(summary_data)

def calc_features(data: dict, nominal_capacity: float):
    capacity = np.trapezoid(data['current'], data['time']) / 3600  # convert to Ah
    currents = np.abs(data['current'])
    c_rate = currents / nominal_capacity
    max_V = np.max(data['voltage'])
    min_V = np.min(data['voltage'])
    mean_V = np.mean(np.abs(data['voltage']))
    max_I = np.max(data['current'])
    min_I = np.min(data['current'])
    mean_I = np.mean(np.abs(data['current']))
    droprate_V = (data['voltage'][0] - data['voltage'][-1]) / 24
    
    return (
        capacity, np.mean(c_rate),
        max_V, min_V, mean_V, 
        max_I, min_I, mean_I, 
        droprate_V
    )

# Define expected features in correct order
ML_FEATURE_ORDER = [
    'Capacity (mAh)', 'Avg C-rate',
    'Min Voltage', 'Mean Voltage',
    'Max Current', 'Min Current', 'Mean Current',
    'Voltage drop rate per hour'
]


''' Prediction of SOH and Battery Condition and RUL using Ml model'''
def pred_soh(ml_feature_data):
    input_scaled = x_scaler.transform(ml_feature_data)
    predicted_scaled = model.predict(input_scaled)
    predicted_soh = y_scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).ravel()[0]    
    condition, rul = battery_condition(predicted_soh)
    return predicted_soh, condition, rul

def battery_condition(soh):
    def soh_recommendation(soh_val):
        if soh_val >= 90:
            return "Excellent"
        elif 80 <= soh_val < 90:
            return "Good"
        elif 70 <= soh_val < 80:
            return "Degraded - Monitor"
        elif 60 <= soh_val < 70:
            return "Poor - Needs Attention"
        else:
            return "Critical - Maintenance Required"

    def estimate_rul(soh_val):
        drop_per_cycle = 0.2  # Empirical estimate
        return int((soh_val - 50) / drop_per_cycle) if soh_val > 50 else 0

    return soh_recommendation(soh), estimate_rul(soh)


'''Handle Usage Count (Charging logs)'''
def read_charging_logs():
    ref = db.reference('/charging_logs')
    raw_data = ref.get()

    if not raw_data:
        return []  # Return an empty list if no logs

    try:
        # Convert all ISO 8601 strings to datetime objects
        datetime_array = [datetime.fromisoformat(ts) for ts in raw_data.values()]
        datetime_array.sort()  # Optional: sort chronologically
        return datetime_array

    except Exception as e:
        print(f"[ERROR] Failed to parse charging logs: {e}")
        return []

def display_charging_by_week(charging_logs_array, start_week=1, end_week=None):
    weekly_log = defaultdict(list)
    for dt in charging_logs_array:
        dt_date = dt.date()
        monday = dt_date - timedelta(days=dt_date.weekday())  # Normalize to Monday
        weekly_log[monday].append(dt)

    sorted_weeks = sorted(weekly_log.items(), key=lambda x: x[0])
    total_weeks = len(sorted_weeks)

    if isinstance(start_week, str) and start_week.lower() == 'end':
        start_week = total_weeks
    else:
        start_week = int(start_week)

    if end_week is None:
        end_week = start_week
    elif isinstance(end_week, str) and end_week.lower() == 'end':
        end_week = total_weeks
    else:
        end_week = int(end_week)

    # Out of range weeks
    if start_week > total_weeks or end_week > total_weeks:
        if start_week == end_week:
            print(f"\n Only {total_weeks} week(s) of data available. You requested week {start_week}.")
        else:
            print(f"\n Only {total_weeks} week(s) of data available. You requested week {start_week} to {end_week}.")


    start_week = max(1, min(start_week, total_weeks))
    end_week = max(1, min(end_week, total_weeks))

    reversed_weeks = list(reversed(sorted_weeks))

    # Build output string instead of printing
    output = ""

    if start_week == end_week:
        output += f"\n Showing charging logs for week {start_week} \n"
    else:
        output += f"\n Showing charging logs from week {start_week} to {end_week})\n"
        
    for idx, (monday, logs) in enumerate(reversed_weeks, start=1):
        if start_week <= idx <= end_week:
            output += f" Week {idx} (Starting on {monday.strftime('%A, %d %B %Y')}):\n"
            for dt in sorted(logs):
                output += f"  - {dt.strftime('%A, %d %B %Y %I:%M %p')}\n"
            output += "\n"

    return output

def write_charging_logs(timestamp_str: str = None): 
    try:
        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
    except ValueError:
        return {
            "status": "error",
            "message": "Invalid timestamp format. Please use ISO 8601 format (e.g., 2025-06-09T13:45:00)"
        }

    iso_timestamp = timestamp.isoformat()
    db.reference('/charging_logs').push(iso_timestamp)

    return {
        "status": "success",
        "message": f"Logged charging timestamp: {iso_timestamp}"
    }

def delete_charging_logs(timestamp: str = None, week: int = None):
    ref = db.reference('/charging_logs')
    all_logs = ref.get()

    if not all_logs:
        return {"status": "error", "message": "No charging logs found."}

    deleted_keys = []

    if timestamp:
        for key, value in all_logs.items():
            if value == timestamp:
                ref.child(key).delete()
                deleted_keys.append(key)
                return {"status": "success", "message": f"Deleted timestamp {timestamp}"}
        return {"status": "error", "message": f"Timestamp {timestamp} not found."}

    elif week is not None:
        # Parse all ISO timestamps into datetime objects
        logs_by_key = {}
        for key, value in all_logs.items():
            try:
                dt = datetime.fromisoformat(value)
                logs_by_key[key] = dt
            except:
                continue  # Skip invalid ones

        # Group by week
        weekly_log = defaultdict(list)
        for key, dt in logs_by_key.items():
            monday = dt.date() - timedelta(days=dt.weekday())
            weekly_log[monday].append((key, dt))

        sorted_weeks = sorted(weekly_log.items(), key=lambda x: x[0])
        total_weeks = len(sorted_weeks)

        if week > total_weeks or week < 1:
            return {
                "status": "error",
                "message": f"Only {total_weeks} weeks of data. You requested week {week}."
            }

        reversed_weeks = list(reversed(sorted_weeks))
        selected_week = reversed_weeks[week - 1]

        for key, dt in selected_week[1]:
            ref.child(key).delete()
            deleted_keys.append(key)

        return {
            "status": "success",
            "message": f"Deleted {len(deleted_keys)} charging logs from week {week}.",
            "deleted_keys": deleted_keys
        }

    return {"status": "error", "message": "Please provide either timestamp or week number."}


'''Determination Day to discharge (dtd) using usage count'''
def calc_dtd(charging_logs_array):
    # Group charging days into week buckets (Monday start)
    weekly_charging_log = defaultdict(set)
    today = datetime.now().date()

    for dt in charging_logs_array:
        dt_date = dt.date()
        week_start = dt_date - timedelta(days=dt_date.weekday())
        weekly_charging_log[week_start].add(dt_date)

    # Get last 3 weeks (this week included)
    sorted_weeks = sorted(weekly_charging_log.keys(), reverse=True)
    recent_weeks = sorted_weeks[:3]

    # Weights for each week (most recent = heaviest)
    weights = [0.6, 0.3, 0.1]
    weighted_sum = 0
    charging_summary = []

    for i, week in enumerate(recent_weeks):
        unique_days = weekly_charging_log[week]
        count = len(unique_days)
        weight = weights[i]
        weighted_sum += count * weight
        charging_summary.append({
            "week_start": week.isoformat(),
            "charged_days": count,
            "weight": weight
        })

    # Decide waiting days based on weighted average
    if weighted_sum <= 1.5:
        usage_class = "Low usage"
        waiting_days = 1
    elif weighted_sum <= 3:
        usage_class = "Medium usage"
        waiting_days = 2
    else:
        usage_class = "High usage"
        waiting_days = 4

    # Analyze most frequent day in past 7 days
    seven_day_window = today - timedelta(days=6)
    recent_charging = [t for t in charging_logs_array if t.date() >= seven_day_window]
    day_names = [t.strftime('%A') for t in recent_charging]
    most_frequent_day = Counter(day_names).most_common(1)[0][0] if day_names else None

    # Recommend discharge day
    day_to_discharge = datetime.now() + timedelta(days=waiting_days)
    pushed = False
    if day_to_discharge.strftime('%A') == most_frequent_day:
        day_to_discharge += timedelta(days=1)
        pushed = True

    return {
        "usage_score": round(weighted_sum, 2),
        "usage_classification": usage_class,
        "charging_summary": charging_summary,
        "waiting_days": waiting_days,
        "most_frequent_charging_day": most_frequent_day,
        "pushed_due_to_overlap": pushed,
        "recommended_discharge_day": day_to_discharge,
        "recommended_discharge_day_display" : day_to_discharge.strftime("%A, %d %B %Y %I:%M %p"),
    }

def write_dtd(discharge_day_str: datetime):
    discharge_day_str = discharge_day_str.isoformat()
    db.reference('/discharge_day').set(discharge_day_str) # Overwrite or update with latest discharge day
    print(f"Discharge day saved to Firebase: {discharge_day_str}")

def read_dtd():
    """Reads the discharge day from Firebase and returns it as a datetime object."""
    ref = db.reference('/discharge_day')
    discharge_iso = ref.get()
    if discharge_iso:
        return datetime.fromisoformat(discharge_iso)
    return None



'''Estimate SOC with current Voltage level'''
def est_soc(voltage_reading: float) -> float:
    voltage_soc_data = np.array([
        [12.6, 100],
        [12.45, 95],
        [12.3, 90],
        [12.0, 80],
        [11.7, 70],
        [11.4, 65],
        [11.1, 60],
        [10.8, 50],
        [10.5, 40],
        [10.2, 30],
        [9.9, 20],
        [9.6, 10],
        [9.0, 0]
    ])

    pack_voltage, soc = voltage_soc_data[:, 0], voltage_soc_data[:, 1]
    soc_estimator_3s = interp1d(pack_voltage, soc, kind='linear', fill_value='extrapolate')
    voltage_reading = np.clip(voltage_reading, pack_voltage.min(), pack_voltage.max())

    return float(soc_estimator_3s(voltage_reading))










''' # Predict SOH, RUL and battery condition
### Implementations
ml_features = analyze_battery_features(0, 'Rawfile1.xlsx', 4200)
print(ml_features)

input_data = ml_features[ML_FEATURE_ORDER].to_numpy()
print('')
pred_soh(input_data)

input_data = np.array([[2.3, 0.00002,
                        11.07, 11.56,
                        0.31, 0.056, 0.09707,
                        0.062]])
print('')
pred_soh(input_data)
'''


''' ### Discharging day algorithm
Charging_days_array = [
    datetime.now() - timedelta(days = 22),
    datetime.now() - timedelta(days = 21),
    datetime.now() - timedelta(days = 20),
    datetime.now() - timedelta(days = 19),
    datetime.now() - timedelta(days = 18),
    datetime.now() - timedelta(days = 17),
    datetime.now() - timedelta(days = 16),
    datetime.now() - timedelta(days = 15),
    datetime.now() - timedelta(days = 14),
    datetime.now() - timedelta(days = 13),
    datetime.now() - timedelta(days = 12),
    datetime.now() - timedelta(days = 11),
    datetime.now() - timedelta(days = 10),
    datetime.now() - timedelta(days = 9),
    datetime.now() - timedelta(days = 8),
    datetime.now() - timedelta(days = 7),
    datetime.now() - timedelta(days = 6),
    datetime.now() - timedelta(days = 5),
    datetime.now() - timedelta(days = 4),
    datetime.now() - timedelta(days = 3),
    datetime.now() - timedelta(days = 2),
    datetime.now() - timedelta(days = 1),
]

# High usage
Charging_days_array1 = [
    datetime.now() - timedelta(days = 21),
    datetime.now() - timedelta(days = 19),
    datetime.now() - timedelta(days = 10),
    datetime.now() - timedelta(days = 9),
    datetime.now() - timedelta(days = 7),
    datetime.now() - timedelta(days = 6),
    datetime.now() - timedelta(days = 5),
    datetime.now() - timedelta(days = 4),
    datetime.now() - timedelta(days = 4),
    datetime.now() - timedelta(days = 3),
    datetime.now() - timedelta(days = 3),
    datetime.now() - timedelta(days = 2),
    datetime.now() - timedelta(days = 2),
    datetime.now() - timedelta(days = 2),
    datetime.now() - timedelta(days = 1),
    datetime.now() - timedelta(days = 1),
]

# Low usage
Charging_days_array2 = [
    datetime.now() - timedelta(days = 4), # This week
    datetime.now() - timedelta(days = 5), # This week

    datetime.now() - timedelta(days = 5) # Last week
]

# Example usage
log_charging_event(False, Charging_days_array)
summary = display_charging_by_week(Charging_days_array, 1,5)  # All available weeks
print(summary)
print('Discharging Event 1')
Day_to_discharge = calc_day_to_discharge(Charging_days_array)
'''

''' 
X scaler = 
2055 total rows of data
Train: 90% ; 1847 data
Validate: 5%; 208 data
Test: 5% ; 208 data

Features used to train the model:
Capacity (mAh), Avg C-rate
Min Voltage, Mean Voltage
Max Current, Min Current, Mean Current
Voltage drop rate per hour

Feature Importances:
Max Current: 0.4077149857
Min Current: 0.1525127733
Avg C-rate: 0.1356651081
Mean Current: 0.1254049009
Capacity (mAh): 0.1206626951
Voltage drop rate per hour: 0.0227843044
Mean Voltage: 0.0181169430
Min Voltage: 0.0171382895

Validation result printed
Validation MSE: 0.0557328897
Validation R²: 0.8548052648

Test result printed
Test MSE: 0.0490561433
Test R²: 0.8803295266

Remarks
1. MSE ise very low (Good)
2. R2 is near to 1 (Good)
3. Model ready to deploy into backend
4. RUL is not accurate; revise
5. Revise recommendations based on SOH (e.g., "Battery needs maintenance").

"WOW" FACTOR
1. DIY
'''