from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse, JSONResponse, FileResponse
import uvicorn
from datetime import datetime
from typing import Optional
from ASM_File import write_raw_data, delete_raw_data_log, get_all_log_id, read_raw_data
from ASM_File import analyze_battery_features, ML_FEATURE_ORDER, pred_soh
from ASM_File import write_charging_logs, read_charging_logs, display_charging_by_week, delete_charging_logs
from ASM_File import calc_dtd,  write_dtd, read_dtd
from ASM_File import est_soc


''' 
Queries

20250607_200556 # Poor SOH
20250607_200743 # Excellent SOH
20250607_200928 # Excellent SOH

key = time_s, current_a, voltage_v ; use either of this to see the key
log_id = recent = now // past = e.g. 20250607_200556
'''

app = FastAPI(
    title="Battery ASM API",
    description="Predicts battery State of Health (SOH), condition, RUL & Day to Discharge.",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Raw Data",
            "description": "Endpoints related to uploading, displaying, and deleting raw battery logs."
        },
        {
            "name": "Machine Learning",
            "description": "Battery SOH, condition and RUL prediction endpoints."
        },
        {
            "name": "Usage Pattern Data",
            "description": "Endpoints related to uploading, displaying, and deleting charging logs."            
        },
        {
            "name": "Day to Discharge",
            "description": "Estimate discharge days."
        },
        {
            "name": "SOC Estimation",
            "description": "Estimate battery SOC from voltage readings."
        }
    ]
)

@app.get("/")
async def root():
    return {"message": "Battery ASM API is up and running!"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("favicon.ico")

'''Handle raw data'''
'''Write via excel, delete by passing its log_id, displaying past data's log_id, display raw_data by passing its log_id, '''
### Display RAW data logs Timestamps
@app.get("/display_log_id", tags=["Raw Data"])
async def display_raw_data_log_id():
    return JSONResponse(content=get_all_log_id())

### Display RAW data from firebase database
@app.get("/display", tags=["Raw Data"])
async def display_raw_data(log_id: str = "now", key: str = None):
    result = read_raw_data(log_id, key)

    if result["status"] != "success":
        return result  # return the error message as it is

    if key:
        return result # key was specified, just return it as-is

    # key = None : Combine rows
    data = result["data"]
    try:
        combined_rows = list(zip(data["time_s"], data["voltage_v"], data["current_a"]))
    except KeyError as e:
        return {
            "log_id": log_id,
            "status": "error",
            "message": f"Missing key when combining rows: {str(e)}"
        }

    return {
        "log_id": log_id,
        "status": "success",
        "message": f"Returning all key when combining rows:",
        "available_keys": result["available_keys"],
        "data": {
            "columns": ["time_s", "voltage_v", "current_a"],
            "combined_rows": combined_rows
        }
    }

### Write RAW data by excel name
@app.post("/write_excel_raw_data", tags=["Raw Data"])
async def write_excel_raw_data(filename: str = Query(..., description="Excel filename without extension (.xlsx)")):
    filepath = f"{filename}.xlsx"
    result = write_raw_data(filepath)

    return {
        "status": result["status"],
        "message": result["message"],
        "log_id": result.get("log_id", None)
    }

### Delete RAW data by its log_id
@app.delete("/delete_log", tags=["Raw Data"])
async def delete_log(log_id: str = Query(..., description="The log_id to delete, e.g. 20250607_200556")):
    return delete_raw_data_log(log_id)


'''Machine learning'''
'''Obtain feature data from raw data then predict SOH, RUL and Battery condition'''
### Process RAW data to get FEATURE data
@app.get("/process", tags=["Machine Learning"])
async def obtain_feature_data(log_id: str = "now"):
    result = read_raw_data(log_id, None)
    if result["status"] != "success":
        return result
    
    data_raw = result["data"]
    feature_data = analyze_battery_features(data_raw, 4200)
    return {
        "feature data": feature_data.to_dict(orient="records")
    }

### Predict SOH, BATTERY CONDITION and RUL
@app.get("/predict", tags=["Machine Learning"])
async def predict(log_id: str = "now"):
    result = read_raw_data(log_id, None)
    if result["status"] != "success":
        return result
    
    data_raw = result["data"]
    feature_data = analyze_battery_features(data_raw, 4200)
    ml_features_data = feature_data[ML_FEATURE_ORDER].to_numpy()
    
    soh, condition, rul = pred_soh(ml_features_data)
    
    return {
        "SOH": f"{soh:.2f}%",
        "Battery Condition": condition,
        "RUL": f"{rul} cycles"
    }


'''Handle Usage Pattern data'''
'''Write, delete via ISO timeformat, display by weeks '''
### Display Summary of Charging Logs
@app.get("/display_charging_logs", response_class=PlainTextResponse, tags = ["Usage Pattern Data"])
async def display_charging_days(start_week: int = 1, end_week: str = "end"):
    charge_logs = read_charging_logs()
    disp_charging = display_charging_by_week(charge_logs, start_week, end_week)
    return disp_charging

### Write to Charging Logs in Firebase
@app.post("/write_charging_log", tags=["Usage Pattern Data"])
async def write_charging_log(timestamp: Optional[str] = Query(None, description="ISO 8601 datetime, e.g. 2025-06-09T14:30:00")):
    return write_charging_logs(timestamp)

### Delete the charging logs in Firebase
@app.delete("/delete_charging_log", tags=["Usage Pattern Data"])
async def delete_charging_log(
    timestamp: Optional[str] = Query(None, description="ISO timestamp to delete"),
    week: Optional[int] = Query(None, description="Week number to delete (latest week is 1)")
):
    return delete_charging_logs(timestamp=timestamp, week=week)


'''Estimation of day to discharge(dtd)'''
'''Read the Charging log and estimate day to discharge'''
### Calculate day of discharge based on Charging Logs
@app.get("/day_of_discharge", tags = ["Day to Discharge"])
async def day_to_discharge():
    charge_logs = read_charging_logs()
    ASM_output = calc_dtd(charge_logs)
    discharge_datetime = ASM_output["recommended_discharge_day"]
    write_dtd(discharge_datetime)
    return {
        "status": "discharge_day_saved",
        "full_result": ASM_output
    }

### Comparing today with dtd and opens switch
@app.get("/discharge_switch", tags = ["Day to Discharge"])
async def discharge_switch():
    now = datetime.now()
    today = now.date()

    discharge_dt = read_dtd()
    if discharge_dt:
        is_discharge_day = discharge_dt.date() == today
        return {
            "status": "discharge switch is on" if is_discharge_day else "discharge switch is off",
            "discharge_switch": is_discharge_day,
            "today": now.strftime("%A, %d %B %Y %I:%M %p"),
            "discharge_day": discharge_dt.strftime("%A, %d %B %Y %I:%M %p"),
        }

    return {
        "status": "error: discharge day is not calculated",
        "discharge_switch": False,
        "today": now.strftime("%A, %d %B %Y %I:%M %p"),
        "discharge_day": None
    }


'''Estimate SOC by using voltage of the battery(very basic, will try to improvise)'''
@app.get("/soc", tags = ["SOC Estimation"])
async def estimate_soc(voltage: float = Query(..., description="Enter the voltage of 3S pack")):
    soc = est_soc(voltage)
    return {
        "voltage": voltage,
        "estimated_SOC_percent": round(soc,2)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)