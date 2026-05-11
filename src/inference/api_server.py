from typing import List, Optional, Dict
import time
import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from src.inference.predict import WearableInferenceEngine
from src.inference.feature_builder import (
    FEATURE_COLS,
    build_feature_dict,
    build_feature_dict_from_activity_id,
    build_feature_dict_from_device_code,
)

print("LOADED api_server.py NEW VERSION")

app = FastAPI(
    title="FL Wearable Inference API",
    description="Inference API for personalized wearable alert prediction",
    version="2.2.0",
)

engine = WearableInferenceEngine(
    model_path="models/flower_global_best.pt",
    scaler_path="models/flower_scaler.pkl",
)


# =========================================================
# Logging helpers
# =========================================================
def log(msg: str):
    print(msg, flush=True)


# =========================================================
# Middleware: log every request/response
# =========================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()

    log(f"\n=== REQUEST START ===")
    log(f"method={request.method} path={request.url.path}")
    log(f"query={dict(request.query_params)}")

    try:
        response = await call_next(request)
        duration_ms = round((time.time() - start) * 1000, 2)

        log(f"=== REQUEST END ===")
        log(f"status_code={response.status_code} duration_ms={duration_ms}")
        return response

    except Exception as e:
        duration_ms = round((time.time() - start) * 1000, 2)
        log("=== REQUEST CRASH ===")
        log(f"duration_ms={duration_ms}")
        log(f"error={repr(e)}")
        log(traceback.format_exc())
        raise


# =========================================================
# Global validation error handler for 422
# =========================================================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    log("\n=== VALIDATION ERROR 422 ===")
    log(f"path={request.url.path}")
    log(f"errors={exc.errors()}")

    try:
        body = await request.body()
        log(f"raw_body={body.decode('utf-8', errors='ignore')}")
    except Exception as e:
        log(f"could_not_read_body={repr(e)}")
    errors = exc.errors()
    for err in errors:
        if "ctx" in err and "error" in err["ctx"]:
            err["ctx"]["error"] = str(err["ctx"]["error"])
    return JSONResponse(
        status_code=422,
        content={"detail": errors},
    )


# =========================================================
# Schemas
# =========================================================
class PredictFeaturesRequest(BaseModel):
    heart_rate: float = Field(gt=0, le=250)
    hr_rolling_mean: float = Field(ge=0, le=250)
    hr_rolling_std: float = Field(ge=0, le=100)
    acc_magnitude: float = Field(ge=0, le=100)

    act_rest: int = Field(ge=0, le=1)
    act_walk: int = Field(ge=0, le=1)
    act_brisk: int = Field(ge=0, le=1)
    act_run: int = Field(ge=0, le=1)

    class Config:
        extra = "forbid"

    @model_validator(mode="after")
    def validate_onehot(self):
        onehot_sum = self.act_rest + self.act_walk + self.act_brisk + self.act_run
        if onehot_sum != 1:
            raise ValueError(
                "Exactly one of act_rest/act_walk/act_brisk/act_run must be 1."
            )
        return self


class PredictFromSensorRequest(BaseModel):
    heart_rate: float = Field(gt=0, le=250)
    hr_window: List[float] = Field(min_length=1, max_length=60)

    acc_x: float = Field(ge=-50, le=50)
    acc_y: float = Field(ge=-50, le=50)
    acc_z: float = Field(ge=-50, le=50)

    activity_group: Optional[str] = None
    activity_id: Optional[int] = None
    device_activity_code: Optional[int] = None

    hour_of_day: Optional[float] = Field(default=None, ge=0.0, lt=24.0)
    timestamp_seconds: Optional[float] = Field(default=None, ge=0.0)
    class Config:
        extra = "forbid"
    @model_validator(mode="after")
    def validate_request(self):
        activity_sources = [
            self.activity_group is not None,
            self.activity_id is not None,
            self.device_activity_code is not None,
        ]
        if sum(activity_sources) != 1:
            raise ValueError(
                "Provide exactly one of: activity_group, activity_id, device_activity_code."
            )

        time_sources = [
            self.hour_of_day is not None,
            self.timestamp_seconds is not None,
        ]
        if sum(time_sources) != 1:
            raise ValueError(
                "Provide exactly one of: hour_of_day or timestamp_seconds."
            )

        if self.activity_group is not None:
            valid_groups = {"rest", "walk", "brisk", "run"}
            if self.activity_group not in valid_groups:
                raise ValueError(
                    f"activity_group must be one of {sorted(valid_groups)}"
                )

        if self.device_activity_code is not None:
            valid_codes = {0, 1, 2, 4}
            if self.device_activity_code not in valid_codes:
                raise ValueError(
                    f"device_activity_code must be one of {sorted(valid_codes)}"
                )

        for x in self.hr_window:
            if x <= 0 or x > 250:
                raise ValueError("All hr_window values must be in (0, 250].")

        return self


class PredictResponse(BaseModel):
    pred_class: int
    pred_label: str
    probabilities: Dict[str, float]
    action: str
    features: Optional[Dict[str, float]] = None


# =========================================================
# Helpers
# =========================================================
def label_to_action(pred_class: int) -> str:
    if pred_class == 0:
        return "OK"
    elif pred_class == 1:
        return "WARNING"
    elif pred_class == 2:
        return "ALERT"
    return "UNKNOWN"


def validate_feature_dict(feature_dict: Dict[str, float]) -> None:
    required = {
        "heart_rate",
        "hr_rolling_mean",
        "hr_rolling_std",
        "acc_magnitude",
        "act_rest",
        "act_walk",
        "act_brisk",
        "act_run",
    }

    missing = required - set(feature_dict.keys())
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    if not (0 < feature_dict["heart_rate"] <= 250):
        raise ValueError("heart_rate must be in (0, 250].")

    if not (0 <= feature_dict["hr_rolling_mean"] <= 250):
        raise ValueError("hr_rolling_mean must be in [0, 250].")

    if not (0 <= feature_dict["hr_rolling_std"] <= 100):
        raise ValueError("hr_rolling_std must be in [0, 100].")

    if not (0 <= feature_dict["acc_magnitude"] <= 100):
        raise ValueError("acc_magnitude must be in [0, 100].")

    onehot_sum = (
        feature_dict["act_rest"]
        + feature_dict["act_walk"]
        + feature_dict["act_brisk"]
        + feature_dict["act_run"]
    )
    if onehot_sum != 1:
        raise ValueError("Exactly one of act_rest/act_walk/act_brisk/act_run must be 1.")


# =========================================================
# Routes
# =========================================================
@app.get("/health")
def health():
    log("GET /health called")
    return {
        "status": "ok",
        "model": "flower_global_best.pt",
        "features": FEATURE_COLS,
    }


@app.get("/labels")
def labels():
    log("GET /labels called")
    return {
        "0": "OK",
        "1": "MEDIUM",
        "2": "HIGH",
    }


@app.get("/device-activity-map")
def device_activity_map():
    log("GET /device-activity-map called")
    return {
        "0": "rest",
        "1": "walk",
        "2": "brisk",
        "4": "run",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictFeaturesRequest):
    try:
        log("\n=== /predict called ===")
        log(f"request={request.model_dump()}")

        feature_dict = request.model_dump()
        validate_feature_dict(feature_dict)

        result = engine.predict(feature_dict, debug=True)
        log(f"predict_result={result}")

        return PredictResponse(
            pred_class=result["pred_class"],
            pred_label=result["pred_label"],
            probabilities=result["probabilities"],
            action=label_to_action(result["pred_class"]),
            features=feature_dict,
        )

    except Exception as e:
        log("=== /predict ERROR ===")
        log(repr(e))
        log(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-from-sensor", response_model=PredictResponse)
def predict_from_sensor(request: PredictFromSensorRequest):
    try:
        log("\n=== /predict-from-sensor called ===")
        log(f"request={request.model_dump()}")

        if request.activity_group is not None:
            feature_dict = build_feature_dict(
                heart_rate=request.heart_rate,
                hr_window=request.hr_window,
                acc_x=request.acc_x,
                acc_y=request.acc_y,
                acc_z=request.acc_z,
                activity_group=request.activity_group,
                hour_of_day=request.hour_of_day,
                timestamp_seconds=request.timestamp_seconds,
            )
        elif request.activity_id is not None:
            feature_dict = build_feature_dict_from_activity_id(
                heart_rate=request.heart_rate,
                hr_window=request.hr_window,
                acc_x=request.acc_x,
                acc_y=request.acc_y,
                acc_z=request.acc_z,
                activity_id=request.activity_id,
                hour_of_day=request.hour_of_day,
                timestamp_seconds=request.timestamp_seconds,
            )
        else:
            feature_dict = build_feature_dict_from_device_code(
                heart_rate=request.heart_rate,
                hr_window=request.hr_window,
                acc_x=request.acc_x,
                acc_y=request.acc_y,
                acc_z=request.acc_z,
                activity_code=request.device_activity_code,
                hour_of_day=request.hour_of_day,
                timestamp_seconds=request.timestamp_seconds,
            )

        log(f"built_feature_dict={feature_dict}")

        validate_feature_dict(feature_dict)

        result = engine.predict(feature_dict, debug=True)
        log(f"predict_result={result}")

        return PredictResponse(
            pred_class=result["pred_class"],
            pred_label=result["pred_label"],
            probabilities=result["probabilities"],
            action=label_to_action(result["pred_class"]),
            features=feature_dict,
        )

    except Exception as e:
        log("=== /predict-from-sensor ERROR ===")
        log(repr(e))
        log(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))