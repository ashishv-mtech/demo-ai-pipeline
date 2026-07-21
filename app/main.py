from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from config import APP_NAME, APP_VERSION, HOST, PORT, DEFAULT_MODEL_NAME
from .schema import ApiResponse, MLModelInput, MLModelOutput
from .service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm default model into memory cache on app startup."""
    print("Starting FastAPI Application...")
    try:
        model_service.load_model(DEFAULT_MODEL_NAME)
        print(f"Default model '{DEFAULT_MODEL_NAME}' successfully loaded into memory.")
    except Exception as e:
        print(f"Warning: Could not pre-load default model: {e}")
    yield
    print("Shutting down FastAPI Application...")


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="FastAPI service for Heart Disease Risk ML predictions using trained pipelines from the models/ directory.",
    lifespan=lifespan
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception Handler for Validation Errors
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content=ApiResponse(
            success=False,
            data=None,
            message=f"Pydantic Validation Error: {exc.errors()}"
        ).model_dump()
    )


# Health Check Endpoint
@app.get("/", response_model=ApiResponse[str], summary="Health Check")
def health_check():
    """Returns the operational status of the ML prediction server."""
    return ApiResponse(success=True, data="Server Ready", message="API Service is operational.")


# Predict Endpoint
@app.post("/predict", response_model=ApiResponse[MLModelOutput], summary="Predict Heart Disease Risk")
def predict(payload: MLModelInput):
    """Predict heart disease risk using the primary/default model pipeline."""
    try:
        result = model_service.predict(input_data=payload)
        return ApiResponse(success=True, data=result, message="Prediction calculated successfully.")
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app="app.main:app", host=HOST, port=PORT, reload=True)

