import numpy as np
import re
import pandas as pd
import calendar
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from catboost import CatBoostClassifier
import logging
import time
from contextlib import asynccontextmanager
import pick_regno
import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Загрузка модели CatBoost...")

    try:
        model = CatBoostClassifier()
        model.load_model("micromodel.cbm")
        logger.info("Модель успешно загружена")

    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        raise e
    
    yield
    logger.info("Завершение работы сервиса")

app = FastAPI(
    title="License Plate Recognition ML Service",
    description="Веб-сервис для анализа распознавания номерных знаков на основе модели CatBoost",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Корневой endpoint для проверки работоспособности сервиса"""
    return {
        "message": "License Plate Recognition ML Service",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Endpoint для проверки здоровья сервиса"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.post("/predict", response_model=models.PredictionResponse)
async def predict_single(data: models.LicensePlateData):
    """Обработка одного запроса для предсказания"""
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    start_time = time.time()
    
    try:
        prediction = pick_regno.pick_regno(
            camera_regno=data.regno_recognize,
            nn_regno=data.afts_regno_ai,
            camera_score=data.recognition_accuracy,
            nn_score=data.afts_regno_ai_score,
            nn_sym_scores=data.afts_regno_ai_char_scores,
            nn_len_scores=data.afts_regno_ai_length_scores,
            camera_type=data.camera_type,
            camera_class=data.camera_class,
            time_check=data.time_check,
            direction=data.direction,
            model_name="micromodel.cbm"
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Обработан запрос за {processing_time:.2f}ms")
        
        return models.PredictionResponse(
            prediction=prediction[0].tolist(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.post("/predict/batch", response_model=models.BatchResponse)
async def predict_batch(request: models.BatchRequest):
    """Обработка пакетного запроса для предсказания"""
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    start_time = time.time()
    predictions = []
    
    try:
        for data in request.data:
            prediction = pick_regno.pick_regno(
                camera_regno=data.regno_recognize,
                nn_regno=data.afts_regno_ai,
                camera_score=data.recognition_accuracy,
                nn_score=data.afts_regno_ai_score,
                nn_sym_scores=data.afts_regno_ai_char_scores,
                nn_len_scores=data.afts_regno_ai_length_scores,
                camera_type=data.camera_type,
                camera_class=data.camera_class,
                time_check=data.time_check,
                direction=data.direction,
                model_name="micromodel.cbm"
            )
            predictions.append(prediction[0].tolist())
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Обработан пакет из {len(request.data)} запросов за {processing_time:.2f}ms")
        
        return models.BatchResponse(
            predictions=predictions,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Ошибка при обработке пакетного запроса: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Информация о загруженной модели"""
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    return {
        "model_type": "CatBoostClassifier",
        "feature_count": len(model.feature_names_) if hasattr(model, 'feature_names_') else "Unknown",
        "feature_names": model.feature_names_.tolist() if hasattr(model, 'feature_names_') else [],
        "classes_count": len(model.classes_) if hasattr(model, 'classes_') else "Unknown"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
