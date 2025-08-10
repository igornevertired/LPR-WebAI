import logging
import os
import time
from typing import List, Optional
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import pandas as pd
from ml_scripts.pick_regno import pick_regno
from models.models import *
import numpy as np

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI(
    title="License Plate Recognition API",
    description="ML-powered license plate recognition service for high-load traffic monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Добавление middleware для оптимизации
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Добавление GZip сжатия для экономии трафика
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Глобальная переменная для отслеживания состояния модели
model_status = {"loaded": False, "path": None, "last_check": None}

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске сервиса"""
    logger.info("Starting License Plate Recognition Service...")
    
    # Проверка доступности модели
    model_path = "src/data/micromodel.cbm"
    if os.path.exists(model_path):
        try:
            # Попытка загрузить модель для проверки
            from ml_scripts.pick_regno import get_cached_model
            get_cached_model(model_path)
            model_status["loaded"] = True
            model_status["path"] = model_path
            model_status["last_check"] = datetime.now()
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            model_status["loaded"] = False
    else:
        logger.warning(f"Model file not found at {model_path}")
        model_status["loaded"] = False

@app.get("/", response_model=dict)
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "License Plate Recognition API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка состояния сервиса"""
    return HealthResponse(
        status="healthy" if model_status["loaded"] else "degraded",
        timestamp=datetime.now(),
        model_loaded=model_status["loaded"],
        version="1.0.0"
    )

@app.post("/predict", response_model=LicensePlateResponse)
async def predict_license_plate(request: LicensePlateRequest):
    """Обработка одиночного запроса для предсказания"""
    if not model_status["loaded"]:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    start_time = time.time()
    
    try:
        if not request.camera_regno or not request.nn_regno:
            raise HTTPException(status_code=400, detail="Номера не могут быть пустыми")
        
        if request.camera_score < 0 or request.camera_score > 100:
            raise HTTPException(status_code=400, detail="Точность камеры должна быть от 0 до 100")
        
        if request.nn_score < 0 or request.nn_score > 1:
            raise HTTPException(status_code=400, detail="Оценка нейросети должна быть от 0 до 1")
        
        if request.direction not in [0, 1]:
            raise HTTPException(status_code=400, detail="Направление должно быть 0 или 1")
        
        prediction = pick_regno(
            camera_regno=request.camera_regno,
            nn_regno=request.nn_regno,
            camera_score=request.camera_score,
            nn_score=request.nn_score,
            nn_sym_scores=request.nn_sym_scores,
            nn_len_scores=request.nn_len_scores,
            camera_type=request.camera_type,
            camera_class=request.camera_class,
            time_check=request.time_check,
            direction=request.direction,
            model_name=request.model_name
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        prediction_list = prediction.tolist()
                confidence = float(np.max(prediction))
        
        logger.info(f"Обработан запрос за {processing_time:.2f}ms, уверенность: {confidence:.4f}")
        
        return LicensePlateResponse(
            prediction=prediction_list,
            confidence=confidence,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.post("/batch-predict", response_model=BatchResponse)
async def batch_predict_license_plates(requests: List[LicensePlateRequest]):
    """Обработка пакетного запроса для предсказания"""
    if not model_status["loaded"]:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    if not requests:
        raise HTTPException(status_code=400, detail="Список запросов не может быть пустым")
    
    if len(requests) > 1000:
        raise HTTPException(status_code=400, detail="Максимальный размер пакета: 1000 запросов")
    
    start_time = time.time()
    results = []
    successful_count = 0
    failed_count = 0
    
    for i, request in enumerate(requests):
        item_start_time = time.time()
        
        try:
            # Валидация входных данных
            if not request.camera_regno or not request.nn_regno:
                raise ValueError("Номера не могут быть пустыми")
            
            if request.camera_score < 0 or request.camera_score > 100:
                raise ValueError("Точность камеры должна быть от 0 до 100")
            
            if request.nn_score < 0 or request.nn_score > 1:
                raise ValueError("Оценка нейросети должна быть от 0 до 1")
            
            if request.direction not in [0, 1]:
                raise ValueError("Направление должно быть 0 или 1")
            
            # Вызов ML модели
            prediction = pick_regno(
                camera_regno=request.camera_regno,
                nn_regno=request.nn_regno,
                camera_score=request.camera_score,
                nn_score=request.nn_score,
                nn_sym_scores=request.nn_sym_scores,
                nn_len_scores=request.nn_len_scores,
                camera_type=request.camera_type,
                camera_class=request.camera_class,
                time_check=request.time_check,
                direction=request.direction,
                model_name=request.model_name
            )
            
            item_processing_time = (time.time() - item_start_time) * 1000
            
            prediction_list = prediction.tolist()
            confidence = float(np.max(prediction))
            
            results.append(BatchPredictionResult(
                index=i,
                status="success",
                prediction=prediction_list,
                confidence=confidence,
                processing_time_ms=item_processing_time
            ))
            
            successful_count += 1
            
        except Exception as e:
            item_processing_time = (time.time() - item_start_time) * 1000
            results.append(BatchPredictionResult(
                index=i,
                status="error",
                error=str(e),
                processing_time_ms=item_processing_time
            ))
            
            failed_count += 1
    
    total_processing_time = (time.time() - start_time) * 1000
    
    logger.info(f"Обработан пакет из {len(requests)} запросов за {total_processing_time:.2f}ms")
    logger.info(f"Успешно: {successful_count}, с ошибками: {failed_count}")
    
    return BatchResponse(
        results=results,
        total_processing_time_ms=total_processing_time,
        successful_items=successful_count,
        failed_items=failed_count
    )

@app.get("/model/info")
async def model_info():
    """Информация о загруженной модели"""
    if not model_status["loaded"]:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    return {
        "model_type": "CatBoostClassifier",
        "model_path": model_status["path"],
        "status": "loaded",
        "last_check": model_status["last_check"].isoformat() if model_status["last_check"] else None,
        "version": "1.0.0"
    }

@app.post("/test-data")
async def test_with_sample_data():
    """Тестирование сервиса с тестовыми данными"""
    if not model_status["loaded"]:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    try:
        # Чтение тестовых данных
        test_data_path = "src/data/test_data.csv"
        if not os.path.exists(test_data_path):
            raise HTTPException(status_code=404, detail="Тестовые данные не найдены")
        
        df = pd.read_csv(test_data_path)
        if df.empty:
            raise HTTPException(status_code=400, detail="Тестовые данные пусты")
        
        # Берем первый пример для тестирования
        sample = df.iloc[0]
        
        # Создаем тестовый запрос
        test_request = LicensePlateRequest(
            camera_regno=sample['regno_recognize'],
            nn_regno=sample['afts_regno_ai'],
            camera_score=sample['recognition_accuracy'],
            nn_score=sample['afts_regno_ai_score'],
            nn_sym_scores=sample['afts_regno_ai_char_scores'],
            nn_len_scores=sample['afts_regno_ai_length_scores'],
            camera_type=sample['camera_type'],
            camera_class=sample['camera_class'],
            time_check=sample['time_check'],
            direction=sample['direction']
        )
        
        result = await predict_license_plate(test_request)
        
        return {
            "message": "Тест успешно выполнен",
            "test_data": sample.to_dict(),
            "prediction_result": result.model_dump()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при тестировании: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка тестирования: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,
        log_level="info",
        access_log=True
    )
