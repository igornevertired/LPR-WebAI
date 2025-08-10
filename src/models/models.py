from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class LicensePlateRequest(BaseModel):
    """Модель запроса для распознавания номерного знака"""
    camera_regno: str = Field(..., description="Распознанный номер с камеры (regno_recognize)")
    nn_regno: str = Field(..., description="Номер, распознанный ИИ (afts_regno_ai)")
    camera_score: float = Field(..., description="Точность распознавания камеры (0-100)")
    nn_score: float = Field(..., description="Оценка ИИ для номера (0-1)")
    nn_sym_scores: str = Field(..., description="Оценки символов ИИ в формате JSON строки")
    nn_len_scores: str = Field(..., description="Оценки длины ИИ в формате JSON строки")
    camera_type: str = Field(..., description="Тип камеры")
    camera_class: str = Field(..., description="Класс камеры")
    time_check: str = Field(..., description="Время проверки в формате YYYY-MM-DD HH:MM:SS")
    direction: int = Field(..., description="Направление движения (0 или 1)")
    model_name: str = Field(default="src/data/micromodel.cbm", description="Путь к модели")

class LicensePlateResponse(BaseModel):
    """Модель ответа для распознавания номерного знака"""
    prediction: List[float] = Field(..., description="Вероятности классов")
    confidence: float = Field(..., description="Уверенность в предсказании")
    processing_time_ms: float = Field(..., description="Время обработки в миллисекундах")

class BatchPredictionResult(BaseModel):
    """Результат обработки одного элемента в пакетном запросе"""
    index: int = Field(..., description="Индекс элемента в пакете")
    status: str = Field(..., description="Статус обработки: success/error")
    prediction: Optional[List[float]] = Field(None, description="Вероятности классов")
    confidence: Optional[float] = Field(None, description="Уверенность в предсказании")
    processing_time_ms: float = Field(..., description="Время обработки в миллисекундах")
    error: Optional[str] = Field(None, description="Описание ошибки")

class BatchResponse(BaseModel):
    """Модель ответа для пакетного запроса"""
    results: List[BatchPredictionResult] = Field(..., description="Результаты обработки")
    total_processing_time_ms: float = Field(..., description="Общее время обработки в миллисекундах")
    successful_items: int = Field(..., description="Количество успешно обработанных элементов")
    failed_items: int = Field(..., description="Количество элементов с ошибками")

class HealthResponse(BaseModel):
    """Модель ответа для проверки состояния сервиса"""
    status: str = Field(..., description="Статус сервиса")
    timestamp: datetime = Field(..., description="Время проверки")
    model_loaded: bool = Field(..., description="Загружена ли модель")
    version: str = Field(..., description="Версия сервиса")