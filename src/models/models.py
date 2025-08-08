from pydantic import BaseModel, Field
from typing import List, Optional

class LicensePlateData(BaseModel):
    regno_recognize: str = Field(..., description="Распознанный номер с камеры")
    afts_regno_ai: str = Field(..., description="Номер, распознанный ИИ")
    recognition_accuracy: float = Field(..., description="Точность распознавания камеры")
    afts_regno_ai_score: float = Field(..., description="Оценка ИИ для номера")
    afts_regno_ai_char_scores: str = Field(..., description="Оценки символов ИИ в формате JSON строки")
    afts_regno_ai_length_scores: str = Field(..., description="Оценки длины ИИ в формате JSON строки")
    camera_type: str = Field(..., description="Тип камеры")
    camera_class: str = Field(..., description="Класс камеры")
    time_check: str = Field(..., description="Время проверки в формате YYYY-MM-DD HH:MM:SS")
    direction: int = Field(..., description="Направление движения (0 или 1)")

class PredictionResponse(BaseModel):
    prediction: List[float] = Field(..., description="Вероятности классов")
    processing_time_ms: float = Field(..., description="Время обработки в миллисекундах")

class BatchRequest(BaseModel):
    data: List[LicensePlateData] = Field(..., description="Массив данных для обработки")

class BatchResponse(BaseModel):
    predictions: List[List[float]] = Field(..., description="Массив предсказаний")
    processing_time_ms: float = Field(..., description="Общее время обработки в миллисекундах")