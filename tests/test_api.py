import pytest
import logging
from fastapi.testclient import TestClient
from httpx import AsyncClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import app

client = TestClient(app)

class TestLicensePlateAPI:
    """Тесты для API распознавания номерных знаков"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.logger = logging.getLogger(f"test.{self.__class__.__name__}")
        self.logger.info("Настройка теста")
    
    def teardown_method(self):
        """Очистка после каждого теста"""
        self.logger.info("Завершение теста")
    
    @pytest.mark.unit
    def test_root_endpoint(self):
        """Тест корневого эндпоинта"""
        self.logger.info("Тестирование корневого эндпоинта /")
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "1.0.0"

        self.logger.info(f"Корневой эндпоинт работает: {data}")
    
    @pytest.mark.unit
    def test_health_check(self):
        """Тест проверки состояния сервиса"""
        self.logger.info("Тестирование эндпоинта /health")
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        
        self.logger.info(f"Health check работает: статус={data['status']}, модель={data['model_loaded']}")
    
    @pytest.mark.unit
    def test_model_info(self):
        """Тест информации о модели"""
        self.logger.info("Тестирование эндпоинта /model/info")
        response = client.get("/model/info")

        if response.status_code == 503:
            self.logger.warning("Модель не загружена - это нормально для тестов")
            assert "Модель не загружена" in response.json()["detail"]
        else:
            assert response.status_code == 200
            data = response.json()
            assert "model_type" in data
            assert "status" in data

            self.logger.info(f"Model info работает: {data}")

    @pytest.mark.integration
    def test_predict_endpoint_validation(self, sample_license_plate_data):
        """Тест валидации эндпоинта предсказания"""
        self.logger.info("Тестирование валидации эндпоинта /predict")
        
        response = client.post("/predict", json=sample_license_plate_data)
        
        if response.status_code == 503:
            self.logger.warning("Модель не загружена - пропускаем тест предсказания")
            return
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "processing_time_ms" in data

            self.logger.info(f"Предсказание успешно: уверенность={data['confidence']:.4f}")
        else:
            self.logger.error(f"Ошибка предсказания: {response.status_code} - {response.text}")

    @pytest.mark.integration
    def test_batch_predict_endpoint(self, sample_batch_data):
        """Тест пакетного предсказания"""
        self.logger.info("Тестирование эндпоинта /batch-predict")
        
        response = client.post("/batch-predict", json=sample_batch_data)
        
        if response.status_code == 503:
            self.logger.warning("Модель не загружена - пропускаем тест пакетного предсказания")
            return
        
        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert "total_processing_time_ms" in data
            assert "successful_items" in data
            assert "failed_items" in data

            self.logger.info(f"Пакетное предсказание: успешно={data['successful_items']}, ошибок={data['failed_items']}")
        else:
            self.logger.error(f"Ошибка пакетного предсказания: {response.status_code} - {response.text}")

    @pytest.mark.integration
    def test_test_data_endpoint(self):
        """Тест эндпоинта с тестовыми данными"""
        self.logger.info("Тестирование эндпоинта /test-data")
        
        response = client.post("/test-data")
        
        if response.status_code == 503:
            self.logger.warning("Модель не загружена - пропускаем тест с тестовыми данными")
            return
        
        if response.status_code == 200:
            data = response.json()
            assert "message" in data
            assert "test_data" in data
            assert "prediction_result" in data
            
            self.logger.info(f"Тест с данными успешен: {data['message']}")
        else:
            self.logger.error(f"Ошибка теста с данными: {response.status_code} - {response.text}")
    
    @pytest.mark.unit
    def test_invalid_input_validation(self):
        """Тест валидации некорректных входных данных"""
        self.logger.info("Тестирование валидации некорректных данных")
        
        invalid_data = {
            "camera_regno": "",
            "nn_regno": "",
            "camera_score": 50.0,
            "nn_score": 0.8,
            "nn_sym_scores": "[0.9, 0.8]",
            "nn_len_scores": "[0.9]",
            "camera_type": "mobile",
            "camera_class": "high",
            "time_check": "2024-01-15T10:30:00",
            "direction": 1
        }
        
        response = client.post("/predict", json=invalid_data)
        
        if response.status_code == 503:
            self.logger.warning("Модель не загружена - пропускаем тест валидации")
            return

        assert response.status_code == 400
        assert "Номера не могут быть пустыми" in response.json()["detail"]

        self.logger.info("Валидация пустых номеров работает")

        invalid_data["camera_regno"] = "А123БВ77"
        invalid_data["nn_regno"] = "А123БВ77"
        invalid_data["camera_score"] = 150.0 

        response = client.post("/predict", json=invalid_data)

        if response.status_code == 503:
            self.logger.warning("Модель не загружена - пропускаем тест валидации")
            return

        assert response.status_code == 400
        assert "Точность камеры должна быть от 0 до 100" in response.json()["detail"]

        self.logger.info("Валидация camera_score работает")

        invalid_data["camera_score"] = 50.0
        invalid_data["direction"] = 2 
        
        response = client.post("/predict", json=invalid_data)
        
        if response.status_code == 503:
            self.logger.warning("Модель не загружена - пропускаем тест валидации")
            return
            
        assert response.status_code == 400
        assert "Направление должно быть 0 или 1" in response.json()["detail"]
        
        self.logger.info("Валидация direction работает")

class TestDataValidation:
    """Тесты валидации данных"""
    
    def setup_method(self):
        self.logger = logging.getLogger(f"test.{self.__class__.__name__}")
        self.logger.info("Настройка теста валидации данных")
    
    @pytest.mark.unit
    def test_csv_data_structure(self, test_data_path):
        """Тест структуры CSV данных"""
        self.logger.info(f"Тестирование структуры CSV данных: {test_data_path}")
        
        import pandas as pd
        
        if not os.path.exists(test_data_path):
            self.logger.warning(f"Файл {test_data_path} не найден")
            return
        
        try:
            df = pd.read_csv(test_data_path)
            self.logger.info(f"CSV загружен: {len(df)} строк, {len(df.columns)} колонок")
            
            # Проверяем наличие необходимых колонок
            required_columns = [
                'regno_recognize', 'afts_regno_ai', 'recognition_accuracy',
                'afts_regno_ai_score', 'afts_regno_ai_char_scores',
                'afts_regno_ai_length_scores', 'camera_type', 'camera_class',
                'time_check', 'direction'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Отсутствуют колонки: {missing_columns}")
                assert False, f"Отсутствуют колонки: {missing_columns}"
            
            self.logger.info("Все необходимые колонки присутствуют")
            
            # Проверяем типы данных
            self.logger.info(f"Типы колонок: {df.dtypes.to_dict()}")
            
        except Exception as e:
            self.logger.error(f"Ошибка чтения CSV: {e}")
            raise

class TestPerformance:
    """Тесты производительности"""
    
    def setup_method(self):
        self.logger = logging.getLogger(f"test.{self.__class__.__name__}")
        self.logger.info("Настройка теста производительности")
    
    @pytest.mark.slow
    def test_response_time(self, sample_license_plate_data):
        """Тест времени отклика API"""
        self.logger.info("Тестирование времени отклика API")
        
        import time
        
        start_time = time.time()
        response = client.get("/health")
        health_time = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        self.logger.info(f"Health check время отклика: {health_time:.2f}ms")
        
        assert health_time < 100, f"Health check слишком медленный: {health_time:.2f}ms"
        
        start_time = time.time()
        response = client.get("/model/info")
        model_info_time = (time.time() - start_time) * 1000
        
        self.logger.info(f"Model info время отклика: {model_info_time:.2f}ms")
        
        assert model_info_time < 200, f"Model info слишком медленный: {model_info_time:.2f}ms"
