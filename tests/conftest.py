import pytest
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture(scope="session")
def setup_test_logging():
    """Настройка логирования для тестов"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/test_results.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("test")

@pytest.fixture
def test_data_path():
    """Путь к тестовым данным"""
    return "src/data/test_data.csv"

@pytest.fixture
def model_path():
    """Путь к модели"""
    return "src/data/micromodel.cbm"

@pytest.fixture
def sample_license_plate_data():
    """Образец данных для тестирования одиночного предсказания"""
    return {
        "camera_regno": "А123БВ77",
        "nn_regno": "А123БВ77",
        "camera_score": 85.5,
        "nn_score": 0.92,
        "nn_sym_scores": "[0.95, 0.88, 0.91, 0.94, 0.89, 0.93, 0.90, 0.87]",
        "nn_len_scores": "[0.89, 0.91]",
        "camera_type": "mobile",
        "camera_class": "high",
        "time_check": "2024-01-15T10:30:00",
        "direction": 1
    }

@pytest.fixture
def sample_batch_data():
    """Образец данных для тестирования пакетного предсказания"""
    return {
        "data": [
            {
                "camera_regno": "А123БВ77",
                "nn_regno": "А123БВ77",
                "camera_score": 85.5,
                "nn_score": 0.92,
                "nn_sym_scores": "[0.95, 0.88, 0.91, 0.94, 0.89, 0.93, 0.90, 0.87]",
                "nn_len_scores": "[0.89, 0.91]",
                "camera_type": "mobile",
                "camera_class": "high",
                "time_check": "2024-01-15T10:30:00",
                "direction": 1
            },
            {
                "camera_regno": "В456ГД78",
                "nn_regno": "В456ГД78",
                "camera_score": 78.2,
                "nn_score": 0.87,
                "nn_sym_scores": "[0.89, 0.85, 0.88, 0.86, 0.87, 0.84, 0.89, 0.86]",
                "nn_len_scores": "[0.86, 0.88]",
                "camera_type": "stationary",
                "camera_class": "medium",
                "time_check": "2024-01-15T11:15:00",
                "direction": 0
            }
        ]
    }

