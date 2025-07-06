import sqlite3
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DatabaseManager")

class ChatbotDatabase:
    """Base de datos exclusiva para la RNN de generación de respuestas"""
    
    def __init__(self, db_path: str = "chatbot_rnn.db"):
        self.db_path = db_path
        self.init_database()
        logger.info(f"Base de datos del Chatbot inicializada: {db_path}")
    
    def init_database(self):
        """Inicializa las tablas específicas para el chatbot"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tabla de conversaciones
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    bot_response TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    confidence_score REAL,
                    response_time_ms INTEGER
                )
            """)
            
            # Tabla de patrones de entrenamiento
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_pattern TEXT NOT NULL,
                    target_pattern TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    last_used DATETIME DEFAULT CURRENT_TIMESTAMP,
                    effectiveness_score REAL DEFAULT 0.5
                )
            """)
            
            # Tabla de métricas del modelo
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL,
                    accuracy REAL,
                    loss REAL,
                    training_time_minutes INTEGER,
                    epoch_count INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def save_conversation(self, user_id: str, session_id: str, user_input: str, 
                         bot_response: str, confidence_score: float = None, 
                         response_time_ms: int = None):
        """Guarda una conversación en la base de datos"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations 
                (user_id, session_id, user_input, bot_response, confidence_score, response_time_ms)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, session_id, user_input, bot_response, confidence_score, response_time_ms))
            conn.commit()
    
    def get_conversation_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Obtiene el historial de conversaciones de un usuario"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_input, bot_response, timestamp, confidence_score
                FROM conversations 
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, limit))
            
            return [{
                'user_input': row[0],
                'bot_response': row[1],
                'timestamp': row[2],
                'confidence_score': row[3]
            } for row in cursor.fetchall()]
    
    def save_training_metrics(self, model_version: str, accuracy: float, 
                            loss: float, training_time_minutes: int, epoch_count: int):
        """Guarda métricas de entrenamiento del modelo"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_metrics 
                (model_version, accuracy, loss, training_time_minutes, epoch_count)
                VALUES (?, ?, ?, ?, ?)
            """, (model_version, accuracy, loss, training_time_minutes, epoch_count))
            conn.commit()

class RealTimeRNNDatabase:
    """Base de datos exclusiva para la RNN de tiempo real con atención"""
    
    def __init__(self, db_path: str = "realtime_rnn.db"):
        self.db_path = db_path
        self.init_database()
        logger.info(f"Base de datos de RNN Tiempo Real inicializada: {db_path}")
    
    def init_database(self):
        """Inicializa las tablas específicas para análisis de tiempo real"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tabla de secuencias temporales
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS time_sequences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sequence_id TEXT NOT NULL,
                    sequence_data TEXT NOT NULL,  -- JSON array
                    predicted_value REAL,
                    actual_value REAL,
                    prediction_error REAL,
                    attention_weights TEXT,  -- JSON array
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla de métricas de rendimiento
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL,
                    mae REAL,
                    mse REAL,
                    test_accuracy REAL,
                    processing_time_ms INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla de patrones detectados
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detected_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_description TEXT,
                    confidence_score REAL,
                    frequency INTEGER DEFAULT 1,
                    first_detected DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_detected DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def save_prediction(self, sequence_id: str, sequence_data: List[float], 
                       predicted_value: float, actual_value: float = None,
                       attention_weights: List[float] = None):
        """Guarda una predicción de secuencia temporal"""
        prediction_error = None
        if actual_value is not None:
            prediction_error = abs(predicted_value - actual_value)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO time_sequences 
                (sequence_id, sequence_data, predicted_value, actual_value, 
                 prediction_error, attention_weights)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                sequence_id,
                json.dumps(sequence_data),
                predicted_value,
                actual_value,
                prediction_error,
                json.dumps(attention_weights) if attention_weights else None
            ))
            conn.commit()
    
    def get_prediction_accuracy(self, days: int = 7) -> Dict[str, float]:
        """Calcula métricas de precisión de los últimos días"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT AVG(prediction_error), COUNT(*)
                FROM time_sequences 
                WHERE actual_value IS NOT NULL 
                AND timestamp >= datetime('now', '-{} days')
            """.format(days))
            
            result = cursor.fetchone()
            avg_error = result[0] if result[0] else 0.0
            count = result[1]
            
            return {
                'average_error': avg_error,
                'predictions_count': count,
                'accuracy_percentage': max(0, 100 - (avg_error * 100))
            }

class EmotionRNNDatabase:
    """Base de datos exclusiva para la RNN de gestión y verificación emocional"""
    
    def __init__(self, db_path: str = "emotion_rnn.db"):
        self.db_path = db_path
        self.init_database()
        logger.info(f"Base de datos de RNN Emocional inicializada: {db_path}")
    
    def init_database(self):
        """Inicializa las tablas específicas para análisis emocional"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tabla de análisis emocionales
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS emotion_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text_input TEXT NOT NULL,
                    detected_emotion TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla de memoria cuántica
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quantum_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    emotion TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    embedding_vector TEXT,  -- JSON array
                    retrieval_count INTEGER DEFAULT 0,
                    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla de patrones emocionales
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS emotion_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    emotion_sequence TEXT NOT NULL,  -- JSON array
                    pattern_strength REAL NOT NULL,
                    occurrences INTEGER DEFAULT 1,
                    first_detected DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_detected DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla de métricas del modelo emocional
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS emotion_model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL,
                    accuracy REAL,
                    precision_per_emotion TEXT,  -- JSON object
                    recall_per_emotion TEXT,     -- JSON object
                    training_samples INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def save_emotion_analysis(self, text_input: str, detected_emotion: str, 
                            confidence_score: float, user_id: str = None, 
                            session_id: str = None):
        """Guarda un análisis emocional"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO emotion_analysis 
                (text_input, detected_emotion, confidence_score, user_id, session_id)
                VALUES (?, ?, ?, ?, ?)
            """, (text_input, detected_emotion, confidence_score, user_id, session_id))
            conn.commit()
    
    def save_quantum_memory(self, memory_id: str, content: str, emotion: str, 
                          importance_score: float, embedding_vector: List[float]):
        """Guarda una memoria cuántica"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO quantum_memories 
                (memory_id, content, emotion, importance_score, embedding_vector)
                VALUES (?, ?, ?, ?, ?)
            """, (
                memory_id,
                content,
                emotion,
                importance_score,
                json.dumps(embedding_vector)
            ))
            conn.commit()
    
    def get_emotion_statistics(self, user_id: str = None, days: int = 30) -> Dict[str, Any]:
        """Obtiene estadísticas emocionales"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            where_clause = "WHERE timestamp >= datetime('now', '-{} days')".format(days)
            if user_id:
                where_clause += f" AND user_id = '{user_id}'"
            
            cursor.execute(f"""
                SELECT detected_emotion, COUNT(*), AVG(confidence_score)
                FROM emotion_analysis 
                {where_clause}
                GROUP BY detected_emotion
                ORDER BY COUNT(*) DESC
            """)
            
            emotions = {}
            total_analyses = 0
            
            for row in cursor.fetchall():
                emotion, count, avg_confidence = row
                emotions[emotion] = {
                    'count': count,
                    'average_confidence': round(avg_confidence, 3)
                }
                total_analyses += count
            
            # Calcular porcentajes
            for emotion in emotions:
                emotions[emotion]['percentage'] = round(
                    (emotions[emotion]['count'] / total_analyses) * 100, 2
                ) if total_analyses > 0 else 0
            
            return {
                'emotions': emotions,
                'total_analyses': total_analyses,
                'period_days': days
            }

class DatabaseManager:
    """Gestor principal que coordina todas las bases de datos de RNN"""
    
    def __init__(self):
        self.chatbot_db = ChatbotDatabase()
        self.realtime_db = RealTimeRNNDatabase()
        self.emotion_db = EmotionRNNDatabase()
        logger.info("Gestor de bases de datos RNN inicializado")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene el estado general del sistema"""
        return {
            'chatbot': {
                'database': self.chatbot_db.db_path,
                'status': 'active'
            },
            'realtime_rnn': {
                'database': self.realtime_db.db_path,
                'status': 'active'
            },
            'emotion_rnn': {
                'database': self.emotion_db.db_path,
                'status': 'active'
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def backup_all_databases(self, backup_dir: str = "backups"):
        """Crea respaldos de todas las bases de datos"""
        import shutil
        import os
        
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        databases = [
            (self.chatbot_db.db_path, f"chatbot_rnn_{timestamp}.db"),
            (self.realtime_db.db_path, f"realtime_rnn_{timestamp}.db"),
            (self.emotion_db.db_path, f"emotion_rnn_{timestamp}.db")
        ]
        
        for source, backup_name in databases:
            if os.path.exists(source):
                backup_path = os.path.join(backup_dir, backup_name)
                shutil.copy2(source, backup_path)
                logger.info(f"Respaldo creado: {backup_path}")
        
        logger.info(f"Respaldo completo del sistema completado en {backup_dir}")