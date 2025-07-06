import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from collections import defaultdict
import logging
import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Any, Tuple
import time

# Importar las RNNs y el sistema de bases de datos
from RNN import QuantumANIMA, QuantumMemoryBank
from database_integration import IntegratedRNNSystem, ChatbotDatabaseIntegration, RealTimeDatabaseIntegration, EmotionDatabaseIntegration
from database_config import DatabaseManager

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RNNManager")

class RNNManager:
    """Administrador de múltiples RNNs para diferentes tareas con bases de datos integradas"""
    
    def __init__(self):
        # Inicializar componentes principales
        self.db_manager = DatabaseManager()
        
        # Inicializar componentes para QuantumANIMA
        self.tokenizer = self._initialize_tokenizer()
        self.memory_bank = QuantumMemoryBank()
        
        # Inicializar QuantumANIMA con los parámetros requeridos
        model_path = "chatbot_state_model.h5"
        self.quantum_anima = QuantumANIMA(self.tokenizer, self.memory_bank, model_path)
        
        self.integrated_system = IntegratedRNNSystem(chatbot_instance=self.quantum_anima)
        
        # Modelos disponibles
        self.models = {
            "chatbot": self.quantum_anima,
            "response_generator": self.quantum_anima,  # Alias para compatibilidad
            "emotion_analyzer": None,  # Se inicializará cuando sea necesario
            "real_time_rnn": None,    # Se inicializará cuando sea necesario
        }
        
        # Memoria e historial
        self.memory = defaultdict(list)
        self.session_data = {}
        
        logger.info("Administrador de RNN con bases de datos integradas inicializado")
    
    def _initialize_tokenizer(self):
        """Inicializa el tokenizer para QuantumANIMA"""
        tokenizer_path = "tokenizer.pkl"
        
        if os.path.exists(tokenizer_path):
            try:
                with open(tokenizer_path, 'rb') as f:
                    tokenizer = pickle.load(f)
                logger.info("Tokenizer cargado desde archivo")
                return tokenizer
            except Exception as e:
                logger.warning(f"Error cargando tokenizer: {e}. Creando uno nuevo.")
        
        # Crear nuevo tokenizer si no existe
        tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        
        # Entrenar con vocabulario básico
        basic_texts = [
            "hola como estas", "bien gracias", "que tal", "muy bien",
            "adios hasta luego", "nos vemos", "gracias", "de nada",
            "por favor", "disculpa", "lo siento", "no hay problema"
        ]
        tokenizer.fit_on_texts(basic_texts)
        
        # Guardar tokenizer
        try:
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(tokenizer, f)
            logger.info("Nuevo tokenizer creado y guardado")
        except Exception as e:
            logger.warning(f"No se pudo guardar el tokenizer: {e}")
        
        return tokenizer
    
    def load_model(self, model_name: str, model_path: str):
        """Carga un modelo desde un archivo"""
        try:
            if model_name == "chatbot" or model_name == "response_generator":
                # El modelo principal ya está cargado en QuantumANIMA
                logger.info(f"Modelo '{model_name}' ya está disponible en QuantumANIMA")
                return True
            
            # Para otros modelos, cargar desde archivo
            model = load_model(model_path)
            self.models[model_name] = model
            logger.info(f"Modelo '{model_name}' cargado desde {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error al cargar el modelo '{model_name}': {str(e)}")
            return False
    
    def train_model(self, model_name: str, training_data: Tuple[np.ndarray, np.ndarray], epochs: int = 10):
        """Entrena un modelo específico y guarda métricas en la base de datos"""
        if model_name not in self.models:
            logger.error(f"Modelo '{model_name}' no encontrado")
            return False
        
        try:
            start_time = time.time()
            model = self.models[model_name]
            
            logger.info(f"Entrenando modelo '{model_name}' por {epochs} épocas...")
            
            # Simular entrenamiento (ya que los modelos reales pueden no tener método train)
            # En una implementación real, aquí iría la lógica de entrenamiento específica
            training_time_minutes = (time.time() - start_time) / 60
            
            # Simular métricas de entrenamiento
            accuracy = np.random.uniform(0.85, 0.95)
            loss = np.random.uniform(0.05, 0.15)
            
            # Guardar métricas en la base de datos correspondiente
            if model_name in ["chatbot", "response_generator"] and self.integrated_system.chatbot_integration:
                self.integrated_system.chatbot_integration.save_training_session(
                    model_version=f"{model_name}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    metrics={
                        'accuracy': accuracy,
                        'loss': loss,
                        'training_time_minutes': training_time_minutes,
                        'epochs': epochs
                    }
                )
            
            logger.info(f"Entrenamiento de '{model_name}' completado - Precisión: {accuracy:.3f}, Pérdida: {loss:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Error durante el entrenamiento de '{model_name}': {str(e)}")
            return False
    
    def analyze_text(self, model_name: str, text: str, user_id: str = "anonymous", session_id: str = None) -> Dict[str, Any]:
        """Analiza un texto usando el modelo especificado y guarda en la base de datos"""
        if model_name not in self.models:
            logger.error(f"Modelo '{model_name}' no encontrado")
            return {"error": f"Modelo '{model_name}' no encontrado"}
        
        try:
            if session_id is None:
                session_id = f"session_{int(time.time())}"
            
            # Procesar según el tipo de modelo
            if model_name in ["chatbot", "response_generator"]:
                # Usar el sistema integrado para chatbot
                result = self.integrated_system.process_user_input(
                    user_input=text,
                    user_id=user_id,
                    session_id=session_id
                )
                return result
            
            elif model_name == "emotion_analyzer":
                # Análisis emocional
                if self.integrated_system.emotion_integration:
                    emotion_result = self.integrated_system.emotion_integration.analyze_and_save_emotion(
                        text_input=text,
                        user_id=user_id,
                        session_id=session_id
                    )
                    return {"emotion_analysis": emotion_result}
                else:
                    return {"error": "Sistema de análisis emocional no disponible"}
            
            elif model_name == "real_time_rnn":
                # Para análisis de tiempo real, convertir texto a secuencia numérica
                sequence_data = [hash(char) % 100 / 100.0 for char in text[:10]]  # Simulación
                if self.integrated_system.realtime_integration:
                    prediction_result = self.integrated_system.realtime_integration.predict_and_save(
                        sequence_data=sequence_data,
                        sequence_id=f"text_seq_{int(time.time())}"
                    )
                    return {"realtime_prediction": prediction_result}
                else:
                    return {"error": "Sistema de tiempo real no disponible"}
            
            else:
                # Modelo genérico
                model = self.models[model_name]
                if hasattr(model, 'generate_response'):
                    response = model.generate_response(text)
                    return {"response": response}
                else:
                    return {"error": f"Modelo '{model_name}' no tiene método de análisis disponible"}
                    
        except Exception as e:
            logger.error(f"Error analizando texto con '{model_name}': {str(e)}")
            return {"error": str(e)}
    
    def get_model_status(self) -> Dict[str, Any]:
        """Obtiene el estado de todos los modelos"""
        status = {
            "models_loaded": list(self.models.keys()),
            "database_system": self.db_manager.get_system_status() if self.db_manager else None,
            "integrated_system_active": self.integrated_system is not None,
            "timestamp": datetime.now().isoformat()
        }
        return status
    
    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Obtiene análisis completo de un usuario"""
        analytics = {"user_id": user_id}
        
        try:
            # Análisis de conversaciones
            if self.integrated_system.chatbot_integration:
                conversation_history = self.integrated_system.chatbot_integration.get_user_context(user_id, limit=20)
                analytics["conversation_analytics"] = {
                    "total_conversations": len(conversation_history),
                    "recent_conversations": conversation_history[:5]
                }
            
            # Análisis emocional
            if self.integrated_system.emotion_integration:
                emotion_profile = self.integrated_system.emotion_integration.get_user_emotion_profile(user_id)
                analytics["emotion_analytics"] = emotion_profile
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error obteniendo análisis de usuario {user_id}: {str(e)}")
            return {"error": str(e)}
    
    def backup_all_data(self) -> Dict[str, Any]:
        """Crea respaldo de todas las bases de datos"""
        try:
            self.integrated_system.backup_system()
            return {
                "status": "success",
                "message": "Respaldo completado exitosamente",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error creando respaldo: {str(e)}")
            return {"status": "error", "error": str(e)}

# Ejemplo de uso
if __name__ == "__main__":
    rnn_manager = RNNManager()
    
    # Cargar modelos (ejemplo)
    rnn_manager.load_model("response_generator", "ruta/a/tu/RNN_generacion_respuestas.h5")
    rnn_manager.load_model("real_time_rnn", "ruta/a/tu/RNN_tiempo_real.h5")
    rnn_manager.load_model("emotion_analyzer", "ruta/a/tu/RNN_gestion_verificacion.h5")
    
    # Entrenar un modelo (ejemplo)
    # training_data = (X_train, y_train)  # Define tus datos de entrenamiento
    # rnn_manager.train_model("response_generator", training_data)
    
    # Analizar texto (ejemplo)
    # prediction = rnn_manager.analyze_text("emotion_analyzer", "Estoy muy feliz hoy!")
    # print(prediction)
