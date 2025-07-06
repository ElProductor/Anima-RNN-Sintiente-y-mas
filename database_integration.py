from database_config import DatabaseManager, ChatbotDatabase, RealTimeRNNDatabase, EmotionRNNDatabase
import time
import json
from typing import Dict, List, Any, Optional
import logging
from RNN import ConversationContext, EmotionType # Importar ConversationContext y EmotionType

# Configuración de logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DatabaseIntegration")

class ChatbotDatabaseIntegration:
    """Integración de la base de datos con la RNN de generación de respuestas"""
    
    def __init__(self, chatbot_instance, database: ChatbotDatabase):
        self.chatbot = chatbot_instance
        self.db = database
        logger.info("Integración de base de datos del Chatbot inicializada")
    
    def generate_and_save_response(self, user_input: str, user_id: str = "anonymous", 
                                 session_id: str = None) -> Dict[str, Any]:
        """Genera una respuesta y la guarda en la base de datos"""
        start_time = time.time()
        
        try:
            # Logging detallado para debugging
            logger.debug(f"Iniciando generate_and_save_response - user_input: {type(user_input)}, user_id: {type(user_id)}, session_id: {type(session_id)}")
            
            # Validar parámetros de entrada
            if not user_input or not isinstance(user_input, str):
                logger.error(f"user_input inválido: {user_input}, tipo: {type(user_input)}")
                raise ValueError("user_input debe ser una cadena válida")
            
            if not user_id:
                user_id = "anonymous"
            
            # Generar session_id si es None
            if session_id is None:
                session_id = f"session_{int(time.time())}"
            
            logger.debug(f"Parámetros validados - user_id: {user_id}, session_id: {session_id}")
            
            # Verificar que self.chatbot existe
            if self.chatbot is None:
                logger.error("self.chatbot es None")
                raise ValueError("Chatbot no inicializado")
            
            # Crear un contexto de conversación
            logger.debug("Creando contexto de conversación...")
            context = ConversationContext(user_id=user_id, session_id=session_id)
            logger.debug(f"Contexto creado: {context}")
            
            # Generar respuesta usando la RNN, pasando el contexto
            logger.debug("Generando respuesta con chatbot...")
            response = self.chatbot.generate_response(user_input, context)
            logger.debug(f"Respuesta generada: {response}, tipo: {type(response)}")
            
            # Entrenar el modelo con la nueva interacción
            logger.debug("Entrenando modelo con nueva interacción...")
            self.chatbot.train_on_interaction(user_input, response, context)
            logger.debug("Modelo entrenado con nueva interacción")
            
            # Validar que la respuesta no sea None
            if response is None or not isinstance(response, str):
                logger.warning(f"Respuesta inválida del chatbot: {response}")
                response = "Lo siento, no pude generar una respuesta en este momento."
            
            # Calcular tiempo de respuesta (asegurar que sea entero)
            current_time = time.time()
            response_time_ms = int((current_time - start_time) * 1000)
            logger.debug(f"Tiempo de respuesta calculado: {response_time_ms}ms")
            
            # Calcular score de confianza (asegurar que sea float válido)
            confidence_score = min(0.95, max(0.0, len(response) / 100.0 + 0.3))
            logger.debug(f"Score de confianza: {confidence_score}")
            
            # Verificar que self.db existe
            if self.db is None:
                logger.error("self.db es None")
                raise ValueError("Base de datos no inicializada")
            
            # Guardar en la base de datos
            logger.debug("Guardando conversación en base de datos...")
            logger.debug(f"Parámetros para save_conversation: user_id={user_id} ({type(user_id)}), session_id={session_id} ({type(session_id)}), user_input={user_input[:50]}... ({type(user_input)}), bot_response={response[:50]}... ({type(response)}), confidence_score={confidence_score} ({type(confidence_score)}), response_time_ms={response_time_ms} ({type(response_time_ms)})")
            
            self.db.save_conversation(
                user_id=user_id,
                session_id=session_id,
                user_input=user_input,
                bot_response=response,
                confidence_score=confidence_score,
                response_time_ms=response_time_ms
            )
            
            logger.info(f"Conversación guardada - Usuario: {user_id}, Tiempo: {response_time_ms}ms")
            
            return {
                'response': response,
                'confidence_score': confidence_score,
                'response_time_ms': response_time_ms,
                'session_id': session_id,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error generando respuesta: {str(e)}")
            logger.error(f"Error completo: {repr(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Asegurar que todos los valores sean válidos en caso de error
            safe_session_id = session_id if session_id else f"session_{int(time.time())}"
            safe_response_time = int((time.time() - start_time) * 1000)
            
            return {
                'response': "Lo siento, ocurrió un error al procesar tu mensaje.",
                'confidence_score': 0.0,
                'response_time_ms': safe_response_time,
                'session_id': safe_session_id,
                'status': 'error',
                'error': str(e)
            }
    
    def get_user_context(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Obtiene el contexto reciente del usuario para mejorar las respuestas"""
        if not user_id:
            return []
        
        # Validar que limit sea un entero válido
        if limit is None or not isinstance(limit, int) or limit <= 0:
            limit = 5
        
        return self.db.get_conversation_history(user_id, limit)
    
    def save_training_session(self, model_version: str, metrics: Dict[str, Any]):
        """Guarda métricas de una sesión de entrenamiento"""
        if not model_version:
            model_version = f"version_{int(time.time())}"
        
        # Validar y asegurar tipos correctos para las métricas
        accuracy = float(metrics.get('accuracy', 0.0)) if metrics.get('accuracy') is not None else 0.0
        loss = float(metrics.get('loss', 0.0)) if metrics.get('loss') is not None else 0.0
        training_time = int(metrics.get('training_time_minutes', 0)) if metrics.get('training_time_minutes') is not None else 0
        epoch_count = int(metrics.get('epochs', 0)) if metrics.get('epochs') is not None else 0
        
        self.db.save_training_metrics(
            model_version=model_version,
            accuracy=accuracy,
            loss=loss,
            training_time_minutes=training_time,
            epoch_count=epoch_count
        )
        logger.info(f"Métricas de entrenamiento guardadas para {model_version}")

class RealTimeDatabaseIntegration:
    """Integración de la base de datos con la RNN de tiempo real"""
    
    def __init__(self, realtime_model, database: RealTimeRNNDatabase):
        self.model = realtime_model
        self.db = database
        self.prediction_cache = {}
        logger.info("Integración de base de datos RNN Tiempo Real inicializada")
    
    def predict_and_save(self, sequence_data: List[float], sequence_id: str = None) -> Dict[str, Any]:
        """Realiza una predicción y la guarda en la base de datos"""
        if sequence_id is None:
            sequence_id = f"seq_{int(time.time() * 1000)}"
        
        # Validar sequence_data
        if not sequence_data or not isinstance(sequence_data, list):
            sequence_data = [0.0]
        
        # Asegurar que todos los elementos sean float
        try:
            sequence_data = [float(x) for x in sequence_data if x is not None]
        except (ValueError, TypeError):
            sequence_data = [0.0]
        
        try:
            # Realizar predicción (simulada por ahora)
            predicted_value = self._simulate_prediction(sequence_data)
            attention_weights = self._simulate_attention_weights(len(sequence_data))
            
            # Guardar predicción
            self.db.save_prediction(
                sequence_id=sequence_id,
                sequence_data=sequence_data,
                predicted_value=predicted_value,
                attention_weights=attention_weights
            )
            
            # Cachear para comparación futura
            self.prediction_cache[sequence_id] = {
                'predicted_value': predicted_value,
                'timestamp': time.time()
            }
            
            logger.info(f"Predicción guardada - ID: {sequence_id}, Valor: {predicted_value:.4f}")
            
            return {
                'sequence_id': sequence_id,
                'predicted_value': predicted_value,
                'attention_weights': attention_weights,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            return {
                'sequence_id': sequence_id,
                'predicted_value': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    def update_with_actual_value(self, sequence_id: str, actual_value: float):
        """Actualiza una predicción con el valor real observado"""
        if not sequence_id:
            logger.warning("sequence_id no puede ser None o vacío")
            return
        
        # Validar actual_value
        if actual_value is None:
            logger.warning("actual_value no puede ser None")
            return
        
        try:
            actual_value = float(actual_value)
        except (ValueError, TypeError):
            logger.warning("actual_value debe ser convertible a float")
            return
        
        if sequence_id in self.prediction_cache:
            predicted_value = self.prediction_cache[sequence_id]['predicted_value']
            error = abs(predicted_value - actual_value)
            
            logger.info(f"Valor real recibido - ID: {sequence_id}, "
                       f"Predicho: {predicted_value:.4f}, Real: {actual_value:.4f}, "
                       f"Error: {error:.4f}")
            
            del self.prediction_cache[sequence_id]
    
    def _simulate_prediction(self, sequence_data: List[float]) -> float:
        """Simula una predicción basada en los datos de secuencia"""
        if not sequence_data:
            return 0.0
        
        # Predicción simple basada en tendencia
        if len(sequence_data) >= 2:
            trend = sequence_data[-1] - sequence_data[-2]
            return sequence_data[-1] + (trend * 0.8)
        else:
            return sequence_data[-1] * 1.1
    
    def _simulate_attention_weights(self, sequence_length: int) -> List[float]:
        """Simula pesos de atención"""
        if sequence_length <= 0:
            return []
        
        import random
        weights = [random.random() for _ in range(sequence_length)]
        total = sum(weights)
        return [w/total for w in weights] if total > 0 else [1.0/sequence_length] * sequence_length
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Obtiene un resumen del rendimiento del modelo"""
        # Validar days
        if days is None or not isinstance(days, int) or days <= 0:
            days = 7
        
        accuracy_data = self.db.get_prediction_accuracy(days)
        return {
            'period_days': days,
            'accuracy_data': accuracy_data,
            'model_status': 'active'
        }

class EmotionDatabaseIntegration:
    """Integración de la base de datos con la RNN de análisis emocional"""
    
    def __init__(self, emotion_analyzer, database: EmotionRNNDatabase):
        self.analyzer = emotion_analyzer
        self.db = database
        self.emotion_history = {}
        logger.info("Integración de base de datos RNN Emocional inicializada")
    
    def analyze_and_save_emotion(self, text_input: str, user_id: str = None, 
                               session_id: str = None) -> Dict[str, Any]:
        """Analiza la emoción del texto y la guarda en la base de datos"""
        try:
            # Validar text_input
            if not text_input or not isinstance(text_input, str):
                raise ValueError("text_input debe ser una cadena válida")
            
            # Simular análisis emocional
            emotion_result = self._simulate_emotion_analysis(text_input)
            
            # Guardar análisis
            self.db.save_emotion_analysis(
                text_input=text_input,
                detected_emotion=emotion_result['emotion'],
                confidence_score=emotion_result['confidence'],
                user_id=user_id,
                session_id=session_id
            )
            
            # Actualizar historial del usuario
            if user_id:
                if user_id not in self.emotion_history:
                    self.emotion_history[user_id] = []
                
                self.emotion_history[user_id].append({
                    'emotion': emotion_result['emotion'],
                    'timestamp': time.time()
                })
                
                # Mantener solo las últimas 10 emociones
                self.emotion_history[user_id] = self.emotion_history[user_id][-10:]
            
            logger.info(f"Emoción detectada - Usuario: {user_id}, "
                       f"Emoción: {emotion_result['emotion']}, "
                       f"Confianza: {emotion_result['confidence']:.3f}")
            
            return {
                'detected_emotion': emotion_result['emotion'],
                'confidence_score': emotion_result['confidence'],
                'emotional_context': self._get_emotional_context(user_id),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error en análisis emocional: {str(e)}")
            return {
                'detected_emotion': 'neutral',
                'confidence_score': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    def save_quantum_memory(self, content: str, emotion: str, importance: float):
        """Guarda una memoria cuántica en la base de datos"""
        # Validar parámetros
        if not content:
            content = "memoria_vacía"
        if not emotion:
            emotion = "neutral"
        if importance is None or not isinstance(importance, (int, float)):
            importance = 0.5
        
        memory_id = f"mem_{int(time.time() * 1000)}"
        
        # Simular vector de embedding
        embedding_vector = [hash(content + str(i)) % 100 / 100.0 for i in range(128)]
        
        self.db.save_quantum_memory(
            memory_id=memory_id,
            content=content,
            emotion=emotion,
            importance_score=float(importance),
            embedding_vector=embedding_vector
        )
        
        logger.info(f"Memoria cuántica guardada - ID: {memory_id}, Emoción: {emotion}")
        return memory_id
    
    def _simulate_emotion_analysis(self, text: str) -> Dict[str, Any]:
        """Simula análisis emocional del texto"""
        if not text:
            return {'emotion': 'neutral', 'confidence': 0.0}
        
        emotions = ['alegría', 'tristeza', 'enojo', 'miedo', 'sorpresa', 'neutral']
        
        # Análisis simple basado en palabras clave
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['feliz', 'contento', 'alegre', 'genial']):
            return {'emotion': 'alegría', 'confidence': 0.85}
        elif any(word in text_lower for word in ['triste', 'deprimido', 'mal']):
            return {'emotion': 'tristeza', 'confidence': 0.80}
        elif any(word in text_lower for word in ['enojado', 'furioso', 'molesto']):
            return {'emotion': 'enojo', 'confidence': 0.75}
        elif any(word in text_lower for word in ['asustado', 'miedo', 'terror']):
            return {'emotion': 'miedo', 'confidence': 0.70}
        elif any(word in text_lower for word in ['sorprendido', 'increíble', 'wow']):
            return {'emotion': 'sorpresa', 'confidence': 0.65}
        else:
            return {'emotion': 'neutral', 'confidence': 0.60}
    
    def _get_emotional_context(self, user_id: str) -> Dict[str, Any]:
        """Obtiene el contexto emocional del usuario"""
        if not user_id or user_id not in self.emotion_history:
            return {'recent_emotions': [], 'dominant_emotion': 'neutral'}
        
        recent_emotions = self.emotion_history[user_id]
        
        # Calcular emoción dominante
        emotion_counts = {}
        for entry in recent_emotions:
            emotion = entry['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral'
        
        return {
            'recent_emotions': [e['emotion'] for e in recent_emotions[-5:]],
            'dominant_emotion': dominant_emotion,
            'emotion_stability': len(set(e['emotion'] for e in recent_emotions)) <= 2
        }
    
    def get_user_emotion_profile(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Obtiene el perfil emocional completo del usuario"""
        if not user_id:
            return {}
        
        # Validar days
        if days is None or not isinstance(days, int) or days <= 0:
            days = 30
        
        return self.db.get_emotion_statistics(user_id, days)

class IntegratedRNNSystem:
    """Sistema integrado que coordina todas las RNN con sus bases de datos"""
    
    def __init__(self, chatbot_instance=None, realtime_model=None, emotion_analyzer=None):
        # Inicializar gestor de bases de datos
        self.db_manager = DatabaseManager()
        
        # Inicializar integraciones
        self.chatbot_integration = ChatbotDatabaseIntegration(
            chatbot_instance, self.db_manager.chatbot_db
        ) if chatbot_instance else None
        
        self.realtime_integration = RealTimeDatabaseIntegration(
            realtime_model, self.db_manager.realtime_db
        ) if realtime_model else None
        
        self.emotion_integration = EmotionDatabaseIntegration(
            emotion_analyzer, self.db_manager.emotion_db
        ) if emotion_analyzer else None
        
        logger.info("Sistema RNN integrado inicializado")
    
    def process_user_input(self, user_input: str, user_id: str = "anonymous", 
                          session_id: str = None) -> Dict[str, Any]:
        """Procesa entrada del usuario usando todas las RNN disponibles"""
        # Validar user_input
        if not user_input or not isinstance(user_input, str):
            return {
                'error': 'user_input debe ser una cadena válida',
                'status': 'error'
            }
        
        # Validar user_id
        if not user_id:
            user_id = "anonymous"
        
        # Generar session_id si es None
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        results = {
            'user_input': user_input,
            'user_id': user_id,
            'session_id': session_id,
            'timestamp': time.time()
        }
        
        # Análisis emocional
        if self.emotion_integration:
            emotion_result = self.emotion_integration.analyze_and_save_emotion(
                user_input, user_id, session_id
            )
            results['emotion_analysis'] = emotion_result
        
        # Generación de respuesta del chatbot
        if self.chatbot_integration:
            chatbot_result = self.chatbot_integration.generate_and_save_response(
                user_input, user_id, session_id
            )
            results['chatbot_response'] = chatbot_result
        
        # Predicción en tiempo real (si hay datos secuenciales)
        if self.realtime_integration:
            # Convertir texto a secuencia numérica simple para demostración
            sequence_data = [float(ord(c)) for c in user_input[:10]]  # Primeros 10 caracteres
            realtime_result = self.realtime_integration.predict_and_save(sequence_data)
            results['realtime_prediction'] = realtime_result
        
        logger.info(f"Procesamiento completo - Usuario: {user_id}, Componentes: {list(results.keys())}")
        return results
    
    def get_comprehensive_analytics(self, user_id: str = None, days: int = 7) -> Dict[str, Any]:
        """Obtiene análisis comprehensivo del sistema"""
        analytics = {
            'period_days': days,
            'timestamp': time.time()
        }
        
        # Análisis del chatbot
        if self.chatbot_integration and user_id:
            analytics['conversation_history'] = self.chatbot_integration.get_user_context(user_id, 10)
        
        # Análisis emocional
        if self.emotion_integration and user_id:
            analytics['emotion_profile'] = self.emotion_integration.get_user_emotion_profile(user_id, days)
        
        # Análisis de tiempo real
        if self.realtime_integration:
            analytics['realtime_performance'] = self.realtime_integration.get_performance_summary(days)
        
        return analytics
    
    def save_system_state(self, state_description: str):
        """Guarda el estado actual del sistema"""
        state_data = {
            'timestamp': time.time(),
            'description': state_description,
            'active_components': {
                'chatbot': self.chatbot_integration is not None,
                'realtime': self.realtime_integration is not None,
                'emotion': self.emotion_integration is not None
            }
        }
        
        logger.info(f"Estado del sistema guardado: {state_description}")
        return state_data
    
    def close(self):
        """Cierra conexiones de base de datos"""
        if self.db_manager:
            self.db_manager.close_all_connections()
        logger.info("Sistema RNN integrado cerrado")

# Función auxiliar para crear instancia completa del sistema
def create_integrated_system(chatbot_instance=None, realtime_model=None, emotion_analyzer=None):
    """Crea una instancia completa del sistema integrado"""
    return IntegratedRNNSystem(
        chatbot_instance=chatbot_instance,
        realtime_model=realtime_model,
        emotion_analyzer=emotion_analyzer
    )

# Ejemplo de uso
if __name__ == "__main__":
    # Crear sistema integrado (sin instancias reales por ahora)
    system = create_integrated_system()
    
    # Ejemplo de procesamiento
    result = system.process_user_input(
        user_input="Hola, ¿cómo estás?",
        user_id="user123",
        session_id="session456"
    )
    
    print("Resultado del procesamiento:")
    print(json.dumps(result, indent=2, default=str))
    
    # Cerrar sistema
    system.close()