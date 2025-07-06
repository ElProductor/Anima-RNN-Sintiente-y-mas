import os
import pickle
import random
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import get_custom_objects
import time

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# Importar componentes de la IA
from RNN import QuantumANIMA, QuantumMemoryBank, ConversationContext, EmotionType
from database_integration import IntegratedRNNSystem, ChatbotDatabaseIntegration
from database_config import DatabaseManager

app = Flask(__name__)
CORS(app)

# Configuración de rutas de archivos
TOKENIZER_PATH = 'tokenizer.pkl'
MEMORY_BANK_PATH = 'quantum_memory.pkl'
MODEL_PATH = 'chatbot_state_model.h5'

# Variables globales para componentes de la IA
anima_ai = None
tokenizer = None
memory_bank = None
integrated_system = None
db_manager = None

def create_default_tokenizer():
    """Crea un tokenizer por defecto con vocabulario básico."""
    tokenizer = Tokenizer(num_words=10000, oov_token="<unk>")
    
    basic_vocab = [
        "hola", "hello", "hi", "buenos", "días", "buenas", "tardes", "noches",
        "cómo", "estás", "how", "are", "you", "bien", "mal", "good", "bad",
        "gracias", "thank", "you", "de", "nada", "welcome", "adiós", "goodbye",
        "bye", "hasta", "luego", "see", "later", "qué", "what", "cuándo", "when",
        "dónde", "where", "por", "why", "porque", "because", "sí", "yes", "no",
        "ayuda", "help", "problema", "problem", "solución", "solution"
    ]
    
    tokenizer.fit_on_texts([" ".join(basic_vocab)])
    return tokenizer

def load_tokenizer(path=TOKENIZER_PATH):
    """Carga el tokenizador desde un archivo o crea uno nuevo."""
    try:
        if os.path.exists(path):
            with open(path, 'rb') as handle:
                tokenizer = pickle.load(handle)
            logger.info(f"Tokenizer cargado desde {path}")
            return tokenizer
        
        logger.warning(f"Archivo {path} no encontrado. Creando tokenizer por defecto.")
        tokenizer = create_default_tokenizer()
            
        with open(path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Tokenizer por defecto guardado en {path}")
            
        return tokenizer
    except Exception as e:
        logger.error(f"Error cargando tokenizer: {e}")
        return create_default_tokenizer()

def initialize_ai_components():
    """Inicializa todos los componentes de la IA."""
    global anima_ai, tokenizer, memory_bank, integrated_system, db_manager
    
    try:
        # Cargar tokenizer
        tokenizer = load_tokenizer()
        
        # Inicializar banco de memoria
        memory_bank = QuantumMemoryBank(memory_file=MEMORY_BANK_PATH)
        logger.info("Banco de memoria inicializado correctamente")
        
        # Inicializar QuantumANIMA
        anima_ai = QuantumANIMA(
            tokenizer=tokenizer, 
            memory_bank=memory_bank, 
            model_path=MODEL_PATH
        )
        logger.info("QuantumANIMA inicializado correctamente")
        
        # Inicializar sistema integrado de bases de datos
        integrated_system = IntegratedRNNSystem(chatbot_instance=anima_ai)
        db_manager = DatabaseManager()
        logger.info("Sistema integrado de bases de datos inicializado correctamente")
        
        return True
        
    except Exception as e:
        logger.error(f"Error crítico en inicialización: {e}")
        return False

def validate_message(message):
    """Valida y limpia el mensaje del usuario."""
    if not message:
        return None, "No se proporcionó ningún mensaje."
    if not isinstance(message, str):
        return None, "El mensaje debe ser una cadena de texto."
    
    message = message.strip()
    if len(message) == 0:
        return None, "El mensaje no puede estar vacío."
    if len(message) > 1000:
        return None, "El mensaje es demasiado largo (máximo 1000 caracteres)."
    
    return message, None

# Ruta principal
@app.route('/')
def home():
    """Endpoint raíz del servidor."""
    return jsonify({
        "status": "running",
        "service": "Quantum ANIMA AI Server",
        "timestamp": datetime.now().isoformat()
    })

# Health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy' if anima_ai else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'anima_ai': anima_ai is not None,
            'tokenizer': tokenizer is not None,
            'memory_bank': memory_bank is not None
        },
        'ai_initialized': anima_ai is not None,
        'database_system_initialized': integrated_system is not None
    })

@app.route('/database/status')
def database_status():
    """Endpoint para verificar el estado de las bases de datos."""
    if not db_manager:
        return jsonify({'error': 'Sistema de bases de datos no inicializado'}), 500
    
    try:
        status = db_manager.get_system_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error obteniendo estado de bases de datos: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/database/backup', methods=['POST'])
def backup_databases():
    """Endpoint para crear respaldo de todas las bases de datos."""
    if not integrated_system:
        return jsonify({'error': 'Sistema integrado no inicializado'}), 500
    
    try:
        integrated_system.backup_system()
        return jsonify({
            'message': 'Respaldo completado exitosamente',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error creando respaldo: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analytics/conversation/<user_id>')
def get_conversation_analytics(user_id):
    """Obtiene análisis de conversaciones para un usuario específico."""
    if not integrated_system or not integrated_system.chatbot_integration:
        return jsonify({'error': 'Sistema de chatbot no inicializado'}), 500
    
    try:
        history = integrated_system.chatbot_integration.get_user_context(user_id, limit=20)
        return jsonify({
            'user_id': user_id,
            'conversation_history': history,
            'total_conversations': len(history)
        })
    except Exception as e:
        logger.error(f"Error obteniendo análisis de conversación: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analytics/emotions/<user_id>')
def get_emotion_analytics(user_id):
    """Obtiene análisis emocional para un usuario específico."""
    if not integrated_system or not integrated_system.emotion_integration:
        return jsonify({'error': 'Sistema de análisis emocional no inicializado'}), 500
    
    try:
        days = request.args.get('days', 30, type=int)
        emotion_profile = integrated_system.emotion_integration.get_user_emotion_profile(user_id, days)
        return jsonify({
            'user_id': user_id,
            'emotion_profile': emotion_profile
        })
    except Exception as e:
        logger.error(f"Error obteniendo análisis emocional: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analytics/realtime/performance')
def get_realtime_performance():
    """Obtiene métricas de rendimiento del modelo de tiempo real."""
    if not integrated_system or not integrated_system.realtime_integration:
        return jsonify({'error': 'Sistema de tiempo real no inicializado'}), 500
    
    try:
        days = request.args.get('days', 7, type=int)
        performance = integrated_system.realtime_integration.get_performance_summary(days)
        return jsonify(performance)
    except Exception as e:
        logger.error(f"Error obteniendo métricas de rendimiento: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def system_dashboard():
    """Endpoint para obtener dashboard completo del sistema."""
    if not integrated_system:
        return jsonify({'error': 'Sistema integrado no inicializado'}), 500
    
    try:
        dashboard = integrated_system.get_system_dashboard()
        return jsonify(dashboard)
    except Exception as e:
        logger.error(f"Error obteniendo dashboard: {e}")
        return jsonify({'error': str(e)}), 500

# Endpoint de chat
@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint principal para el chat."""
    if not anima_ai or not integrated_system:
        return jsonify({
            'error': 'La IA no está inicializada correctamente.',
            'status': 'service_unavailable'
        }), 503
    
    try:
        data = request.json
        if not data:
            return jsonify({
                'error': 'No se proporcionaron datos JSON.',
                'status': 'bad_request'
            }), 400
        
        user_message = data.get('message')
        user_id = data.get('user_id', 'default_user')
        session_id = data.get('session_id', f'session_{int(time.time())}')
        
        validated_message, error = validate_message(user_message)
        if error:
            return jsonify({
                'error': error,
                'status': 'bad_request'
            }), 400
        
        # Procesar entrada usando el sistema integrado
        result = integrated_system.process_user_input(
            user_input=validated_message,
            user_id=user_id,
            session_id=session_id
        )
        
        # Extraer respuesta del chatbot
        chatbot_response = result.get('chatbot_response', {})
        emotion_analysis = result.get('emotion_analysis', {})
        
        response_data = {
            'response': chatbot_response.get('response', 'Error al generar respuesta'),
            'confidence_score': chatbot_response.get('confidence_score', 0.0),
            'response_time_ms': chatbot_response.get('response_time_ms', 0),
            'session_id': session_id,
            'emotion': {
                'detected_emotion': emotion_analysis.get('detected_emotion', 'neutral'),
                'confidence': emotion_analysis.get('confidence_score', 0.0),
                'emotional_context': emotion_analysis.get('emotional_context', {})
            },
            'timestamp': time.time(),
            'status': chatbot_response.get('status', 'success')
        }
        
        logger.info(f"Chat procesado - Usuario: {user_id}, Emoción: {emotion_analysis.get('detected_emotion', 'neutral')}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error en el endpoint de chat: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'response': 'Lo siento, ocurrió un error interno.',
            'status': 'error'
        }), 500

# Inicializar la IA al arrancar la aplicación
if __name__ == '__main__':
    if initialize_ai_components():
        # Iniciar el servidor Flask
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.critical("No se pudo inicializar la IA. El servidor no se iniciará.")
