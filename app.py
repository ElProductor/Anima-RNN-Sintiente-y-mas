from flask import Flask, request, jsonify
from flask_cors import CORS
from RNN_manager import RNNManager
import numpy as np
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ANIMAApp")

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Inicializar el administrador de RNN con bases de datos integradas
rnn_manager = RNNManager()
logger.info("Aplicación ANIMA inicializada con sistema de bases de datos integrado")

@app.route('/load_model', methods=['POST'])
def load_model():
    """Endpoint para cargar un modelo"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No se proporcionaron datos JSON"}), 400
            
        model_name = data.get('model_name')
        model_path = data.get('model_path')
        
        if not model_name:
            return jsonify({"error": "model_name es requerido"}), 400
        
        # Para modelos principales, no se requiere ruta
        if model_name in ["chatbot", "response_generator"]:
            success = rnn_manager.load_model(model_name, "")
        elif not model_path:
            return jsonify({"error": "model_path es requerido para este tipo de modelo"}), 400
        else:
            success = rnn_manager.load_model(model_name, model_path)
        
        if success:
            logger.info(f"Modelo '{model_name}' cargado exitosamente")
            return jsonify({
                "status": "success",
                "message": f"Modelo '{model_name}' cargado exitosamente",
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": f"Error al cargar el modelo '{model_name}'"}), 500
            
    except Exception as e:
        logger.error(f"Error en endpoint load_model: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    """Endpoint para entrenar un modelo"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No se proporcionaron datos JSON"}), 400
            
        model_name = data.get('model_name')
        training_data = data.get('training_data')  # Debe ser un diccionario con 'X' y 'y'
        epochs = data.get('epochs', 10)
        
        if not model_name:
            return jsonify({"error": "model_name es requerido"}), 400
        
        # Para modelos principales, simular datos de entrenamiento si no se proporcionan
        if not training_data:
            if model_name in ["chatbot", "response_generator"]:
                # Simular datos de entrenamiento para el chatbot
                X_train = np.random.random((100, 50))  # 100 muestras, 50 características
                y_train = np.random.random((100, 30))  # 100 muestras, 30 salidas
            else:
                return jsonify({"error": "training_data es requerido para este modelo"}), 400
        else:
            try:
                X_train = np.array(training_data['X'])
                y_train = np.array(training_data['y'])
            except KeyError as e:
                return jsonify({"error": f"training_data debe contener claves 'X' y 'y': {str(e)}"}), 400
            except Exception as e:
                return jsonify({"error": f"Error procesando training_data: {str(e)}"}), 400
        
        success = rnn_manager.train_model(model_name, (X_train, y_train), epochs)
        
        if success:
            logger.info(f"Modelo '{model_name}' entrenado exitosamente")
            return jsonify({
                "status": "success",
                "message": f"Modelo '{model_name}' entrenado exitosamente por {epochs} épocas",
                "epochs": epochs,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": f"Error durante el entrenamiento del modelo '{model_name}'"}), 500
            
    except Exception as e:
        logger.error(f"Error en endpoint train_model: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    """Endpoint para analizar texto usando un modelo"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No se proporcionaron datos JSON"}), 400
            
        model_name = data.get('model_name')
        text = data.get('text')
        user_id = data.get('user_id', 'anonymous')
        session_id = data.get('session_id')
        
        if not model_name or not text:
            return jsonify({"error": "model_name y text son requeridos"}), 400
        
        result = rnn_manager.analyze_text(model_name, text, user_id, session_id)
        
        if "error" in result:
            logger.error(f"Error analizando texto: {result['error']}")
            return jsonify(result), 500
        
        logger.info(f"Texto analizado exitosamente con modelo '{model_name}' para usuario '{user_id}'")
        return jsonify({
            "status": "success",
            "model_used": model_name,
            "user_id": user_id,
            "analysis_result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error en endpoint analyze_text: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Nuevos endpoints para el sistema integrado

@app.route('/status', methods=['GET'])
def get_status():
    """Endpoint para obtener el estado del sistema"""
    try:
        status = rnn_manager.get_model_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error obteniendo estado: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/user_analytics/<user_id>', methods=['GET'])
def get_user_analytics(user_id):
    """Endpoint para obtener análisis de un usuario específico"""
    try:
        analytics = rnn_manager.get_user_analytics(user_id)
        
        if "error" in analytics:
            return jsonify(analytics), 500
        
        return jsonify(analytics)
    except Exception as e:
        logger.error(f"Error obteniendo análisis de usuario: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/backup', methods=['POST'])
def backup_data():
    """Endpoint para crear respaldo de las bases de datos"""
    try:
        result = rnn_manager.backup_all_data()
        
        if result.get("status") == "error":
            return jsonify(result), 500
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error creando respaldo: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint principal para chat (compatible con el sistema existente)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No se proporcionaron datos JSON"}), 400
            
        message = data.get('message')
        user_id = data.get('user_id', 'anonymous')
        session_id = data.get('session_id')
        
        if not message:
            return jsonify({"error": "message es requerido"}), 400
        
        # Usar el modelo de chatbot para generar respuesta
        result = rnn_manager.analyze_text('chatbot', message, user_id, session_id)
        
        if "error" in result:
            return jsonify({
                "error": result["error"],
                "response": "Lo siento, ocurrió un error al procesar tu mensaje."
            }), 500
        
        # Extraer respuesta del chatbot
        chatbot_response = result.get('chatbot_response', {})
        emotion_analysis = result.get('emotion_analysis', {})
        
        return jsonify({
            "response": chatbot_response.get('response', 'Error al generar respuesta'),
            "confidence_score": chatbot_response.get('confidence_score', 0.0),
            "response_time_ms": chatbot_response.get('response_time_ms', 0),
            "session_id": chatbot_response.get('session_id', session_id),
            "emotion": {
                "detected_emotion": emotion_analysis.get('detected_emotion', 'neutral'),
                "confidence": emotion_analysis.get('confidence_score', 0.0)
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error en endpoint chat: {str(e)}")
        return jsonify({
            "error": str(e),
            "response": "Lo siento, ocurrió un error interno."
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar la salud del sistema"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": "ANIMA RNN Manager",
        "version": "2.0.0"
    })

if __name__ == "__main__":
    logger.info("Iniciando servidor ANIMA en puerto 5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
