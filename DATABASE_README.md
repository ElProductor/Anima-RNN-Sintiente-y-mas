# Sistema de Bases de Datos para Redes Neuronales ANIMA

Este documento describe el sistema de bases de datos exclusivas implementado para cada una de las redes neuronales del proyecto ANIMA.

## Arquitectura del Sistema

El sistema está compuesto por tres bases de datos SQLite independientes, cada una optimizada para un tipo específico de red neuronal:

### 1. Base de Datos del Chatbot (`chatbot_rnn.db`)
**Propósito**: Almacenar conversaciones, patrones de entrenamiento y métricas del modelo de generación de respuestas.

**Tablas principales**:
- `conversations`: Historial completo de conversaciones
- `training_patterns`: Patrones de entrada y salida para entrenamiento
- `model_metrics`: Métricas de rendimiento del modelo

### 2. Base de Datos de Tiempo Real (`realtime_rnn.db`)
**Propósito**: Gestionar secuencias temporales, predicciones y análisis de rendimiento en tiempo real.

**Tablas principales**:
- `time_sequences`: Secuencias temporales y predicciones
- `performance_metrics`: Métricas de rendimiento del modelo
- `detected_patterns`: Patrones detectados en los datos

### 3. Base de Datos Emocional (`emotion_rnn.db`)
**Propósito**: Almacenar análisis emocionales, memorias cuánticas y patrones emocionales.

**Tablas principales**:
- `emotion_analysis`: Análisis emocionales de textos
- `quantum_memories`: Sistema de memoria cuántica
- `emotion_patterns`: Patrones emocionales de usuarios
- `emotion_model_metrics`: Métricas del modelo de análisis emocional

## Archivos del Sistema

### `database_config.py`
Contiene las clases principales para cada base de datos:
- `ChatbotDatabase`: Gestión de la base de datos del chatbot
- `RealTimeRNNDatabase`: Gestión de la base de datos de tiempo real
- `EmotionRNNDatabase`: Gestión de la base de datos emocional
- `DatabaseManager`: Coordinador principal del sistema

### `database_integration.py`
Contiene las clases de integración que conectan las RNN con sus bases de datos:
- `ChatbotDatabaseIntegration`: Integración chatbot-base de datos
- `RealTimeDatabaseIntegration`: Integración tiempo real-base de datos
- `EmotionDatabaseIntegration`: Integración análisis emocional-base de datos
- `IntegratedRNNSystem`: Sistema coordinador completo

## Uso del Sistema

### Inicialización
```python
from database_integration import IntegratedRNNSystem
from RNN import QuantumANIMA

# Inicializar la RNN principal
chatbot = QuantumANIMA()

# Inicializar el sistema integrado
system = IntegratedRNNSystem(chatbot_instance=chatbot)
```

### Procesamiento de Conversaciones
```python
# Procesar entrada del usuario
result = system.process_user_input(
    user_input="Hola, ¿cómo estás?",
    user_id="usuario123",
    session_id="session_001"
)

# El resultado incluye:
# - Respuesta del chatbot
# - Análisis emocional
# - Métricas de rendimiento
# - Todo se guarda automáticamente en las bases de datos correspondientes
```

### Análisis y Métricas
```python
# Obtener historial de conversaciones
history = system.chatbot_integration.get_user_context("usuario123", limit=10)

# Obtener perfil emocional del usuario
emotion_profile = system.emotion_integration.get_user_emotion_profile("usuario123", days=30)

# Obtener métricas de rendimiento
performance = system.realtime_integration.get_performance_summary(days=7)
```

## Endpoints de la API

El servidor Flask ha sido actualizado con nuevos endpoints para gestionar las bases de datos:

### Endpoints de Estado
- `GET /health` - Estado general del servidor
- `GET /database/status` - Estado de las bases de datos
- `GET /dashboard` - Dashboard completo del sistema

### Endpoints de Análisis
- `GET /analytics/conversation/<user_id>` - Análisis de conversaciones por usuario
- `GET /analytics/emotions/<user_id>` - Perfil emocional por usuario
- `GET /analytics/realtime/performance` - Métricas de rendimiento en tiempo real

### Endpoints de Gestión
- `POST /database/backup` - Crear respaldo de todas las bases de datos
- `POST /chat` - Endpoint principal (actualizado con integración de bases de datos)

## Características Principales

### 1. Separación de Responsabilidades
Cada red neuronal tiene su propia base de datos optimizada para sus necesidades específicas:
- **Chatbot**: Enfocado en conversaciones y generación de respuestas
- **Tiempo Real**: Optimizado para secuencias temporales y predicciones
- **Emocional**: Especializado en análisis emocional y memoria cuántica

### 2. Integración Transparente
Las bases de datos se integran automáticamente con las RNN existentes sin modificar su lógica principal.

### 3. Análisis Avanzado
Cada base de datos proporciona métricas y análisis específicos:
- Precisión de predicciones
- Patrones emocionales
- Rendimiento de conversaciones
- Tendencias temporales

### 4. Respaldos Automáticos
Sistema de respaldo integrado que permite crear copias de seguridad de todas las bases de datos.

### 5. Escalabilidad
Diseño modular que permite agregar nuevas RNN y sus bases de datos correspondientes fácilmente.

## Estructura de Datos

### Conversaciones del Chatbot
```json
{
  "user_id": "usuario123",
  "session_id": "session_001",
  "user_input": "¿Cómo estás?",
  "bot_response": "¡Hola! Estoy muy bien, gracias por preguntar.",
  "confidence_score": 0.85,
  "response_time_ms": 150,
  "timestamp": "2024-01-15T10:30:00"
}
```

### Análisis Emocional
```json
{
  "text_input": "Me siento muy feliz hoy",
  "detected_emotion": "alegría",
  "confidence_score": 0.92,
  "emotional_context": {
    "recent_emotions": ["neutral", "alegría", "alegría"],
    "dominant_emotion": "alegría",
    "emotion_stability": true
  }
}
```

### Predicciones de Tiempo Real
```json
{
  "sequence_id": "seq_1642248600000",
  "sequence_data": [1.2, 1.5, 1.8, 2.1],
  "predicted_value": 2.3,
  "actual_value": 2.25,
  "prediction_error": 0.05,
  "attention_weights": [0.1, 0.2, 0.3, 0.4]
}
```

## Monitoreo y Mantenimiento

### Métricas Clave
- **Chatbot**: Tiempo de respuesta, puntuación de confianza, satisfacción del usuario
- **Tiempo Real**: Error de predicción, precisión, tiempo de procesamiento
- **Emocional**: Precisión de detección emocional, estabilidad emocional del usuario

### Respaldos
El sistema crea respaldos automáticos con timestamp:
- `chatbot_rnn_YYYYMMDD_HHMMSS.db`
- `realtime_rnn_YYYYMMDD_HHMMSS.db`
- `emotion_rnn_YYYYMMDD_HHMMSS.db`

### Logs
Todos los componentes generan logs detallados para monitoreo y debugging:
- Inicialización de componentes
- Procesamiento de datos
- Errores y excepciones
- Métricas de rendimiento

## Beneficios del Sistema

1. **Especialización**: Cada base de datos está optimizada para su tipo de RNN específica
2. **Escalabilidad**: Fácil agregar nuevas RNN y bases de datos
3. **Mantenimiento**: Bases de datos independientes facilitan el mantenimiento
4. **Análisis**: Métricas detalladas para cada componente
5. **Respaldos**: Sistema de respaldo integrado y automatizado
6. **Integración**: Funciona transparentemente con el código existente

## Próximos Pasos

1. Implementar análisis predictivo basado en datos históricos
2. Agregar alertas automáticas para anomalías en los datos
3. Desarrollar dashboard web para visualización de métricas
4. Implementar sincronización entre bases de datos para análisis cruzado
5. Agregar soporte para bases de datos distribuidas

---

**Nota**: Este sistema ha sido diseñado para ser completamente compatible con el código existente del proyecto ANIMA, proporcionando una base sólida para el crecimiento y evolución del sistema de IA.