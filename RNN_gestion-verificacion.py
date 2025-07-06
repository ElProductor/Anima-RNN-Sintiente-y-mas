import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, MultiHeadAttention, Embedding, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from collections import defaultdict
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Configuración avanzada de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("macro_rnn.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MacroRNNSystem")

class MemoryBank:
    """Sistema de memoria que almacena contenido y emociones"""
    
    def __init__(self, capacity: int = 10000, embedding_dim: int = 256):
        self.capacity = capacity
        self.memory_states = {}
        self.embedding_dim = embedding_dim
        self.emotion_vectors = self._initialize_emotion_vectors()
        self.embedding_model = self._create_embedding_model()
        logger.info(f"MemoryBank inicializado con capacidad para {capacity} memorias")
    
    def _create_embedding_model(self) -> tf.keras.Model:
        """Construye un modelo de embeddings para contenido textual"""
        inputs = Input(shape=(1,), dtype=tf.string)
        vectorizer = tf.keras.layers.TextVectorization(max_tokens=10000, output_sequence_length=10)
        vectorizer.adapt([""])
        embedding = Embedding(input_dim=10000, output_dim=self.embedding_dim)(vectorizer(inputs))
        pooled = tf.reduce_mean(embedding, axis=1)
        return Model(inputs, pooled)
    
    def _initialize_emotion_vectors(self) -> Dict[str, np.ndarray]:
        """Crea vectores de embeddings para emociones"""
        emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
        return {emotion: np.random.normal(size=self.embedding_dim) for emotion in emotions}
    
    def store_memory(self, memory_id: str, content: str, emotion: str, importance: float):
        """Almacena una memoria con su representación vectorial"""
        if emotion not in self.emotion_vectors:
            emotion = 'neutral'
        
        content_vector = self.embedding_model(tf.constant([content])).numpy()[0]
        memory_vector = np.concatenate([
            content_vector,
            self.emotion_vectors[emotion],
            [importance]
        ])
        
        self.memory_states[memory_id] = {
            'content': content,
            'emotion': emotion,
            'importance': importance,
            'vector': memory_vector.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        logger.debug(f"Memoria almacenada: {memory_id[:8]}...")
    
    def retrieve_memory(self, query: str, emotion_context: str = None) -> List[Dict]:
        """Recupera memorias relevantes basadas en similitud coseno"""
        query_vector = self.embedding_model(tf.constant([query])).numpy()[0]
        retrieved_memories = []
        
        for memory_id, memory in self.memory_states.items():
            content_sim = self._cosine_similarity(query_vector, memory['vector'][:self.embedding_dim])
            emotion_factor = 1.0
            
            if emotion_context and emotion_context == memory['emotion']:
                emotion_factor = 1.5
            elif emotion_context:
                emotion_factor = self._cosine_similarity(
                    self.emotion_vectors[emotion_context],
                    memory['vector'][self.embedding_dim:2*self.embedding_dim]
                )
            
            score = content_sim * emotion_factor * memory['importance']
            
            if score > 0.4:  # Umbral de relevancia
                retrieved_memories.append({
                    'id': memory_id,
                    'content': memory['content'],
                    'score': score,
                    'emotion': memory['emotion']
                })
        
        return sorted(retrieved_memories, key=lambda x: x['score'], reverse=True)[:5]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calcula la similitud coseno entre dos vectores"""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot / (norm_a * norm_b + 1e-8)  # Evitar división por cero
    
    def save_memory(self, file_path: str):
        """Guarda la memoria en disco"""
        with open(file_path, 'w') as f:
            json.dump({
                'memory_states': self.memory_states,
                'emotion_vectors': {k: v.tolist() for k, v in self.emotion_vectors.items()}
            }, f, indent=2)
        logger.info(f"Memoria guardada en {file_path}")
    
    def load_memory(self, file_path: str):
        """Carga la memoria desde disco"""
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.memory_states = data['memory_states']
            self.emotion_vectors = {k: np.array(v) for k, v in data['emotion_vectors'].items()}
            logger.info(f"Memoria cargada desde {file_path} ({len(self.memory_states)} memorias)")
        else:
            logger.warning(f"Archivo de memoria no encontrado: {file_path}")

class EmotionClassifier:
    """Clasificador de emociones utilizando un modelo de aprendizaje profundo"""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128):
        self.model = self._build_model(vocab_size, embedding_dim)
        self.vectorizer = tf.keras.layers.TextVectorization(max_tokens=vocab_size, output_sequence_length=50)
        logger.info("Clasificador de emociones inicializado")
    
    def _build_model(self, vocab_size: int, embedding_dim: int) -> tf.keras.Model:
        """Construye un modelo de clasificación de emociones"""
        inputs = Input(shape=(1,), dtype=tf.string)
        x = self.vectorizer(inputs)
        x = Embedding(vocab_size, embedding_dim)(x)
        x = LSTM(64)(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(6, activation='softmax')(x)  # 6 emociones
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=CategoricalCrossentropy(), metrics=['accuracy'])
        return model
    
    def train(self, texts: List[str], emotions: List[str], epochs: int = 10):
        """Entrena el modelo con datos etiquetados"""
        emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
        label_map = {emotion: idx for idx, emotion in enumerate(emotion_labels)}
        
        self.vectorizer.adapt(texts)
        y = tf.one_hot([label_map[e] for e in emotions], depth=6)
        
        self.model.fit(np.array(texts), y, epochs=epochs, batch_size=32, validation_split=0.2)
        logger.info("Entrenamiento del clasificador de emociones completado")
    
    def analyze(self, text: str) -> Tuple[str, float]:
        """Predice la emoción de un texto"""
        prediction = self.model.predict([text], verbose=0)[0]
        emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
        emotion_idx = np.argmax(prediction)
        return emotion_labels[emotion_idx], float(prediction[emotion_idx])

class ResponseGenerator:
    """Generador de respuestas utilizando un modelo de transformador"""
    
    def __init__(self, max_length: int = 100):
        self.model = None
        self.tokenizer = tf.keras.layers.TextVectorization(max_tokens=20000, output_sequence_length=max_length)
        self.max_length = max_length
        logger.info("Generador de respuestas inicializado")
    
    def _build_model(self, vocab_size: int, embedding_dim: int = 256) -> tf.keras.Model:
        """Construye un modelo generativo basado en transformador"""
        inputs = Input(shape=(self.max_length,), dtype=tf.int32)
        x = Embedding(vocab_size, embedding_dim)(inputs)
        
        for _ in range(3):
            attn = MultiHeadAttention(num_heads=4, key_dim=embedding_dim)(x, x)
            attn = Dropout(0.2)(attn)
            x = LayerNormalization(epsilon=1e-6)(x + attn)
            ff = Dense(embedding_dim * 4, activation='relu')(x)
            ff = Dense(embedding_dim)(ff)
            ff = Dropout(0.2)(ff)
            x = LayerNormalization(epsilon=1e-6)(x + ff)
        
        outputs = Dense(vocab_size, activation='softmax')(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss=CategoricalCrossentropy(), metrics=['accuracy'])
        return model
    
    def train(self, conversations: List[Tuple[str, str]], epochs: int = 20):
        """Entrena el modelo con pares (input, output)"""
        all_texts = [input_text for input_text, _ in conversations] + [output_text for _, output_text in conversations]
        self.tokenizer.adapt(all_texts)
        
        inputs = []
        outputs = []
        for input_text, output_text in conversations:
            inputs.append(self.tokenizer(input_text).numpy())
            outputs.append(self.tokenizer(output_text).numpy())
        
        inputs = tf.ragged.constant(inputs).to_tensor()
        outputs = tf.ragged.constant(outputs).to_tensor()
        
        vocab_size = len(self.tokenizer.get_vocabulary())
        self.model = self._build_model(vocab_size)
        
        self.model.fit(inputs, outputs, epochs=epochs, batch_size=32, validation_split=0.1)
        logger.info("Entrenamiento del generador de respuestas completado")
    
    def generate_response(self, input_text: str, max_length: int = 50) -> str:
        """Genera una respuesta a partir del texto de entrada"""
        if not self.model:
            logger.error("Modelo no entrenado")
            return "Lo siento, aún no estoy preparado para responder."
        
        input_seq = self.tokenizer([input_text]).numpy()
        generated = []
        
        for _ in range(max_length):
            probs = self.model.predict(input_seq, verbose=0)[0, -1, :]
            next_token = np.argmax(probs)
            
            if next_token == 0:  # 0 es típicamente padding
                break
                
            generated.append(next_token)
            input_seq = np.append(input_seq, [[next_token]], axis=1)
        
        vocab = self.tokenizer.get_vocabulary()
        return ' '.join([vocab[token] for token in generated])

class MacroRNNSystem:
    """Sistema Macro RNN mejorado con componentes integrados"""
    
    def __init__(self, memory_file: str = "quantum_memory.json"):
        self.memory_bank = MemoryBank()
        self.emotion_classifier = EmotionClassifier()
        self.response_generator = ResponseGenerator()
        self.user_feedback = defaultdict(list)
        self.memory_file = memory_file
        
        if os.path.exists(memory_file):
            self.memory_bank.load_memory(memory_file)
        
        logger.info("Macro RNN System inicializado")
    
    def analyze_emotion(self, text: str, user_id: str) -> Dict[str, Any]:
        """Analiza la emoción del texto usando el modelo especializado"""
        emotion, confidence = self.emotion_classifier.analyze(text)
        
        memory_id = f"{user_id}_{datetime.now().timestamp()}"
        self.memory_bank.store_memory(memory_id, text, emotion, confidence)
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'memory_id': memory_id
        }
    
    def generate_response(self, input_text: str, user_id: str) -> str:
        """Genera respuesta considerando contexto y memoria"""
        emotion_result = self.analyze_emotion(input_text, user_id)
        
        context_memories = self.memory_bank.retrieve_memory(
            query=input_text,
            emotion_context=emotion_result['emotion']
        )
        
        enriched_input = self._enrich_input(input_text, context_memories)
        response = self.response_generator.generate_response(enriched_input)
        
        self._log_interaction(user_id, input_text, response, emotion_result)
        return response
    
    def _enrich_input(self, input_text: str, memories: List[Dict]) -> str:
        """Combina texto de entrada con memorias relevantes"""
        enriched = f"Usuario: {input_text}"
        
        if memories:
            memory_context = " | ".join([m['content'] for m in memories[:3]])
            enriched += f" [Contexto: {memory_context}]"
        
        return enriched
    
    def _log_interaction(self, user_id: str, input_text: str, response: str, emotion: Dict):
        """Registra la interacción completa"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'input': input_text,
            'response': response,
            'emotion': emotion,
            'feedback': None
        }
        self.user_feedback[user_id].append(interaction)
        logger.info(f"Interacción registrada para {user_id}")
    
    def collect_user_feedback(self, user_id: str, feedback: str, rating: int):
        """Registra retroalimentación del usuario con calificación"""
        if user_id in self.user_feedback and self.user_feedback[user_id]:
            last_interaction = self.user_feedback[user_id][-1]
            last_interaction['feedback'] = feedback
            last_interaction['rating'] = rating
            logger.info(f"Feedback recibido de {user_id}: {rating}/5")
    
    def evolve_models(self, training_data: List[Tuple[str, str]] = None):
        """Evoluciona modelos usando retroalimentación"""
        if training_data:
            self.response_generator.train(training_data)
            logger.info("Generador de respuestas actualizado")
        
        self.memory_bank.save_memory(self.memory_file)
        logger.info("Evolución del sistema completada")

# Ejemplo de uso
if __name__ == "__main__":
    macro_rnn_system = MacroRNNSystem()
    
    # Entrenar componentes básicos (en producción se haría con datos reales)
    emotion_data = [
        ("Estoy muy feliz hoy!", "joy"),
        ("Me siento terrible", "sadness"),
        ("Esto es inaceptable!", "anger"),
        ("Tengo miedo de lo que viene", "fear"),
        ("No puedo creerlo!", "surprise"),
        ("El clima está nublado", "neutral")
    ]
    texts, emotions = zip(*emotion_data)
    macro_rnn_system.emotion_classifier.train(list(texts), list(emotions))
    
    # Entrenar generador de respuestas
    conversation_data = [
        ("Hola", "Hola! ¿Cómo estás hoy?"),
        ("¿Qué hora es?", "Son las 10:30 AM"),
        ("Cuéntame un chiste", "¿Por qué los pájaros no usan Facebook? ¡Porque ya tienen Twitter!")
    ]
    macro_rnn_system.response_generator.train(conversation_data)
    
    # Interacción de ejemplo
    user_id = "usuario_123"
    print("\n--- Interacción 1 ---")
    input_text = "Hola, ¿cómo estás?"
    response = macro_rnn_system.generate_response(input_text, user_id)
    print(f"Usuario: {input_text}")
    print(f"AI: {response}")
    
    print("\n--- Interacción 2 ---")
    input_text = "Estoy muy feliz porque es mi cumpleaños"
    response = macro_rnn_system.generate_response(input_text, user_id)
    print(f"Usuario: {input_text}")
    print(f"AI: {response}")
    
    # Registrar retroalimentación
    macro_rnn_system.collect_user_feedback(user_id, "La respuesta fue adecuada", 4)
    
    # Evolucionar sistema
    macro_rnn_system.evolve_models()
