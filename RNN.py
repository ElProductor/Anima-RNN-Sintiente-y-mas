import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout, Concatenate, LayerNormalization, MultiHeadAttention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("anima.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ANIMA")

class EmotionType(Enum):
    NEUTRAL = 0
    JOY = 1
    SADNESS = 2
    ANGER = 3
    SURPRISE = 4

class ConversationContext:
    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id
        self.timestamp = datetime.now()
        self.history = []

class QuantumMemoryBank:
    def __init__(self, memory_file: str = "quantum_memory.pkl"):
        self.memory_file = memory_file
        self.memory_states = {}
        self._load_memory()

    def store_memory(self, memory_id: str, content: str, emotion: EmotionType, importance: float):
        self.memory_states[memory_id] = {
            'content': content,
            'emotion': emotion.name,
            'importance': importance,
            'timestamp': datetime.now().isoformat()
        }
        self._save_memory()
        logger.info(f"Memory stored: {memory_id[:8]}...")

    def retrieve_memory(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.memory_states:
            return []
        # Simple retrieval for now, can be improved with vector similarity
        # This is a placeholder for a more complex retrieval logic
        all_content = [mem['content'] for mem in self.memory_states.values()] 
        # A real implementation would use TF-IDF or other embeddings here
        # For now, we just return the last k memories
        return list(self.memory_states.values())[-top_k:]

    def _save_memory(self):
        try:
            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.memory_states, f)
        except Exception as e:
            logger.error(f"Error saving memory: {e}")

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    self.memory_states = pickle.load(f)
                logger.info(f"Memory loaded from {self.memory_file}")
            except Exception as e:
                logger.error(f"Error loading memory: {e}")

class QuantumANIMA:
    def __init__(self, tokenizer: Tokenizer, memory_bank: QuantumMemoryBank, model_path: str):
        self.tokenizer = tokenizer
        self.memory_bank = memory_bank
        self.model = self._load_model(model_path)
        
        # Initialize default values first
        self.max_input_len = 100
        self.max_target_len = 100
        self.vocab_size = self.tokenizer.num_words if self.tokenizer.num_words else 10000
        
        # Try to get actual values from model if it loaded successfully
        if self.model:
            try:
                if isinstance(self.model.input_shape, list):
                    # Model with multiple inputs
                    if len(self.model.input_shape) >= 2:
                        self.max_input_len = self.model.input_shape[0][1] if self.model.input_shape[0][1] else 100
                        self.max_target_len = self.model.input_shape[1][1] if self.model.input_shape[1][1] else 100
                else:
                    # Model with a single input
                    self.max_input_len = self.model.input_shape[1] if self.model.input_shape[1] else 100
                    # For single input models, assume same length for target
                    self.max_target_len = self.max_input_len
            except Exception as e:
                logger.warning(f"Could not determine model dimensions, using defaults: {e}")
                # Keep default values

    def _load_model(self, model_path: str) -> Optional[Model]:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return None
        try:
            # No custom objects needed for the simplified model
            model = load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading Keras model: {e}")
            return None

    def generate_response(self, user_input: str, context: ConversationContext) -> str:
        if not self.model:
            logger.warning("Model not loaded, returning fallback response")
            return "Lo siento, el modelo de IA no está disponible en este momento."

        try:
            # 1. Preprocess user input
            encoder_input_data = self._preprocess_input(user_input)

            # 2. Generate response token by token
            response_text = self._decode_sequence(encoder_input_data)

            # 3. Store interaction in memory
            memory_id = f"{context.user_id}_{context.timestamp.isoformat()}"
            self.memory_bank.store_memory(memory_id, user_input, EmotionType.NEUTRAL, 0.5)
            context.history.append((user_input, response_text))

            return response_text if response_text else "Lo siento, no pude generar una respuesta."
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Lo siento, ocurrió un error al procesar tu mensaje."

    def train_on_interaction(self, user_input: str, response: str, context: ConversationContext):
        """Trains the model on a new user interaction."""
        logger.info(f"Training on interaction: User input='{user_input}', Response='{response}'")

        if not self.model:
            logger.warning("Model not loaded, cannot train.")
            return

        try:
            # Preprocess user input (encoder input)
            encoder_input_data = self._preprocess_input(user_input)

            # Preprocess response (decoder input and target)
            # For decoder input, we need <start> token at the beginning
            # For decoder target, we need <end> token at the end
            target_seq_input = self.tokenizer.texts_to_sequences(["<start> " + response])
            target_seq_output = self.tokenizer.texts_to_sequences([response + " <end>"])

            padded_target_input = pad_sequences(target_seq_input, maxlen=self.max_target_len, padding='post')
            padded_target_output = pad_sequences(target_seq_output, maxlen=self.max_target_len, padding='post')

            # Ensure target_seq_output is one-hot encoded for categorical_crossentropy if needed
            # Assuming the model's last layer uses sparse_categorical_crossentropy, so integer labels are fine.
            # If using categorical_crossentropy, convert to one-hot:
            # padded_target_output_one_hot = tf.keras.utils.to_categorical(padded_target_output, num_classes=self.vocab_size)

            # Train the model on this single interaction
            # The model expects two inputs: encoder_input_data and decoder_input_data
            # And one output: decoder_target_data
            history = self.model.train_on_batch([encoder_input_data, padded_target_input], padded_target_output)
            logger.info(f"Training on interaction complete. Loss: {history[0]}, Accuracy: {history[1]}")

        except Exception as e:
            logger.error(f"Error during training on interaction: {e}")

    def _preprocess_input(self, text: str) -> np.ndarray:
        input_seq = self.tokenizer.texts_to_sequences([text])
        padded_seq = pad_sequences(input_seq, maxlen=self.max_input_len, padding='post')
        return padded_seq

    def _decode_sequence(self, input_seq: np.ndarray) -> str:
        """
        Decodes a sequence using the loaded model.
        This is a simplified version that works with the current model architecture.
        """
        try:
            # Ensure max_target_len is not None
            if self.max_target_len is None:
                self.max_target_len = 100
                logger.warning("max_target_len was None, setting to default value 100")
            
            # Get start and end token indices
            start_token_index = self.tokenizer.word_index.get('<start>', 1)
            end_token_index = self.tokenizer.word_index.get('<end>', 2)

            # Initialize decoder input sequence with start token
            target_seq = np.zeros((1, self.max_target_len))
            target_seq[0, 0] = start_token_index

            decoded_sentence = ''
            
            for i in range(1, self.max_target_len):
                try:
                    # Predict next token
                    # The model expects two inputs: encoder_input_data and decoder_input_data
                    output_tokens = self.model.predict([input_seq, target_seq], verbose=0)

                    # Get the token with highest probability
                    sampled_token_index = np.argmax(output_tokens[0, i-1, :])

                    # Convert index to word
                    sampled_word = self.tokenizer.index_word.get(sampled_token_index, '')

                    # Stop if end token is reached
                    if sampled_word == '<end>' or sampled_token_index == end_token_index:
                        break

                    # Add word to decoded sentence
                    if sampled_word:  # Only add non-empty words
                        decoded_sentence += ' ' + sampled_word

                    # Update decoder input sequence for next prediction
                    target_seq[0, i] = sampled_token_index

                except Exception as e:
                    logger.error(f"Error in decoding step {i}: {e}")
                    break

            return decoded_sentence.strip() if decoded_sentence else "Lo siento, no pude generar una respuesta."
            
        except Exception as e:
            logger.error(f"Error in _decode_sequence: {e}")
            return "Lo siento, ocurrió un error durante la generación de la respuesta."