import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# --- 1. Datos de Conversación de Ejemplo ---
# Un corpus más grande y variado es crucial para un buen rendimiento.
conversations = [
    ("hola", "hola, ¿cómo estás?"),
    ("cómo estás", "estoy bien, gracias por preguntar. ¿y tú?"),
    ("estoy bien", "me alegra oír eso."),
    ("qué puedes hacer", "puedo responder tus preguntas y aprender de nuestras conversaciones."),
    ("adiós", "hasta luego, que tengas un buen día."),
    ("quién te creó", "fui creado por un desarrollador que explora la inteligencia artificial."),
    ("cuál es tu propósito", "mi propósito es asistir y conversar."),
    ("gracias", "de nada.")
]

input_texts, target_texts = zip(*conversations)

# --- 2. Preprocesamiento y Tokenización ---

# Añadir tokens de inicio/fin a los textos de destino
target_texts_processed = [f"<start> {text} <end>" for text in target_texts]

# Crear y ajustar el tokenizador
# Usamos un solo tokenizador para ambos (input y target) por simplicidad
tokenizer = Tokenizer(filters='', oov_token='<unk>')
tokenizer.fit_on_texts(list(input_texts) + target_texts_processed)

# Guardar el tokenizador para usarlo en el servidor
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("Tokenizador guardado en 'tokenizer.pkl'")

VOCAB_SIZE = len(tokenizer.word_index) + 1
EMBEDDING_DIM = 64
LSTM_UNITS = 128

# Convertir texto a secuencias de enteros
input_sequences = tokenizer.texts_to_sequences(input_texts)
_target_sequences = tokenizer.texts_to_sequences(target_texts_processed)

# Rellenar secuencias para que tengan la misma longitud
max_input_len = max(len(seq) for seq in input_sequences)
max_target_len = max(len(seq) for seq in _target_sequences)

encoder_input_data = pad_sequences(input_sequences, maxlen=max_input_len, padding='post')

# Preparar datos para el decodificador (teacher forcing)
decoder_input_data = pad_sequences([seq[:-1] for seq in _target_sequences], maxlen=max_target_len, padding='post')
decoder_target_sequences = pad_sequences([seq[1:] for seq in _target_sequences], maxlen=max_target_len, padding='post')

decoder_target_data = np.zeros((len(input_texts), max_target_len, VOCAB_SIZE), dtype='float32')

for i, seq in enumerate(decoder_target_sequences):
    for t, word_idx in enumerate(seq):
        if word_idx > 0:
            decoder_target_data[i, t, word_idx] = 1.0

# --- 3. Construcción del Modelo Seq2Seq con LSTM ---

# --- Codificador ---
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(encoder_inputs)
encoder_lstm = LSTM(LSTM_UNITS, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# --- Decodificador ---
decoder_inputs = Input(shape=(None,))
decoder_embedding_layer = Embedding(VOCAB_SIZE, EMBEDDING_DIM)
decoder_embedding = decoder_embedding_layer(decoder_inputs)
decoder_lstm = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(VOCAB_SIZE, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# --- Modelo Completo ---
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 4. Entrenamiento ---

# Necesitamos dar forma a los datos del decodificador para que coincidan con la salida
# Aquí hay una simplificación: el modelo predice el siguiente token, por lo que la entrada del decodificador
# y el objetivo deben estar alineados correctamente.

# Para este ejemplo simple, vamos a crear un modelo que no use teacher forcing en el entrenamiento
# para simplificar la preparación de datos. Esto es menos efectivo pero más fácil de implementar.

# Re-simplificando el objetivo:
# El objetivo es predecir la secuencia completa, no token por token con teacher forcing.
# Esto requiere una arquitectura ligeramente diferente o una preparación de datos más compleja.

# Por el bien de generar un archivo de modelo funcional, vamos a entrenar con los datos preparados,
# aunque la decodificación en `server.py` tendrá que ser implementada para generar respuestas
# token por token.

print("\nEntrenando el modelo...")
# El modelo espera que decoder_input_data sea una secuencia de entrada para el decodificador.
# Vamos a crearla a partir de las secuencias objetivo, desplazadas una posición.
decoder_input_data_training = np.zeros((len(input_texts), max_target_len), dtype='int32')
decoder_target_data_training = np.zeros((len(input_texts), max_target_len, VOCAB_SIZE), dtype='float32')

for i, target_sequence in enumerate(_target_sequences):
    for t, word_index in enumerate(target_sequence):
        decoder_input_data_training[i, t] = word_index
        if t > 0:
            decoder_target_data_training[i, t - 1, word_index] = 1.0

model.fit([encoder_input_data, decoder_input_data_training], decoder_target_data_training,
          batch_size=2,
          epochs=150,
          validation_split=0.2)

# --- 5. Guardar el Modelo ---
model.save('chatbot_state_model.h5')
print("\nModelo guardado en 'chatbot_state_model.h5'")
