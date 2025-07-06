import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Generar datos de ejemplo con patrones significativos
def generate_data(num_samples, time_steps):
    """Genera datos sintéticos con patrones temporales reconocibles"""
    # Crear secuencias con tendencia y estacionalidad
    t = np.linspace(0, 20, time_steps)
    
    X = np.zeros((num_samples, time_steps, 1))
    y = np.zeros((num_samples, 1))
    
    for i in range(num_samples):
        # Componentes de la secuencia
        trend = 0.5 * t
        seasonal = 2 * np.sin(0.5 * t + np.random.uniform(0, 2*np.pi))
        noise = 0.2 * np.random.randn(time_steps)
        
        # Combinar componentes
        sequence = trend + seasonal + noise
        X[i, :, 0] = sequence
        
        # La etiqueta es el valor promedio de la secuencia
        y[i] = np.mean(sequence)
    
    return X.astype(np.float32), y.astype(np.float32)

# Implementación corregida del mecanismo de atención
class AttentionLayer(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        # Capas para calcular los pesos de atención
        self.W = self.add_weight(name='att_weight', 
                                shape=(input_shape[-1], self.units),
                                initializer='glorot_uniform',
                                trainable=True)
        self.U = self.add_weight(name='att_energy', 
                                shape=(self.units, 1),
                                initializer='glorot_uniform',
                                trainable=True)
        self.b = self.add_weight(name='att_bias', 
                                shape=(self.units,),
                                initializer='zeros',
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        
        # Calcular puntuaciones de atención
        # e = tanh(W * h + b)
        e = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        
        # Calcular pesos de atención
        # a = softmax(e * U)
        scores = tf.matmul(e, self.U)
        attention_weights = tf.nn.softmax(scores, axis=1)
        
        # Calcular vector de contexto
        # c = sum(a * h)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        
        return context_vector

    def get_config(self):
        config = super().get_config().copy()
        config.update({'units': self.units})
        return config

# Crear el modelo RNN avanzado usando API funcional
def create_advanced_real_time_rnn(input_shape):
    """Crea un modelo RNN con atención para análisis de secuencias temporales"""
    inputs = keras.Input(shape=input_shape)
    
    # Capa LSTM con retorno de secuencia completa
    x = layers.LSTM(128, return_sequences=True)(inputs)
    
    # Capa de atención
    context_vector = AttentionLayer(units=64)(x)
    
    # Capas adicionales
    x = layers.Dense(64, activation='relu')(context_vector)
    x = layers.Dropout(0.3)(x)  # Regularización
    outputs = layers.Dense(1)(x)  # Salida
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compilar con optimizador Adam y tasa de aprendizaje ajustable
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# Parámetros
num_samples = 2000  # Más muestras para mejor generalización
time_steps = 20     # Secuencias más largas

# Generar datos de entrenamiento y validación
X, y = generate_data(num_samples, time_steps)

# Dividir en conjunto de entrenamiento y prueba
split_idx = int(0.8 * num_samples)
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# Crear el modelo
model = create_advanced_real_time_rnn((time_steps, 1))
model.summary()

# Callbacks para mejorar el entrenamiento
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
]

# Entrenar el modelo
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Evaluar el modelo con datos de prueba
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# Guardar el modelo con objetos personalizados
model.save('advanced_real_time_rnn.h5', save_format='h5')

# Visualización del rendimiento
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_history(history)

# Función para probar el modelo con datos nuevos
def test_model(model, time_steps):
    """Genera y prueba una secuencia nueva"""
    # Crear secuencia de prueba
    t = np.linspace(0, 20, time_steps)
    sequence = 0.5 * t + 2 * np.sin(0.5 * t) + 0.2 * np.random.randn(time_steps)
    sequence = sequence.reshape(1, time_steps, 1)
    
    # Predecir valor
    prediction = model.predict(sequence, verbose=0)
    true_value = np.mean(sequence)
    
    print(f"\nSecuencia de prueba: {sequence[0,:,0]}")
    print(f"Predicción: {prediction[0][0]:.4f}, Valor real: {true_value:.4f}")
    print(f"Error: {abs(prediction[0][0] - true_value):.4f}")

# Probar el modelo
test_model(model, time_steps)