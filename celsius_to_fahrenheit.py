import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt

# Tensorflow error logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Creating numpy array for input feature vector and output labels (m = 7)
celsius_features = np.array([-40, -10,  0,  8, 15, 22,  38], dtype=float)
fahrenheit_labels = np.array([-40,  14, 32, 46, 59, 72, 100], dtype=float)

# Printing input/output values of dataset
for index in range(len(celsius_features)):
    print(f'Celsius: {celsius_features[index]} -> Fahrenheit: {fahrenheit_labels[index]}')

# Creating a layer with one neuron, with an input of a 1 x 1 matrix 
# (matches input vector dimension for one training example)
layer1 = tf.keras.layers.Dense(units=1, input_shape=[1])

# Initializing a sequential model with layer array (Only one layer in this case)
model = tf.keras.Sequential([layer1])

# Compiling the model, with loss function and optimization algorithm parameters + learning rate
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))

# Training the model, 1000 iterations for the same 7 training examples
history = model.fit(celsius_features, fahrenheit_labels, epochs=1000, verbose=False)

# Plot labels
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Creating plot
plt.plot(history.history['loss'])

# Displaying plot
plt.show()


print(model.predict([87.4]))