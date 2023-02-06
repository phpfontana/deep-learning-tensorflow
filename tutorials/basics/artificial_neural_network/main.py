# importing libraries
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Verifying TensorFlow version
print(f'TensorFlow version: {tf.__version__}')
print(f"Keras version: {tf.keras.__version__}")

# Device configuration
print(f"\nVerifying GPU Access:")
device = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU is", "available" if device else "NOT AVAILABLE")

# Hyper-parameters
learning_rate = 0.001
num_epochs = 10
batch_size = 128
num_classes = 10

# MNIST dataset
mnist = tf.keras.datasets.mnist

# Train/test split
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape
x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_test = x_test.reshape(x_test.shape[0], 28 * 28)

# Normalizing inputs
x_train, x_test = x_train / 255.0, x_test / 255.0

# Clearing backend
tf.keras.backend.clear_session()

# Setting random seed
tf.random.set_seed(42)

# Creating model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)))

model.add(tf.keras.layers.Dense(64, activation='relu'))

model.add(tf.keras.layers.Dense(32, activation='relu'))

model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compiling model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=['accuracy'])

model.build()

# Model summary
model.summary()

# Training step
history = model.fit(x_train, y_train,
                    validation_split=0.2,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=1)

# Evaluation step
print()
model.evaluate(x_test, y_test)

# Loss curve
pd.DataFrame(history.history).plot()
