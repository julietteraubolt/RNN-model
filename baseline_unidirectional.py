# Assignment 2 RNN - ICS661 Juliette Raubolt
# RNN model for sentiment analysis key features:
# - Text preprocessing with TextVectorization
# - Embedding layer
# - LSTM layer (unidirectional)
# - Dense layer with ReLU activation and 64 nodes
# - Output layer with sigmoid activation
# - Binary crossentropy loss function
# - Adam optimizer
# - Accuracy, precision, recall, and F1 score metrics
# - Epochs = 10

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

# Loading data with text_dataset_from_directory
# There is a training and testing directory and each
# has a subdirectory for pos and neg reviews which is 
# why text_dataset_from_directory is used
train_dir = 'train'
test_dir = 'test'

# Training data
train_dataset = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=32,
    validation_split=0.2,
    subset='training',
    seed=1337, 
    label_mode='binary')

# Validation data
validation_dataset = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=32,
    validation_split=0.2,
    subset='validation',
    seed=1337, 
    label_mode='binary')

# Testing data
test_dataset = tf.keras.preprocessing.text_dataset_from_directory(
    test_dir,
    batch_size=32,
    label_mode='binary')

# Text preprocessing 
max_features = 10000    # Determines the vocab size (only 10,000 most frequent words used)
sequence_length = 250   # Determines the length of the reviews

# Vectorization layer to tokenize, map, limit, and pad the data 
# to be used in the model
vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (no labels) and learn the vocabulary
train_text = train_dataset.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# Define a function to vectorize the datasets
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# Vectorize the data
train_dataset = train_dataset.map(vectorize_text)
validation_dataset = validation_dataset.map(vectorize_text)
test_dataset = test_dataset.map(vectorize_text)

# Create the model
model = tf.keras.Sequential([
    layers.Embedding(input_dim=max_features,output_dim=128, input_length=sequence_length), # Embedding layer  
    layers.LSTM(128),                                                                      # LSTM layer 
    layers.Dense(64, activation='relu'),                                                   # Dense layer  
    layers.Dense(1, activation='sigmoid')                                                  # Output layer  
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=10)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_dataset)

# Calculate precision and recall by predicting the test dataset
# and comparing the predictions to the true labels
y_pred_prob = model.predict(test_dataset)
y_pred = np.where(y_pred_prob >= 0.5, 1, 0)  
y_true = np.concatenate([y for x, y in test_dataset], axis=0)
test_precision = precision_score(y_true, y_pred)
test_recall = recall_score(y_true, y_pred)

# Calculate F1 Score based on precision and recall
if(test_precision + test_recall == 0):
    f1_score = 0
else:
    f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)

# Print the results
print(f'Test loss: {test_loss}')
print('Test Accuracy:', test_acc)
print(f'Precision: {test_precision}')
print(f'Recall: {test_recall}')
print(f'F1 Score: {f1_score}')

# Plot the training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
