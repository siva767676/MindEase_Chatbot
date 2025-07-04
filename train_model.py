import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Load the dataset
with open('mental_health_dataset.pickle', 'rb') as handle:
    data = pickle.load(handle)
    texts = data['texts']
    labels = data['labels']

# Convert labels to one-hot encoding
label_to_id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
id_to_label = {idx: label for label, idx in label_to_id.items()}
label_ids = np.array([label_to_id[label] for label in labels])
y = tf.keras.utils.to_categorical(label_ids, num_classes=len(label_to_id))

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(label_ids),
    y=label_ids
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=2000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)

# Create model with regularization
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(2000, 32, input_length=50),
    tf.keras.layers.SpatialDropout1D(0.2),  # Spatial dropout for embeddings
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(0.01))),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(label_to_id), activation='softmax')
])

# Compile with a lower learning rate
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Add callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weight_dict
)

# Save the model and tokenizer
model.save('mental_health_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle)

# Print training results
print("\nTraining Results:")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

# Save label mappings
with open('label_mappings.pickle', 'wb') as handle:
    pickle.dump({'label_to_id': label_to_id, 'id_to_label': id_to_label}, handle)

print("\nModel, tokenizer, and label mappings saved successfully!")