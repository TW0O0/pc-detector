import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3, ResNet50V2, DenseNet121
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt
import numpy as np
from data_utils import create_data_generators, visualize_augmented_images

# Set paths
data_dir = '../data/processed'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
model_save_dir = './saved_models'
os.makedirs(model_save_dir, exist_ok=True)

# Parameters
img_size = (224, 224)
batch_size = 32
epochs_initial = 10
epochs_fine_tuning = 20
model_name = 'efficientnet_pachyonychia'  # Base name for saving models

# Create data generators
train_generator, val_generator = create_data_generators(
    train_dir, val_dir, img_size, batch_size
)

# Visualize some augmented training samples
visualize_augmented_images(train_generator)

# Print dataset information
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Class indices: {train_generator.class_indices}")

# Create model using transfer learning
def create_model(base_model_type='efficientnet'):
    if base_model_type == 'efficientnet':
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=(*img_size, 3)
        )
    elif base_model_type == 'resnet':
        base_model = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=(*img_size, 3)
        )
    elif base_model_type == 'densenet':
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(*img_size, 3)
        )
    else:
        raise ValueError("Unsupported model type")
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create the model architecture
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.AUC(name='auc'), 
                tf.keras.metrics.Precision(name='precision'), 
                tf.keras.metrics.Recall(name='recall')]
    )
    
    return model, base_model

# Create the model
model, base_model = create_model()
model.summary()

# Define callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_save_dir, f"{model_name}_initial_best.h5"),
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    )
]

# Initial training phase
print("Starting initial training phase...")
history_initial = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs_initial,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    callbacks=callbacks
)

# Save the initial model
model.save(os.path.join(model_save_dir, f"{model_name}_initial.h5"))

# Fine-tuning phase
print("Starting fine-tuning phase...")
# Unfreeze some layers of the base model
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Recompile with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', 
            tf.keras.metrics.AUC(name='auc'), 
            tf.keras.metrics.Precision(name='precision'), 
            tf.keras.metrics.Recall(name='recall')]
)

# Update the checkpoint filepath for fine-tuning
callbacks[0].filepath = os.path.join(model_save_dir, f"{model_name}_fine_tuned_best.h5")

# Continue training with fine-tuning
history_fine_tuning = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs_fine_tuning,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    callbacks=callbacks
)

# Save the final model
model.save(os.path.join(model_save_dir, f"{model_name}_final.h5"))

# Plot training history
def plot_training_history(history_initial, history_fine_tuning=None):
    # Combine histories if fine-tuning was performed
    if history_fine_tuning:
        # Combine the history objects
        total_epochs = len(history_initial.history['accuracy']) + len(history_fine_tuning.history['accuracy'])
        epochs_range = range(1, total_epochs + 1)
        
        # Combine metrics
        acc = history_initial.history['accuracy'] + history_fine_tuning.history['accuracy']
        val_acc = history_initial.history['val_accuracy'] + history_fine_tuning.history['val_accuracy']
        loss = history_initial.history['loss'] + history_fine_tuning.history['loss']
        val_loss = history_initial.history['val_loss'] + history_fine_tuning.history['val_loss']
        auc = history_initial.history['auc'] + history_fine_tuning.history['auc']
        val_auc = history_initial.history['val_auc'] + history_fine_tuning.history['val_auc']
        
        # Mark the transition point
        transition_epoch = len(history_initial.history['accuracy'])
    else:
        epochs_range = range(1, len(history_initial.history['accuracy']) + 1)
        acc = history_initial.history['accuracy']
        val_acc = history_initial.history['val_accuracy']
        loss = history_initial.history['loss']
        val_loss = history_initial.history['val_loss']
        auc = history_initial.history['auc']
        val_auc = history_initial.history['val_auc']
        transition_epoch = None
    
    # Create figure with two subplots
    plt.figure(figsize=(15, 6))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, acc, label='Training')
    plt.plot(epochs_range, val_acc, label='Validation')
    if transition_epoch:
        plt.axvline(x=transition_epoch, color='r', linestyle='--', label='Start Fine-tuning')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, loss, label='Training')
    plt.plot(epochs_range, val_loss, label='Validation')
    if transition_epoch:
        plt.axvline(x=transition_epoch, color='r', linestyle='--', label='Start Fine-tuning')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot AUC
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, auc, label='Training')
    plt.plot(epochs_range, val_auc, label='Validation')
    if transition_epoch:
        plt.axvline(x=transition_epoch, color='r', linestyle='--', label='Start Fine-tuning')
    plt.title('AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, f'{model_name}_training_history.png'))
    plt.show()

# Plot the training history
plot_training_history(history_initial, history_fine_tuning)

print(f"Training complete. Models saved in {model_save_dir}")
