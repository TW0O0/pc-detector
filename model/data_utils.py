import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(train_dir, val_dir, img_size=(224, 224), batch_size=32):
    """
    Create data generators with augmentation for training and validation
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
        channel_shift_range=0.1  # Subtle color variation
    )
    
    # Validation data generator (only rescaling)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create the generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',  # Change to 'categorical' for multi-class
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',  # Change to 'categorical' for multi-class
        shuffle=False
    )
    
    return train_generator, val_generator

def visualize_augmented_images(generator, num_images=5):
    """
    Visualize samples of augmented images
    """
    # Get a batch of images
    images, labels = next(generator)
    
    plt.figure(figsize=(15, 5))
    for i in range(min(num_images, len(images))):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.title(f"Class: {int(labels[i])}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmented_samples.png')  # Save the visualization
    plt.show()
    
    return 'augmented_samples.png'  # Return the filename for reference
