import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load and preprocess your dataset
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    r'"C:\Users\doron\Documents\Documents\collage\2nd YEAR\SEM 3\DP-1\Criminal-Face-Identification-System-Source\Criminal-Face-Identification-System-Source\img"',  # Directory with training images
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    r'"C:\Users\doron\Documents\Documents\collage\2nd YEAR\SEM 3\DP-1\Criminal-Face-Identification-System-Source\Criminal-Face-Identification-System-Source\img"',  # Directory with validation images
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Save the trained model
model.save('target_model.h5')

