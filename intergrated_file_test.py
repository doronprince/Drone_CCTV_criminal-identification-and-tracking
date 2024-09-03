import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from djitellopy import Tello

# Function to capture video and save it for training
def capture_training_video(output_dir='data/train'):
    cap = cv2.VideoCapture(0)  # 0 is the default camera
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame for training
        cv2.imwrite(f'{output_dir}/frame_{count}.jpg', frame)
        count += 1

        # Display the frame
        cv2.imshow('Live Feed', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to train the CNN model
def train_model():
    # Load and preprocess your dataset
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        'data/train',  # Directory with training images
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        'data/validation',  # Directory with validation images
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
    return model

# Function to preprocess frame for inference
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (64, 64))  # Resize to model input size
    normalized_frame = resized_frame / 255.0  # Normalize pixel values
    return normalized_frame.reshape(1, 64, 64, 3)  # Add batch dimension

# Function to control the drone based on AI predictions
def control_drone(model):
    # Initialize Tello drone
    tello = Tello()
    tello.connect()
    tello.streamon()

    while True:
        # Get the frame from the drone's camera
        frame = tello.get_frame_read().frame

        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)
        prediction = model.predict(preprocessed_frame)

        # If target is detected
        if prediction > 0.5:
            tello.move_forward(20)  # Example: Move the drone forward
            # You can add more complex movements based on your needs

        # Display the frame
        cv2.imshow('Drone Feed', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tello.streamoff()
    tello.end()

# Main function to orchestrate the workflow
def main():
    # Step 1: Capture video for training
    print("Capturing training data...")
    capture_training_video()

    # Step 2: Train the model
    print("Training model...")
    model = train_model()

    # Step 3: Control the drone using the trained model
    print("Starting drone control...")
    control_drone(model)

if __name__ == "__main__":
    main()
