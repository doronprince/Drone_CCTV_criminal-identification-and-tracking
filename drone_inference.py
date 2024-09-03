from djitellopy import Tello
import cv2
import tensorflow as tf
import numpy as np
import face_recognition_models

# Load the trained model
model = tf.keras.models.load_model('target_model.h5')

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (64, 64))  # Resize to model input size
    normalized_frame = resized_frame / 255.0  # Normalize pixel values
    return normalized_frame.reshape(1, 64, 64, 3)  # Add batch dimension

# Initialize Tello drone
tello = Tello()
tello.connect()
tello.streamon()

# Load a sample picture and learn how to recognize it
known_image = face_recognition_models.load_image_file("known_face.jpg")
known_face_encoding = face_recognition_models.face_encodings(known_image)[0]

known_face_encodings = [known_face_encoding]
known_face_names = ["Target"]

# Get the drone's camera stream
frame_read = tello.get_frame_read()

while True:
    # Get the current frame
    frame = frame_read.frame

    # Convert the frame from BGR (OpenCV format) to RGB (face_recognition format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess the frame for the model
    preprocessed_frame = preprocess_frame(frame)
    prediction = model.predict(preprocessed_frame)

    # Face recognition process
    face_locations = face_recognition_models.face_locations(rgb_frame)
    face_encodings = face_recognition_models.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition_models.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            # Control the drone based on face location
            top, right, bottom, left = face_location
            face_center_x = (left + right) // 2
            face_center_y = (top + bottom) // 2

            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2

            move_x = face_center_x - frame_center_x
            move_y = frame_center_y - face_center_y

            # Thresholds to determine if the drone should move
            threshold_x = 30
            threshold_y = 30

            if abs(move_x) > threshold_x:
                if move_x > 0:
                    tello.move_right(20)  # Adjust the speed as needed
                else:
                    tello.move_left(20)

            if abs(move_y) > threshold_y:
                if move_y > 0:
                    tello.move_down(20)
                else:
                    tello.move_up(20)

            # Optional: Move forward/backward based on face size (distance)
            face_height = bottom - top
            target_height = 100  # Example target face height for a specific distance

            if face_height < target_height - 20:
                tello.move_forward(20)
            elif face_height > target_height + 20:
                tello.move_back(20)

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # If no face is recognized, use the model-based tracking
    if not face_locations and prediction > 0.5:
        tello.move_forward(20)  # Example: Move the drone forward
        # You can add more complex movements based on your needs

    # Display the frame
    cv2.imshow("Drone Feed", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.streamoff()
tello.end()
cv2.destroyAllWindows()
