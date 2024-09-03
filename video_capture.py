import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame
    cv2.imshow('Live Feed', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
