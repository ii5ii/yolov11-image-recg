import cv2
import os
from ultralytics import YOLO

# Load the YOLOv11 model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
print("Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")

# Open webcam (index 0 or 1)
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Perform real-time object detection
        results = model.track(frame, persist=True)  # Use `.track()` for continuous tracking

        # Extract detections and overlay results on frame
        result_frame = results[0].plot()

        # Show the output live
        cv2.imshow("YOLOv8 Live Detection", result_frame)

        # Ensure continuous frame updates (Press 'q' to exit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released. Program exited.")
