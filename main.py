import cv2
from ultralytics import YOLO
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Path to the trained model
MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "detect"
    / "curvilinear_shape_detector-4"
    / "weights"
    / "best.pt"
)

# Check if model exists
if not MODEL_PATH.exists():
    print("Model not found.")
    print("Train the model first using:")
    print("python train.py")
    print(f"Expected model path: {MODEL_PATH}")
    exit()

# Load model
model = YOLO(str(MODEL_PATH))

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found.")
    print("Try changing VideoCapture(0) to VideoCapture(1).")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame from camera.")
        break

    # Run detection
    results = model(frame, conf=0.5)

    # Draw detections
    annotated_frame = results[0].plot()

    # Show image
    cv2.imshow("Curvilinear 3D Shape Detection", annotated_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()