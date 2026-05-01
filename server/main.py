import cv2
import cvzone
from ultralytics import YOLO
import math

class SecuritySystem:
    # Initialize the security system
    def __init__(self):
        self.model = YOLO("models/yolov8n.pt")
        self.video = cv2.VideoCapture(0)
        self.class_names = []
        with open("assets/coco.names", "r") as file:
            self.class_names = file.read().splitlines()

    # Run the security system
    def run(self):
        while True:
            ret, frame = self.video.read()
            frame = cv2.flip(frame, 1)
            results = self.model(frame, stream=True)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    w = x2 - x1
                    h = y2 - y1
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    name = self.class_names[cls]

                    if name == 'person':
                        cvzone.cornerRect(frame, (x1, y1, w, h), 10, 3)
                        cvzone.putTextRect(frame, f'{name} {conf}', (x1, y1 - 10), 1, 1, (0, 0, 0))

                cv2.imshow("VISION SECURITY SYSTEM", frame)


if __name__ == "__main__":
    security_system = SecuritySystem()
    security_system.run()