import cv2
import cvzone
from ultralytics import YOLO
import math

class SecuritySystem:
    # Initialize the security system
    def __init__(self):
        self.model = YOLO("models/yolov8m.pt")
        self.video = cv2.VideoCapture(0)
        self.class_names = []
        with open("assets/coco.names", "r") as file:
            self.class_names = file.read().splitlines()

    # Run the security system
    def run(self):
        while True:
            ret, frame = self.video.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            results = self.model(frame, stream=True)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    name = self.class_names[int(box.cls[0])]

                    if name == 'person':
                        cvzone.cornerRect(frame, (x1, y1, w, h), 10, 3)
                        cvzone.putTextRect(frame, f'{name} {conf}', (x1, y1 - 10), 1, 1, (0, 0, 0))

            cv2.imshow("VISION SECURITY SYSTEM", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    security_system = SecuritySystem()
    security_system.run()