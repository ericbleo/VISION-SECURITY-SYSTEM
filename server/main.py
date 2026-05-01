import cv2
import cvzone
from ultralytics import YOLO
import math
import os
from dotenv import load_dotenv
import resend
import time

load_dotenv()

class SecuritySystem:
    # Initialize the security system
    def __init__(self):
        self.model = YOLO("models/yolov8n.pt")
        self.video = cv2.VideoCapture(0)
        self.class_names = []
        with open("assets/coco.names", "r") as file:
            self.class_names = file.read().splitlines()
        resend.api_key = os.getenv("RESEND_API_KEY")
        self.recipient_emails = os.getenv("RECIPIENT_EMAILS").split(",")
        self.last_email_time = 0

    # Send email notification
    def send_email_notification(self):
        for email in self.recipient_emails:
            try:
                resend.Emails.send({
                    "from": "Eric Security System <onboarding@resend.dev>",
                    "to": email,
                    "subject": "MOVEMENT DETECTED 🚨",
                    "html": "<strong>A person was detected by your security system.</strong>"
                })
                print(f"Email sent to {email}")
            except Exception as e:
                print(f"Error sending email to {email}: {e}")

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

                        current_time = time.time()
                        # 60 seconds cooldown between emails
                        if current_time - self.last_email_time > 60:
                            self.send_email_notification()
                            self.last_email_time = current_time

            cv2.imshow("VISION SECURITY SYSTEM", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    security_system = SecuritySystem()
    security_system.run()