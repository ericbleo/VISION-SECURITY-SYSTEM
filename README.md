# Vision Security System

A Python program that watches your webcam and sends you an email with a photo whenever a person is detected.

---

## How it works

1. Your webcam feed is analyzed in real time using an AI model
2. When a person is detected, a snapshot is taken and emailed to you
3. Emails are sent at most once every 60 seconds so your inbox doesn't get flooded

---

## Requirements

- Python 3.10 or newer
- A webcam connected to your computer
- A free [Resend](https://resend.com) account (for sending emails)

---

## Setup

### 1. Install dependencies

```bash
pip install opencv-python cvzone ultralytics python-dotenv resend
```

### 2. Download the AI model

Download `yolov8n.pt` from [Ultralytics](https://github.com/ultralytics/assets/releases) and place it inside the `app/models/` folder:

```
app/
  models/
    yolov8n.pt
```

### 3. Create your `.env` file

Inside the `app/` folder, create a file named `.env` and add the following:

```
RESEND_API_KEY=your_resend_api_key_here
RECIPIENT_EMAILS=your@email.com
```

- Get your API key from [resend.com/api-keys](https://resend.com/api-keys)
- To send to multiple emails, separate them with commas: `email1@x.com,email2@x.com`

> **Note:** Resend's free plan only sends to the email address registered on your account. To send to any address, you need to verify a domain on Resend.

### 4. Run the program

```bash
cd app
python main.py
```

A window will open showing your webcam feed with detections drawn on it.
Press **Q** to quit.

---

## Project structure

```
app/
  main.py          ← main program
  .env             ← your API keys (never share this)
  assets/
    coco.names     ← list of object class names
  models/
    yolov8n.pt     ← AI model file
```
