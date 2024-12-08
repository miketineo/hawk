import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import json
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve RTSP credentials and host from environment variables
RTSP_USERNAME = os.getenv("RTSP_USERNAME")
RTSP_PASSWORD = os.getenv("RTSP_PASSWORD")
RTSP_HOST = os.getenv("RTSP_HOST")
RTSP_PORT = os.getenv("RTSP_PORT", "554")  # Default RTSP port if not specified

# Construct the RTSP URL
RTSP_URL = f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@{RTSP_HOST}:{RTSP_PORT}/stream"  # Adjust the stream path if necessary

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the YOLOv8 model once at startup
model = YOLO('yolov8x.pt').to('cuda')  # Ensure you have a compatible GPU

# Serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def get():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

# Existing /video_feed WebSocket endpoint
@app.websocket("/video_feed")
async def video_feed(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            frame_data = await websocket.receive_bytes()
            # Process frame_data as per existing implementation
            # ...
    except WebSocketDisconnect:
        print("Client disconnected from /video_feed.")
    except Exception as e:
        print(f"Error in /video_feed: {e}")
    finally:
        await websocket.close()

# New WebSocket endpoint for RTSP stream
@app.websocket("/cam1")
async def cam1_feed(websocket: WebSocket):
    await websocket.accept()

    # Open RTSP stream using OpenCV
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        await websocket.send_text(json.dumps({"error": "Unable to open RTSP stream."}))
        await websocket.close()
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await websocket.send_text(json.dumps({"error": "Failed to read frame from RTSP stream."}))
                break

            # Perform YOLO inference
            results = model(frame, verbose=False)
            annotated_frame = frame.copy()

            for result in results:
                for box in result.boxes:
                    # Extract box coordinates and label
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    label = result.names[cls_id]

                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, label, (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue

            # Convert to bytes
            frame_bytes = buffer.tobytes()

            # Send as binary WebSocket message
            await websocket.send_bytes(frame_bytes)

            # Control frame rate (e.g., 10 FPS)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print("Client disconnected from /cam1.")
    except Exception as e:
        print(f"Error in /cam1: {e}")
    finally:
        cap.release()
        await websocket.close()

