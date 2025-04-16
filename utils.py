import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

def process_image(image_path, model_path, hands):
    # Load YOLO model
    model = YOLO(model_path)
    
    # Read image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe Hands
    results_hands = hands.process(image_rgb)
    
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Draw hand landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
    
    # Process with YOLO
    results_yolo = model(image)
    
    # Parse results
    prediction = None
    for result in results_yolo:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Get class name from model
            class_name = model.names[cls]
            label = f"{class_name}: {conf:.2f}"
            
            # Put label
            cv2.putText(image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            prediction = class_name
    
    return image, prediction

def process_video(video_path, model_path, hands):
    # Load YOLO model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Prepare output video
    output_path = video_path.replace('.', '_processed.')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Hands
        results_hands = hands.process(frame_rgb)
        
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Draw hand landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        
        # Process with YOLO
        results_yolo = model(frame)
        
        # Draw results
        for result in results_yolo:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Get class name from model
                class_name = model.names[cls]
                label = f"{class_name}: {conf:.2f}"
                
                # Put label
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Write frame to output
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    
    return output_path

def process_camera(model_path, hands):
    # Load YOLO model
    model = YOLO(model_path)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    # Prepare to save a frame
    saved_frame = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Hands
        results_hands = hands.process(frame_rgb)
        
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Draw hand landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        
        # Process with YOLO
        results_yolo = model(frame)
        
        # Draw results
        for result in results_yolo:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Get class name from model
                class_name = model.names[cls]
                label = f"{class_name}: {conf:.2f}"
                
                # Put label
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Sign Language Detection', frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            saved_frame = frame
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    if saved_frame is not None:
        # Save the captured frame
        output_path = 'static/uploads/capture.jpg'
        cv2.imwrite(output_path, saved_frame)
        return output_path
    
    return None