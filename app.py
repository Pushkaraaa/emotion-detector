import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import time

st.title("Real-time Emotion Detection")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a placeholder for the webcam feed
video_placeholder = st.empty()

# Create a placeholder for the emotion results
result_placeholder = st.text("")

# Add a stop button
stop_button = st.button("Stop")

while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access webcam")
        break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    try:
        # Analyze emotions
        result = DeepFace.analyze(
            frame_rgb,
            actions=['emotion'],
            enforce_detection=False
        )
        
        # Get dominant emotion and format the results
        emotion = result[0]['dominant_emotion']
        emotion_scores = result[0]['emotion']
        
        # Create a formatted string with all emotion scores
        emotion_text = f"""
        Dominant Emotion: {emotion.upper()}
        
        All Emotions:
        - Angry: {emotion_scores['angry']:.2f}%
        - Disgust: {emotion_scores['disgust']:.2f}%
        - Fear: {emotion_scores['fear']:.2f}%
        - Happy: {emotion_scores['happy']:.2f}%
        - Sad: {emotion_scores['sad']:.2f}%
        - Surprise: {emotion_scores['surprise']:.2f}%
        - Neutral: {emotion_scores['neutral']:.2f}%
        """
        
        # Update result text
        result_placeholder.text(emotion_text)
        print(f"Dominant Emotion: {emotion}")
        
    except Exception as e:
        pass  # Skip frame if face not detected
    
    # Display the frame
    video_placeholder.image(frame_rgb)
    
    # Add a small delay to reduce CPU usage
    time.sleep(0.5)

# Release resources
cap.release() 