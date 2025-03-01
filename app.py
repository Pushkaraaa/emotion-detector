import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import time
from collections import deque
from scipy.signal import find_peaks
import pandas as pd
import altair as alt

# At the start of your script, add these UI configurations
st.set_page_config(layout="wide", page_title="Emotion & Movement Analysis")

# Add custom CSS with darker backgrounds for metrics
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #1E1E1E;
        border-radius: 5px;
        padding: 15px 20px;
        margin: 10px 0;
    }
    
    div[data-testid="metric-container"] > div[data-testid="metric-value"] {
        color: #FFFFFF;
    }
    
    div[data-testid="metric-container"] > div[data-testid="metric-label"] {
        color: #CCCCCC;
    }
    
    div[data-testid="metric-container"] > div[data-testid="metric-delta"] {
        color: #00FF00;
    }
    
    div[data-testid="stHorizontalBlock"] {
        background-color: #2D2D2D;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Create placeholder containers ONCE at the start
col1, col2 = st.columns([2, 1])  # 2:1 ratio for video:results

with col1:
    st.title("Real-time Analysis")
    video_placeholder = st.empty()
    stop_button = st.button("Stop Recording")

with col2:
    st.title("Analysis Results")
    
    # Create empty containers for metrics
    st.markdown("### Movement Analysis")
    metrics_container = st.empty()
    st.markdown("### Emotion Analysis")
    emotion_chart_container = st.empty()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize variables for jitter detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
point_history = deque(maxlen=20)  # Increased to store last 20 positions
time_history = deque(maxlen=20)   # Store timestamps for each position
jitter_threshold = 15

def detect_restlessness(points_history, times_history):
    if len(points_history) < 4:
        return False, 0
    
    time_window = times_history[-1] - times_history[0]
    if time_window < 0.2:
        return False, 0
    
    # Calculate movement amplitude
    amplitudes = [abs(points_history[i][0] - points_history[i-1][0]) for i in range(1, len(points_history))]
    avg_amplitude = sum(amplitudes) / len(amplitudes) if amplitudes else 0
    
    is_restless = avg_amplitude > 120  # Increased threshold to 120
    
    return is_restless, avg_amplitude

def detect_frequency_movement(points_history, times_history):
    if len(points_history) < 8:
        return False, 0
    
    points = np.array(points_history)
    times = np.array(times_history)
    
    x_movements = points[:, 0]
    y_movements = points[:, 1]
    
    time_window = times[-1] - times[0]
    if time_window == 0:
        return False, 0
        
    # Calculate frequency of direction changes
    x_changes = len(np.where(np.diff(np.signbit(np.diff(x_movements))))[0])
    y_changes = len(np.where(np.diff(np.signbit(np.diff(y_movements))))[0])
    
    x_freq = x_changes / time_window
    y_freq = y_changes / time_window
    
    is_jittering = (x_freq > 4 and x_freq < 12) or (y_freq > 4 and y_freq < 12)
    
    return is_jittering, max(x_freq, y_freq)

def detect_fidgeting(points_history, times_history):
    if len(points_history) < 8:
        return False, 0
    
    points = np.array(points_history)
    times = np.array(times_history)
    
    dt = np.diff(times)
    if len(dt) == 0 or np.any(dt == 0):
        return False, 0
        
    velocities = np.diff(points, axis=0) / dt[:, np.newaxis]
    accelerations = np.diff(velocities, axis=0) / dt[1:, np.newaxis]
    
    peaks_x, _ = find_peaks(np.abs(accelerations[:, 0]))
    peaks_y, _ = find_peaks(np.abs(accelerations[:, 1]))
    
    time_window = times[-1] - times[0]
    if time_window == 0:
        return False, 0
        
    freq_x = len(peaks_x) / time_window
    freq_y = len(peaks_y) / time_window
    movement_freq = max(freq_x, freq_y)
    
    is_fidgeting = movement_freq > 0.4  # Cutoff at 0.4 Hz
    
    return is_fidgeting, movement_freq

try:
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        try:
            # Track the first face detected
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                center_point = (x + w//2, y + h//2)
                current_time = time.time()
                
                point_history.append(center_point)
                time_history.append(current_time)
                
                # Detect movements using updated methods
                is_restless, amplitude = detect_restlessness(point_history, time_history)
                is_jittering, freq = detect_frequency_movement(point_history, time_history)
                is_fidgeting, fidget_freq = detect_fidgeting(point_history, time_history)
                
                # Draw rectangle around face
                cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Update metrics using the container
                metrics_container.columns(3)[0].metric(
                    "Restlessness",
                    f"{amplitude:.1f}",
                    delta="Active" if is_restless else "Normal",
                    delta_color="inverse"
                )
                metrics_container.columns(3)[1].metric(
                    "Rapid Movement",
                    f"{freq:.1f} Hz",
                    delta="Detected" if is_jittering else "Normal",
                    delta_color="inverse"
                )
                metrics_container.columns(3)[2].metric(
                    "Fidgeting",
                    f"{fidget_freq:.1f} Hz",
                    delta="Active" if is_fidgeting else "Normal",
                    delta_color="inverse"
                )
            
            # Analyze emotions
            result = DeepFace.analyze(
                frame_rgb,
                actions=['emotion'],
                enforce_detection=False
            )
            
            # Get dominant emotion and format the results
            emotion = result[0]['dominant_emotion']
            emotion_scores = result[0]['emotion']
            
            # Update emotion chart
            emotion_data = pd.DataFrame({
                'Emotion': list(emotion_scores.keys()),
                'Score': list(emotion_scores.values())
            }).sort_values('Score', ascending=True)

            chart = alt.Chart(emotion_data).mark_bar().encode(
                x='Score:Q',
                y=alt.Y('Emotion:N', sort='-x'),
                color=alt.condition(
                    alt.datum.Emotion == emotion,
                    alt.value('orange'),
                    alt.value('steelblue')
                )
            ).properties(height=200)
            
            emotion_chart_container.altair_chart(chart, use_container_width=True)
            
        except Exception as e:
            pass  # Skip frame if face not detected or other error occurs
        
        # Update video frame
        video_placeholder.image(frame_rgb)
        
        # Add a status indicator
        st.sidebar.success("Camera is running")
        
        # Add a small delay to reduce CPU usage
        time.sleep(0.1)

except Exception as e:
    st.sidebar.error(f"Error: {str(e)}")
finally:
    cap.release()
    st.sidebar.warning("Camera stopped") 