import cv2
import time
import csv
import os
import threading
from datetime import datetime
import google.generativeai as genai
import numpy as np
import streamlit as st
from PIL import Image

# =====================
# CONFIGURATION
# =====================
GEMINI_API_KEY =  "AIzaSyCw-Jd3yfjCYz0lJ-9gN1tpEiAphdWseHM" # Use secrets in Streamlit
LOG_FILE = "chair_counts.csv"
FRAME_INTERVAL = 5  # seconds
RESIZE_WIDTH = 640   # Balanced resolution for accuracy and speed

# =====================
# AGENT DEFINITIONS
# =====================

class DetectionAgent:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def detect_chairs(self, frame) -> dict:
        try:
            # Convert to PIL Image for better compression
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            prompt = """
            Analyze this image and:
            1. Identify all chairs
            2. Count the total number of chairs
            3. Return response in JSON format only: 
                {"count": number, "description": "string"}
            Example: {"count": 4, "description": "3 office chairs and 1 armchair"}
            """
            
            response = self.model.generate_content(
                contents=[prompt, pil_img],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=100,
                    temperature=0.2
                )
            )
            
            # Extract JSON from response
            if response.candidates and response.candidates[0].content.parts:
                json_str = response.candidates[0].content.parts[0].text
                json_str = json_str[json_str.find('{'):json_str.rfind('}')+1]
                return eval(json_str)
            return {"count": 0, "description": "Detection failed"}
        except Exception as e:
            st.error(f"Detection error: {str(e)}")
            return {"count": 0, "description": "Error in detection"}

class LoggingAgent:
    def __init__(self, log_file: str):
        self.log_file = log_file
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "chair_count", "description"])

    def log(self, count: int, description: str):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), count, description])

# =====================
# STREAMLIT APP
# =====================

def main():
    st.set_page_config(
        page_title="Real-Time Chair Detection",
        page_icon="🪑",
        layout="wide"
    )
    
    # Initialize session state
    if 'detection_agent' not in st.session_state:
        st.session_state.detection_agent = DetectionAgent(GEMINI_API_KEY)
        st.session_state.logger = LoggingAgent(LOG_FILE)
        st.session_state.running = False
        st.session_state.last_detection = {"count": 0, "description": "No chairs detected"}
        st.session_state.last_update = time.time()
    
    # Header
    st.title("🪑 Real-Time Chair Detection System")
    st.markdown("""
    This system uses Gemini Vision AI to detect and count chairs in real-time.
    - **Instructions**: Click 'Start Detection' to begin. Detection runs every 5 seconds.
    - **Output**: Shows chair count and description of detected chairs.
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        if st.button("🚀 Start Detection", disabled=st.session_state.running):
            st.session_state.running = True
            st.success("Detection started!")
            
        if st.button("🛑 Stop Detection", disabled=not st.session_state.running):
            st.session_state.running = False
            st.info("Detection stopped.")
        
        st.divider()
        st.header("Detection Settings")
        st.session_state.detection_interval = st.slider(
            "Detection Interval (seconds)", 
            min_value=2, 
            max_value=10, 
            value=5
        )
        
        st.divider()
        st.header("Current Status")
        status_placeholder = st.empty()
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Live Camera Feed")
        camera_placeholder = st.empty()
        
    with col2:
        st.subheader("Detection Results")
        results_placeholder = st.empty()
        st.divider()
        st.subheader("Detection Logs")
        log_placeholder = st.empty()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not access webcam.")
        st.stop()
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESIZE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESIZE_WIDTH * 9/16)
    
    try:
        while st.session_state.running:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from camera")
                break
            
            # Convert to RGB for display
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame in Streamlit
            camera_placeholder.image(
                display_frame, 
                caption="Live Camera Feed", 
                use_column_width=True
            )
            
            # Check if it's time for detection
            current_time = time.time()
            if current_time - st.session_state.last_update > st.session_state.detection_interval:
                # Run chair detection
                result = st.session_state.detection_agent.detect_chairs(frame)
                
                # Update session state
                st.session_state.last_detection = result
                st.session_state.last_update = current_time
                
                # Log results
                st.session_state.logger.log(result["count"], result["description"])
                
                # Update status
                status_placeholder.success(f"Last detection: {datetime.now().strftime('%H:%M:%S')}")
            
            # Display results
            with results_placeholder.container():
                st.metric("Chair Count", st.session_state.last_detection["count"])
                st.subheader("Description")
                st.info(st.session_state.last_detection["description"])
                
                # Add some visual feedback
                if st.session_state.last_detection["count"] > 0:
                    st.balloons()
            
            # Display logs
            try:
                with open(LOG_FILE, "r") as f:
                    logs = f.readlines()[-10:]  # Show last 10 entries
                with log_placeholder.container():
                    st.code("\n".join(logs), language="csv")
            except:
                pass
            
            # Add small delay to prevent high CPU usage
            time.sleep(0.1)
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        st.session_state.running = False
    
    # Show message when not running
    if not st.session_state.running:
        with camera_placeholder:
            st.info("Click 'Start Detection' to begin chair detection")
        
        with results_placeholder:
            st.metric("Chair Count", 0)
            st.info("No active detection")
        
        # Show sample image when idle
        st.divider()
        st.subheader("Sample Chair Detection")
        st.image("https://images.unsplash.com/photo-1503602642458-232111445657?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&h=400", 
                 caption="Example of chairs that can be detected")

if __name__ == "__main__":
    main()
# # =====================

# import cv2
# import time
# import csv
# import os
# import threading
# from datetime import datetime
# import google.generativeai as genai

# # =====================
# # CONFIGURATION
# # =====================
# GEMINI_API_KEY = "AIzaSyCw-Jd3yfjCYz0lJ-9gN1tpEiAphdWseHM"
# LOG_FILE = "mobile_counts.csv"
# FRAME_INTERVAL = 3  # seconds
# RESIZE_WIDTH = 320   # Reduced resolution for faster processing

# # =====================
# # AGENT DEFINITIONS
# # =====================

# class PerceptionAgent:
#     def __init__(self, camera_index=0):
#         self.cap = cv2.VideoCapture(camera_index)
#         if not self.cap.isOpened():
#             raise RuntimeError("Webcam not accessible.")
#         # Set camera resolution for faster processing
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESIZE_WIDTH)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESIZE_WIDTH * 9/16)

#     def capture_frame(self):
#         ret, frame = self.cap.read()
#         if not ret:
#             raise RuntimeError("Failed to capture frame from webcam.")
#         return frame

#     def release(self):
#         self.cap.release()

# class DetectionAgent:
#     def __init__(self, api_key: str):
#         genai.configure(api_key=api_key)
#         self.model = genai.GenerativeModel('gemini-1.5-flash')  # Faster model

#     def detect_mobile(self, frame) -> str:
#         # Resize frame for faster processing
#         small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
#         # Encode frame as JPEG
#         ret, img_bytes = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
#         if not ret:
#             return "error"
#         img_bytes = img_bytes.tobytes()
        
#         prompt = "Is there a mobile phone visible in this image? Reply only with 'yes' or 'no'."
#         try:
#             response = self.model.generate_content(
#                 contents=[prompt, {"mime_type": "image/jpeg", "data": img_bytes}],
#                 generation_config=genai.types.GenerationConfig(
#                     max_output_tokens=1,
#                     temperature=0.0
#                 )
#             )
#             answer = response.text.strip().lower()
#             return answer if answer in ("yes", "no") else "unknown"
#         except:
#             return "error"

# class MemoryAgent:
#     def __init__(self):
#         self.last_detected = False

#     def update(self, detected: bool):
#         self.last_detected = detected

#     def was_last_detected(self) -> bool:
#         return self.last_detected

# class CountAgent:
#     def __init__(self):
#         self.count = 0

#     def increment(self):
#         self.count += 1

#     def get_count(self):
#         return self.count

# class LoggingAgent:
#     def __init__(self, log_file: str):
#         self.log_file = log_file
#         if not os.path.exists(self.log_file):
#             with open(self.log_file, 'w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(["timestamp", "count"])

#     def log(self, count: int):
#         with open(self.log_file, 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow([datetime.now().isoformat(), count])

# class CoordinatorAgent:
#     def __init__(self):
#         self.perception = PerceptionAgent()
#         self.detection = DetectionAgent(GEMINI_API_KEY)
#         self.memory = MemoryAgent()
#         self.counter = CountAgent()
#         self.logger = LoggingAgent(LOG_FILE)
#         self.running = True
#         self.latest_frame = None
#         self.detection_result = None
#         self.border_active = False
#         self.border_timestamp = 0
#         self.lock = threading.Lock()

#     def detection_worker(self):
#         while self.running:
#             with self.lock:
#                 if self.latest_frame is None:
#                     time.sleep(0.1)
#                     continue
                
#                 frame = self.latest_frame.copy()
            
#             detected = self.detection.detect_mobile(frame)
            
#             with self.lock:
#                 self.detection_result = detected
                
#                 # Update state only if detection was successful
#                 if detected == "yes":
#                     if not self.memory.was_last_detected():
#                         self.counter.increment()
#                         print(f"📱 Mobile detected. Count: {self.counter.get_count()}")
#                         self.logger.log(self.counter.get_count())
#                     else:
#                         print("📱 Mobile still present. No new count.")
#                     self.memory.update(True)
#                     self.border_active = True
#                     self.border_timestamp = time.time()
#                 elif detected == "no":
#                     print("No mobile detected.")
#                     self.memory.update(False)
#                 else:
#                     print(f"Detection issue: {detected}")
            
#             time.sleep(FRAME_INTERVAL)

#     def run(self):
#         print("[CoordinatorAgent] Starting mobile detection system. Press 'Q' to quit.")
        
#         # Start detection thread
#         detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
#         detection_thread.start()
        
#         try:
#             while self.running:
#                 start_time = time.time()
                
#                 # Capture frame
#                 frame = self.perception.capture_frame()
                
#                 # Store frame for detection thread
#                 with self.lock:
#                     self.latest_frame = frame
                
#                 # Create display frame
#                 display_frame = frame.copy()
                
#                 # Handle border display (with timeout)
#                 border_timeout = 2.0  # seconds to keep border visible
#                 if self.border_active and (time.time() - self.border_timestamp) < border_timeout:
#                     border_thickness = 10
#                     border_color = (0, 255, 0)  # Green
#                     display_frame = cv2.copyMakeBorder(
#                         display_frame,
#                         top=border_thickness,
#                         bottom=border_thickness,
#                         left=border_thickness,
#                         right=border_thickness,
#                         borderType=cv2.BORDER_CONSTANT,
#                         value=border_color
#                     )
#                 else:
#                     self.border_active = False
                
#                 # Display frame
#                 cv2.imshow('Webcam - Mobile Detection', display_frame)
                
#                 # Handle exit
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     print("[CoordinatorAgent] Exiting.")
#                     break
                
#                 # Control frame rate
#                 elapsed = time.time() - start_time
#                 if elapsed < 0.03:  # ~30 FPS
#                     time.sleep(0.03 - elapsed)
                    
#         except KeyboardInterrupt:
#             print("\n[CoordinatorAgent] Stopping system.")
#         finally:
#             self.running = False
#             detection_thread.join(timeout=1.0)
#             self.perception.release()
#             cv2.destroyAllWindows()

# # =====================
# # MAIN ENTRY POINT
# # =====================

# def main():
#     coordinator = CoordinatorAgent()
#     coordinator.run()

# if __name__ == "__main__":
#     main()