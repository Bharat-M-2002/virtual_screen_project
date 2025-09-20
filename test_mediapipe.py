import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
WRITE_TIMEOUT = 0.8  # Reduced for better responsiveness
ERASER_HOLD_TIME = 1.0  # Reduced for better UX
DRAWING_THICKNESS = 5
SMOOTHING_BUFFER_SIZE = 2  # Reduced for more responsive drawing
ERASER_RADIUS = 40
CURSOR_RADIUS = 10
CAMERA_WIDTH = 840
CAMERA_HEIGHT = 680
CAMERA_FPS = 60
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
GESTURE_CONFIDENCE_THRESHOLD = 0.8  # New: confidence threshold for gesture switching

# Colors
DRAWING_COLOR = (0, 255, 255)  # Yellow
CURSOR_COLOR = (0, 255, 0)     # Green
ERASER_COLOR = (0, 0, 0)       # Black
TEXT_COLOR = (255, 255, 255)   # White
MODE_COLORS = {
    "write": (0, 255, 255),
    "cursor": (0, 255, 0),
    "eraser": (0, 0, 255),
    "idle": (128, 128, 128)
}

class HandGestureDrawing:
    def __init__(self):
        self.setup_mediapipe()
        self.setup_camera()
        self.init_variables()
        self.create_save_folder()

    def setup_mediapipe(self):
        """Initialize MediaPipe hands detection"""
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE
            )
            logger.info("MediaPipe hands initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            raise

    def setup_camera(self):
        """Initialize camera with error handling"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Camera not accessible")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
            logger.info("Camera initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            raise

    def init_variables(self):
        """Initialize all working variables"""
        self.canvas = None
        self.prev_x, self.prev_y = 0, 0
        self.points_buffer = deque(maxlen=SMOOTHING_BUFFER_SIZE)
        self.eraser_start_time = None
        self.eraser_active = False
        self.mode = "idle"
        self.last_write_time = 0
        self.save_counter = 1
        self.gesture_confidence = 0
        self.last_gesture = "idle"
        self.mode_switch_cooldown = 0
        self.fps_counter = deque(maxlen=30)
        self.last_fps_time = time.time()

    def create_save_folder(self):
        """Create folder for saving drawings"""
        self.save_folder = "Screenshots"
        try:
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)
                logger.info(f"Created save folder: {self.save_folder}")
        except Exception as e:
            logger.error(f"Failed to create save folder: {e}")
            self.save_folder = "."

    def fingers_up(self, hand_landmarks):
        """
        Improved finger detection with better accuracy
        """
        lm = hand_landmarks.landmark
        fingers = []
        
        # Get hand orientation (checking if right or left hand)
        wrist_x = lm[0].x
        mcp_x = lm[9].x  # Middle finger MCP
        is_right_hand = mcp_x > wrist_x
        
        # Thumb - improved detection
        thumb_tip_x = lm[4].x
        thumb_ip_x = lm[3].x
        
        if is_right_hand:
            fingers.append(1 if thumb_tip_x > thumb_ip_x + 0.02 else 0)
        else:
            fingers.append(1 if thumb_tip_x < thumb_ip_x - 0.02 else 0)
        
        # Other fingers - check if tip is above PIP joint
        tip_indices = [8, 12, 16, 20]
        pip_indices = [6, 10, 14, 18]
        
        for tip_idx, pip_idx in zip(tip_indices, pip_indices):
            fingers.append(1 if lm[tip_idx].y < lm[pip_idx].y - 0.02 else 0)
        
        return fingers

    def determine_gesture(self, fingers):
        """
        Clear gesture detection with confidence scoring
        """
        # Count fingers (excluding thumb for main gesture)
        thumb_up = fingers[0]
        other_fingers = fingers[1:]
        fingers_up_count = sum(other_fingers)
        
        # Define clear gesture patterns
        gesture = "idle"
        confidence = 0.0
        
        # Pattern matching with confidence
        if fingers_up_count == 1 and other_fingers[0] == 1:  # Only index
            gesture = "write"
            confidence = 1.0
        elif fingers_up_count == 2 and other_fingers[0] == 1 and other_fingers[1] == 1:  # Index + Middle
            gesture = "cursor"
            confidence = 1.0
        elif fingers_up_count >= 3:  # Three or more fingers
            gesture = "eraser"
            confidence = 0.9 if fingers_up_count == 3 else 1.0
        elif fingers_up_count == 0:  # Fist
            gesture = "idle"
            confidence = 1.0
        else:
            # Ambiguous gesture
            gesture = "idle"
            confidence = 0.3
        
        return gesture, confidence

    def update_mode(self, new_gesture, confidence):
        """
        Intelligent mode switching with proper priority
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time < self.mode_switch_cooldown:
            return self.mode
        
        # Immediate switch for high-confidence gestures when leaving write mode
        if self.mode == "write" and new_gesture != "write" and confidence >= GESTURE_CONFIDENCE_THRESHOLD:
            # Stop writing immediately when user shows clear different gesture
            self.reset_drawing_state()
            self.mode = new_gesture
            self.mode_switch_cooldown = current_time + 0.1  # Small cooldown to prevent flickering
            return self.mode
        
        # Write mode persistence (only for ambiguous/idle gestures)
        if self.mode == "write" and new_gesture == "idle" and confidence < GESTURE_CONFIDENCE_THRESHOLD:
            # Continue writing if recent activity and gesture is ambiguous
            if current_time - self.last_write_time < WRITE_TIMEOUT:
                return "write"
        
        # Standard mode switching for other cases
        if confidence >= GESTURE_CONFIDENCE_THRESHOLD and new_gesture != self.mode:
            self.mode = new_gesture
            self.mode_switch_cooldown = current_time + 0.1
            
            # Reset states when switching modes
            if new_gesture != "write":
                self.reset_drawing_state()
            if new_gesture != "eraser":
                self.eraser_start_time = None
                self.eraser_active = False
        
        return self.mode

    def handle_write_mode(self, x, y):
        """Enhanced drawing with better smoothing"""
        if self.eraser_active:
            return
        
        self.points_buffer.append((x, y))
        self.last_write_time = time.time()
        
        if len(self.points_buffer) >= 2:
            # Use weighted average for smoother lines
            weights = np.array([0.2, 0.3, 0.5][-len(self.points_buffer):])
            weights = weights / weights.sum()
            
            smoothed_x = int(np.average([p[0] for p in self.points_buffer], weights=weights))
            smoothed_y = int(np.average([p[1] for p in self.points_buffer], weights=weights))
            
            if self.prev_x != 0 and self.prev_y != 0:
                # Draw with anti-aliasing
                cv2.line(self.canvas, (self.prev_x, self.prev_y), 
                        (smoothed_x, smoothed_y), DRAWING_COLOR, DRAWING_THICKNESS, cv2.LINE_AA)
            
            self.prev_x, self.prev_y = smoothed_x, smoothed_y

    def handle_cursor_mode(self, frame, x, y):
        """Enhanced cursor with better visibility"""
        # Draw cursor with animation effect
        radius = CURSOR_RADIUS + int(3 * np.sin(time.time() * 5))
        cv2.circle(frame, (x, y), radius, CURSOR_COLOR, 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 3, CURSOR_COLOR, -1)
        
        # Add crosshair for precision
        cv2.line(frame, (x - 10, y), (x + 10, y), CURSOR_COLOR, 1)
        cv2.line(frame, (x, y - 10), (x, y + 10), CURSOR_COLOR, 1)
        
        self.reset_drawing_state()

    def handle_eraser_mode(self, frame, x, y):
        """Improved eraser with visual feedback"""
        if self.eraser_start_time is None:
            self.eraser_start_time = time.time()
            self.eraser_active = False
        
        elapsed = time.time() - self.eraser_start_time
        progress = min(elapsed / ERASER_HOLD_TIME, 1.0)
        
        if progress >= 1.0:
            self.eraser_active = True
            # Active erasing
            cv2.circle(self.canvas, (x, y), ERASER_RADIUS, ERASER_COLOR, -1)
            
            # Visual feedback - filled circle
            overlay = frame.copy()
            cv2.circle(overlay, (x, y), ERASER_RADIUS, (255, 100, 100), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        else:
            # Loading animation
            end_angle = int(360 * progress)
            cv2.ellipse(frame, (x, y), (ERASER_RADIUS, ERASER_RADIUS), 
                       0, 0, end_angle, (0, 0, 255), 3)
            
            # Show countdown
            cv2.putText(frame, f"{(1-progress)*ERASER_HOLD_TIME:.1f}s", 
                       (x - 20, y - ERASER_RADIUS - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        self.reset_drawing_state()

    def reset_drawing_state(self):
        """Reset drawing-related variables"""
        self.points_buffer.clear()
        self.prev_x, self.prev_y = 0, 0

    def handle_idle_mode(self):
        """Handle idle state"""
        # Only reset if we've been idle for a moment
        if time.time() - self.last_write_time > WRITE_TIMEOUT:
            self.reset_drawing_state()
        self.eraser_start_time = None
        self.eraser_active = False

    def save_drawing(self):
        """Save current drawing"""
        try:
            white_bg = np.ones((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8) * 255
            canvas_resized = cv2.resize(self.canvas, (CAMERA_WIDTH, CAMERA_HEIGHT))
            stroke_mask = np.any(canvas_resized != [0, 0, 0], axis=-1)
            white_bg[stroke_mask] = [0, 0, 0]
            
            filename = os.path.join(self.save_folder, f"drawing_{self.save_counter:03d}.png")
            if cv2.imwrite(filename, white_bg):
                print(f"✅ Drawing saved as {filename}")
                self.save_counter += 1
            else:
                logger.error("Failed to save image")
        except Exception as e:
            logger.error(f"Error saving drawing: {e}")

    def clear_canvas(self, shape):
        """Clear the drawing canvas"""
        self.canvas = np.zeros(shape, dtype=np.uint8)
        self.reset_drawing_state()
        self.eraser_start_time = None
        self.eraser_active = False
        self.mode = "idle"
        print("🧹 Canvas cleared!")

    def calculate_fps(self):
        """Calculate and return current FPS"""
        current_time = time.time()
        self.fps_counter.append(current_time)
        if len(self.fps_counter) > 1:
            fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])
            return int(fps)
        return 0

    def draw_ui(self, frame):
        """Enhanced UI with better visual feedback"""
        h, w = frame.shape[:2]
        
        # FPS counter
        fps = self.calculate_fps()
        cv2.putText(frame, f"FPS: {fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Mode display with color coding
        mode_color = MODE_COLORS.get(self.mode, (255, 255, 255))
        mode_text = f"Mode: {self.mode.upper()}"
        
        # Background for mode
        (text_w, text_h), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (w - text_w - 25, 10), (w - 5, text_h + 25), (0, 0, 0), -1)
        cv2.rectangle(frame, (w - text_w - 25, 10), (w - 5, text_h + 25), mode_color, 2)
        cv2.putText(frame, mode_text, (w - text_w - 20, text_h + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        
        # Gesture confidence indicator
        if self.gesture_confidence < GESTURE_CONFIDENCE_THRESHOLD:
            cv2.putText(frame, "?", (w - 35, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # Instructions with icons
        instructions = [
            # "👆 1 finger: Draw",
            # "✌️ 2 fingers: Cursor",
            # "🖐️ 3+ fingers: Eraser",
            # "━━━━━━━━━━━━━━━━",
            "[Q] Quit  [C] Clear  [S] Save"
        ]
        
        # Semi-transparent background for instructions
        overlay = frame.copy()
        # cv2.rectangle(overlay, (5, h - 120), (250, h - 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        for i, instruction in enumerate(instructions):
            y_pos = h - 20 + (i * 20)
            cv2.putText(frame, instruction, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)

    def process_frame(self):
        """Optimized frame processing"""
        try:
            ret, frame = self.cap.read()
            if not ret:
                return None
            
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            if self.canvas is None:
                self.canvas = np.zeros((h, w, c), dtype=np.uint8)
            
            # Process hand detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False  # Performance optimization
            result = self.hands.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                
                # Get index finger position
                lm = hand_landmarks.landmark
                x = int(lm[8].x * w)
                y = int(lm[8].y * h)
                
                # Detect gesture
                fingers = self.fingers_up(hand_landmarks)
                gesture, confidence = self.determine_gesture(fingers)
                self.gesture_confidence = confidence
                
                # Update mode with intelligent switching
                self.mode = self.update_mode(gesture, confidence)
                
                # Execute mode action
                if self.mode == "write":
                    self.handle_write_mode(x, y)
                elif self.mode == "cursor":
                    self.handle_cursor_mode(frame, x, y)
                elif self.mode == "eraser":
                    self.handle_eraser_mode(frame, x, y)
                else:
                    self.handle_idle_mode()
                
                # Draw hand landmarks (lighter)
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(150, 150, 150), thickness=1))
            else:
                # No hands detected
                if self.mode == "write" and time.time() - self.last_write_time > WRITE_TIMEOUT:
                    self.mode = "idle"
                    self.handle_idle_mode()
            
            # Composite output
            output = cv2.addWeighted(frame, 0.7, self.canvas, 0.8, 0)
            self.draw_ui(output)
            
            return output
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None

    def run(self):
        """Main application loop"""
        print("\n🎨 Hand Gesture Drawing App - Optimized")
        print("=" * 50)
        print("✋ Gestures:")
        print("  👆 1 finger (index) = Draw")
        print("  ✌️ 2 fingers (index+middle) = Cursor")
        print("  🖐️ 3+ fingers = Eraser (hold 1s to activate)")
        print("\n⌨️ Keyboard:")
        print("  Q = Quit  |  C = Clear  |  S = Save")
        print("=" * 50)
        print("\n✨ Tips:")
        print("  • Make clear, distinct gestures for best recognition")
        print("  • Drawing continues briefly after lowering finger")
        print("  • Switch modes with confident gestures")
        print("=" * 50 + "\n")
        
        try:
            while True:
                output = self.process_frame()
                if output is None:
                    break
                
                cv2.imshow("Hand Gesture Drawing - Optimized", output)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.clear_canvas(output.shape)
                elif key == ord('s'):
                    self.save_drawing()
                    
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        try:
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """Main entry point"""
    try:
        app = HandGestureDrawing()
        app.run()
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()