import cv2
import numpy as np
import time
import logging
import os
from collections import deque

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s: %(message)s')

# Optimize for Raspberry Pi
# os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'  # Disable MSMF backend for Windows
# os.environ['OPENCV_VIDEOIO_PRIORITY_V4L'] = '1'  # Priority for V4L (Video for Linux)


class EggDetector:
    def __init__(self):
        # Pixel-to-mm conversion factor (needs calibration for actual setup)
        self.px_to_mm = 0.5  # Example: 1 pixel = 0.5mm

        # Size thresholds in mm
        # Large Eggs: 1-1/8 inch (28.575 mm) diameter
        self.large_min_diameter = 25.0  # mm
        self.large_max_diameter = 32.0  # mm

        # Medium Eggs: smaller than large eggs
        self.medium_min_diameter = 15.0  # mm
        self.medium_max_diameter = 24.9  # mm

        # Track counts by category
        self.large_good_count = 0
        self.large_bad_count = 0
        self.medium_good_count = 0
        self.medium_bad_count = 0

        self.min_area = 500
        self.camera = None
        self.camera_index = 0  # Default to first camera

        # Raspberry Pi optimizations
        self.resize_width = 640  # Reduced from 1280 for Pi
        self.blur_kernel_size = 3  # Reduced from 5 for Pi
        self.morph_kernel_size = 3  # Reduced from 5 for Pi
        self.show_tracks = True

        # Pi-specific camera settings
        self.use_picamera = True  # Set to True to use the Pi camera module

        self.tracked_eggs = {}
        self.next_egg_id = 1
        self.tracked_positions = {}
        self.max_track_history = 20

        self.frame_times = deque(maxlen=30)
        self.confidence_threshold = 0.5  # Minimum confidence to classify
        self.fps = 0

        # Depth estimation parameters
        self.calibration_distance_mm = 300  # Example: camera was calibrated at 300mm
        self.known_pixels_at_calibration = 100  # Example: object was 100px at calibration distance

    def connect_camera(self):
        try:
            # Release existing camera if any
            if self.camera is not None:
                self.camera.release()

            if self.use_picamera:
                try:
                    # Try to import picamera2 (for newer Raspberry Pi OS)
                    from picamera2 import Picamera2

                    # Initialize the camera with picamera2
                    self.picam2 = Picamera2()

                    # Configure the camera with lower resolution for better performance
                    config = self.picam2.create_preview_configuration(
                        main={"size": (640, 480), "format": "RGB888"}
                    )
                    self.picam2.configure(config)
                    self.picam2.start()

                    print("Connected to Raspberry Pi Camera using picamera2")
                    self.camera = "picamera2"  # Mark that we're using picamera2
                    return True

                except (ImportError, ModuleNotFoundError):
                    try:
                        # Fall back to older picamera if picamera2 is not available
                        import picamera
                        from picamera.array import PiRGBArray

                        # Initialize picamera
                        self.picam = picamera.PiCamera()
                        self.picam.resolution = (640, 480)
                        self.raw_capture = PiRGBArray(self.picam, size=(640, 480))

                        # Warm up the camera
                        time.sleep(0.1)

                        print("Connected to Raspberry Pi Camera using picamera")
                        self.camera = "picamera"  # Mark that we're using picamera
                        return True

                    except (ImportError, ModuleNotFoundError):
                        # If neither picamera module is available, fall back to OpenCV
                        print("PiCamera modules not found, falling back to OpenCV")
                        self.use_picamera = False

            # Fall back to regular OpenCV if picamera is not used or failed
            self.camera = cv2.VideoCapture(self.camera_index)

            if not self.camera.isOpened():
                logging.error(f"Could not open camera at index {self.camera_index}")
                return False

            # Set camera properties for better performance on Pi
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Set a lower FPS for Pi
            self.camera.set(cv2.CAP_PROP_FPS, 15)

            print(f"Connected to camera at index {self.camera_index} using OpenCV")
            return True

        except Exception as e:
            logging.error(f"Error connecting to camera: {e}")
            return False

    def determine_egg_size(self, diameter_mm):
        """Determine egg size based on dimensions."""
        if diameter_mm >= self.large_min_diameter and diameter_mm <= self.large_max_diameter:
            return "Large"
        elif diameter_mm >= self.medium_min_diameter and diameter_mm <= self.medium_max_diameter:
            return "Medium"
        else:
            return "Unknown Size"

    def track_eggs(self, egg_info):
        """Track eggs across consecutive frames."""
        updated_tracks = {}

        for egg in egg_info:
            x, y = egg['position']
            found_match = False

            for prev_id, (px, py) in self.tracked_eggs.items():
                if abs(px - x) < 50 and abs(py - y) < 50:  # Tracking threshold
                    updated_tracks[prev_id] = (x, y)
                    egg['id'] = prev_id
                    found_match = True
                    break

            # Create new track if no match found
            if not found_match:
                updated_tracks[self.next_egg_id] = (x, y)
                egg['id'] = self.next_egg_id
                self.next_egg_id += 1

                # Initialize track history for new egg
                self.tracked_positions[egg['id']] = deque(maxlen=self.max_track_history)

            # Update track history
            if egg['id'] in self.tracked_positions:
                self.tracked_positions[egg['id']].append((x, y))
            else:
                self.tracked_positions[egg['id']] = deque([(x, y)], maxlen=self.max_track_history)

        # Update current egg positions
        self.tracked_eggs = updated_tracks

    def optimize_frame(self, frame):
        """Resize frame for faster processing if needed."""
        if frame is None:
            return None

        try:
            height, width = frame.shape[:2]

            if width > self.resize_width:
                scale = self.resize_width / width
                # Use INTER_AREA for more efficient resizing
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

            return frame

        except Exception as e:
            logging.error(f"Error optimizing frame: {e}")
            return frame  # Return original frame on error

    def process_frame(self, frame):
        """Process a frame to detect and classify eggs."""
        start_time = time.time()
        result_frame = None
        egg_info = []

        try:
            if frame is None or frame.size == 0:
                return None, None

            frame = self.optimize_frame(frame)
            if frame is None:
                return None, None

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Apply thresholding for color detection
            # For white eggs (good eggs), we're looking for high value, low saturation
            mask_good = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))

            # For off-white/tan eggs (bad eggs), we're looking for slightly more saturation
            mask_bad = cv2.inRange(hsv, np.array([20, 30, 180]), np.array([40, 70, 230]))

            mask = cv2.bitwise_or(mask_good, mask_bad)

            # Reduce noise
            mask = cv2.GaussianBlur(mask, (self.blur_kernel_size, self.blur_kernel_size), 0)
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            result_frame = frame.copy()

            # Reset category counts
            self.large_good_count = 0
            self.large_bad_count = 0
            self.medium_good_count = 0
            self.medium_bad_count = 0

            for contour in contours:
                area = cv2.contourArea(contour)

                if area > self.min_area:
                    if len(contour) >= 5:
                        # Fit an ellipse to the contour
                        ellipse = cv2.fitEllipse(contour)

                        # Calculate dimensions
                        (center, axes, angle) = ellipse
                        major_axis = max(axes)
                        minor_axis = min(axes)

                        # Calculate average diameter in mm
                        diameter_mm = ((major_axis + minor_axis) / 2) * self.px_to_mm

                        # Create mask segment for color analysis
                        mask_segment = np.zeros_like(mask)
                        cv2.drawContours(mask_segment, [contour], -1, 255, -1)
                        good_match = cv2.countNonZero(cv2.bitwise_and(mask_segment, mask_good))
                        bad_match = cv2.countNonZero(cv2.bitwise_and(mask_segment, mask_bad))

                        total_match = max(good_match + bad_match, 1)
                        good_confidence = good_match / total_match
                        bad_confidence = bad_match / total_match

                        # Determine egg quality based on color
                        if good_confidence >= bad_confidence and good_confidence >= self.confidence_threshold:
                            quality = "Good"
                            confidence = good_confidence
                            color = (0, 255, 0)  # Green
                        elif bad_confidence > good_confidence and bad_confidence >= self.confidence_threshold:
                            quality = "Bad"
                            confidence = bad_confidence
                            color = (0, 0, 255)  # Red
                        else:
                            quality = "Uncertain"
                            confidence = max(good_confidence, bad_confidence)
                            color = (255, 255, 0)  # Yellow

                        # Determine egg size
                        size = self.determine_egg_size(diameter_mm)

                        # Combine size and quality for category
                        category = f"{size} {quality}"

                        # Update category counts
                        if category == "Large Good":
                            self.large_good_count += 1
                        elif category == "Large Bad":
                            self.large_bad_count += 1
                        elif category == "Medium Good":
                            self.medium_good_count += 1
                        elif category == "Medium Bad":
                            self.medium_bad_count += 1

                        # Draw ellipse
                        cv2.ellipse(result_frame, ellipse, color, 2)

                        egg_info.append({
                            'id': 0,  # Will be assigned in track_eggs
                            'quality': quality,
                            'size': size,
                            'category': category,
                            'position': tuple(map(int, center)),
                            'dimensions': axes,
                            'angle': angle,
                            'area': area,
                            'confidence': confidence,
                            'diameter_mm': diameter_mm
                        })

            self.track_eggs(egg_info)

            result_frame = self.visualize_results(result_frame, egg_info)

            # Calculate FPS
            process_time = time.time() - start_time
            self.frame_times.append(process_time)
            self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0

            return result_frame, egg_info

        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            if frame is not None:
                return frame, []  # Return original frame on error
            return None, []

    def estimate_depth(self, diameter_pixels, known_dimension_mm=28.575):
        """Estimate the depth/distance of an egg from the camera."""
        try:
            # Use the pinhole camera model
            focal_length = (self.known_pixels_at_calibration * self.calibration_distance_mm) / known_dimension_mm

            # Calculate depth
            depth_mm = (known_dimension_mm * focal_length) / diameter_pixels

            return depth_mm

        except Exception as e:
            logging.error(f"Error estimating depth: {e}")
            return None

    def visualize_results(self, frame, egg_info):
        """Add visualization elements to the frame."""
        try:
            # Draw egg ellipses and labels
            for egg in egg_info:
                center = egg['position']
                axes = egg['dimensions']
                angle = egg['angle']
                egg_id = egg['id']
                category = egg['category']

                # Set color based on egg quality
                if egg['quality'] == "Good":
                    color = (0, 255, 0)  # Green
                elif egg['quality'] == "Bad":
                    color = (0, 0, 255)  # Red
                else:
                    color = (255, 255, 0)  # Yellow

                # Draw ellipse
                cv2.ellipse(frame, (center, axes, angle), color, 2)

                # Display category information
                info_text = f"#{egg_id} {category}"
                cv2.putText(frame, info_text,
                            (int(center[0] - axes[0] / 2), int(center[1] - axes[1] / 2) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Add depth information
                avg_diameter = (axes[0] + axes[1]) / 2
                depth = self.estimate_depth(avg_diameter)
                if depth is not None:
                    cv2.putText(frame,
                                f"Depth: {depth:.0f}mm",
                                (int(center[0] + 10), int(center[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Draw tracking history
                if self.show_tracks and egg_id in self.tracked_positions:
                    points = list(self.tracked_positions[egg_id])
                    if len(points) > 1:
                        # Use cv2.polylines() for smoother trajectory visualization
                        cv2.polylines(frame, [np.array(points)], isClosed=False, color=color, thickness=2)

            # Add statistics overlay
            overlay = frame.copy()
            stats_height = 180
            cv2.rectangle(overlay, (0, 0), (300, stats_height), (0, 0, 0), -1)

            alpha = 0.7
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Display counts by category
            cv2.putText(frame, f"Total Eggs: {len(egg_info)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame, f"Large Good: {self.large_good_count}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(frame, f"Large Bad: {self.large_bad_count}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(frame, f"Medium Good: {self.medium_good_count}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(frame, f"Medium Bad: {self.medium_bad_count}",
                        (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Add FPS counter
            cv2.putText(frame, f"FPS: {self.fps:.1f}",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            return frame

        except Exception as e:
            logging.error(f"Error in visualization: {e}")
            return frame  # Return original frame on error

    def run_detection(self):
        """Run the egg detection system."""
        if self.camera is None:
            if not self.connect_camera():
                print("Failed to connect to camera")
                return

        try:
            while True:
                # Get frame based on camera type
                if self.camera == "picamera2":
                    # For picamera2
                    frame = self.picam2.capture_array()
                    # Convert BGR to RGB if necessary
                    if frame.shape[2] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    ret = True
                elif self.camera == "picamera":
                    # For older picamera
                    self.picam.capture(self.raw_capture, format="bgr")
                    frame = self.raw_capture.array
                    ret = True
                    # Clear the stream for the next frame
                    self.raw_capture.truncate(0)
                    self.raw_capture.seek(0)
                else:
                    # For OpenCV camera
                    print(self.camera)
                    ret, frame = self.camera.read()
                    print(ret, frame)

                if not ret or frame is None:
                    print("Error: Could not read frame from camera")
                    # time.sleep(1)
                    continue

                # Process frame at reduced resolution for Pi
                start_time = time.time()
                result_frame, _ = self.process_frame(frame)
                process_time = time.time() - start_time

                # Print FPS every second
                if int(time.time()) % 3 == 0:
                    print(f"Processing FPS: {1.0 / process_time:.1f}")

                if result_frame is not None:
                    cv2.imshow("Egg Detection", result_frame)

                # Break if 'q' is pressed - longer wait time for Pi
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

                # Use a more substantial delay on Pi to prevent overheating
                time.sleep(0.05)

        except Exception as e:
            print(f"Error in detection loop: {e}")
        finally:
            # Clean up based on camera type
            if self.camera == "picamera2":
                self.picam2.stop()
            elif self.camera == "picamera":
                self.picam.close()
            elif self.camera is not None and self.camera != "picamera" and self.camera != "picamera2":
                self.camera.release()

            cv2.destroyAllWindows()
            print("Detection system shutdown")


def main():
    """Main entry point for the application."""
    try:
        detector = EggDetector()
        detector.run_detection()
    except Exception as e:
        logging.critical(f"Critical error in main: {e}")
        return 1
    return 0


if __name__ == "__main__":
    main()



#
#
#
#
# 1. Update your Raspberry Pi OS
# bashsudo apt update
# sudo apt upgrade -y
# 2. Install Required Packages
# bash# Install Python and pip if not already installed
# sudo apt install python3 python3-pip -y
#
# # Install OpenCV dependencies
# sudo apt install libopencv-dev -y
# sudo apt install python3-opencv -y
#
# # Install NumPy and other dependencies
# sudo pip3 install numpy
#
# # For PiCamera (if using Raspberry Pi Camera Module)
# sudo apt install python3-picamera -y
# # For newer Raspberry Pi OS with picamera2
# sudo apt install python3-picamera2 -y
# 3. Enable the Camera
# If using the Raspberry Pi Camera Module:
# bashsudo raspi-config
# Navigate to "Interface Options" > "Camera" and enable it. Reboot after enabling.
# 4. Optimize the Raspberry Pi for Performance
# Increase swap space (helps with memory-intensive operations)
# bashsudo nano /etc/dphys-swapfile
# Change the CONF_SWAPSIZE from 100 to 1024:
# CONF_SWAPSIZE=1024
# Save and exit the editor (Ctrl+X, then Y, then Enter). Apply the changes:
# bashsudo /etc/init.d/dphys-swapfile restart
# Set CPU Governor to Performance Mode
# Create a startup script:
# bashsudo nano /etc/rc.local
# Before the exit 0 line, add:
# bashecho performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_g