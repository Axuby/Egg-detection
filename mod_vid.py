import cv2
import numpy as np
import time
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s: %(message)s')


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

        self.resize_width = 1280
        self.blur_kernel_size = 5
        self.morph_kernel_size = 5
        self.show_tracks = True

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

            self.camera = cv2.VideoCapture(self.camera_index)

            if not self.camera.isOpened():
                logging.error(f"Could not open camera at index {self.camera_index}")
                return False

            # Set camera properties for better quality
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            print(f"Connected to camera at index {self.camera_index}")
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
                ret, frame = self.camera.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    time.sleep(1)
                    continue

                # Process frame
                result_frame, _ = self.process_frame(frame)

                if result_frame is not None:
                    cv2.imshow("Egg Detection", result_frame)

                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(0.01)  # Small delay to prevent high CPU usage

        except Exception as e:
            print(f"Error in detection loop: {e}")
        finally:
            if self.camera is not None:
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