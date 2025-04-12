import cv2
import numpy as np
import time
from collections import deque


class EggDetector:
    def __init__(self, config_file=None):
        # color ranges for the Good eggs range - Satin Blossom White (HSV range)
        self.good_egg_lower = np.array([0, 0, 220])
        self.good_egg_upper = np.array([180, 30, 255])

        # Bad eggs range - Satin Ivory Silk (HSV range)
        self.bad_egg_lower = np.array([20, 20, 200])
        self.bad_egg_upper = np.array([40, 60, 255])


        self.min_area = 500


        self.camera = None
        self.camera_index = 1


        self.resize_width = 1280
        self.blur_kernel_size = 5
        self.morph_kernel_size = 5
        self.show_tracks = True


        self.tracked_eggs = {}
        self.next_egg_id = 1
        self.track_buffer = 10
        self.tracked_positions = {}
        self.max_track_history = 20


        self.frame_times = deque(maxlen=30)




        self.confidence_threshold = 0.5  # Minimum confidence to classify




    def select_camera(self):
        available_cameras = []

        # Check first 10 camera indices
        for i in range(3):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available_cameras.append(i)
                    cap.release()
            except Exception as e:
                print(e)

        print(available_cameras, "Available cameras")

        if not available_cameras:
            print("No cameras detected!")
            return False


        print("Available cameras:")
        for i, cam_idx in enumerate(available_cameras):
            print(f"{i + 1}: Camera index {cam_idx}")

        # Get user selection
        try:
            selection = int(input(f"Select camera (1-{len(available_cameras)}): "))
            if 1 <= selection <= len(available_cameras):
                self.camera_index = available_cameras[selection - 1]
                return self.connect_camera(self.camera_index)
            else:
                print("Invalid selection!")
                return False
        except ValueError:
            print("Please enter a number!")
            return False

    def connect_camera(self, camera_index=None):

        if camera_index is not None:
            self.camera_index = camera_index

        try:
            # Release existing camera if any
            if self.camera is not None:
                self.camera.release()

            self.camera = cv2.VideoCapture(self.camera_index)

            if not self.camera.isOpened():
                print(f"Error: Could not open camera at index {self.camera_index}")
                return False

            # Set camera properties for better quality if needed
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            print(f"Connected to camera at index {self.camera_index}")
            return True

        except Exception as e:
            print(f"Error connecting to camera: {e}")
            return False

    def track_eggs(self, egg_info):

        updated_tracks = {}
        current_positions = {}

        for egg in egg_info:
            x, y = egg['position']
            found_match = False

            for prev_id, (px, py) in self.tracked_eggs.items():
                if abs(px - x) < 50 and abs(py - y) < 50:  # Increased from 40 to 50 for better tracking
                    updated_tracks[prev_id] = (x, y)
                    current_positions[prev_id] = (x, y)
                    egg['id'] = prev_id
                    found_match = True
                    break

            # Create new track if no match found
            if not found_match:
                updated_tracks[self.next_egg_id] = (x, y)
                current_positions[self.next_egg_id] = (x, y)
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

    def calibrate_colors(self, frame, roi_good=None, roi_bad=None):
        """Calibrate color ranges from sample regions in the frame."""
        if frame is None:
            print("Cannot calibrate: No frame provided")
            return False

        if roi_good is not None:
            try:
                # Extract HSV values from good egg sample region
                x1, y1, x2, y2 = roi_good
                sample = frame[y1:y2, x1:x2]
                if sample.size > 0:
                    hsv_good = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)

                    # Calculate min/max values with some margin
                    h_min = max(0, np.percentile(hsv_good[:, :, 0], 5) - 10)
                    h_max = min(180, np.percentile(hsv_good[:, :, 0], 95) + 10)
                    s_min = max(0, np.percentile(hsv_good[:, :, 1], 5) - 10)
                    s_max = min(255, np.percentile(hsv_good[:, :, 1], 95) + 10)
                    v_min = max(0, np.percentile(hsv_good[:, :, 2], 5) - 10)
                    v_max = min(255, np.percentile(hsv_good[:, :, 2], 95) + 10)

                    self.good_egg_lower = np.array([h_min, s_min, v_min])
                    self.good_egg_upper = np.array([h_max, s_max, v_max])

                    print(f"Good egg HSV range calibrated: {self.good_egg_lower} to {self.good_egg_upper}")
            except Exception as e:
                print(f"Error calibrating good egg color: {e}")
                return False

        if roi_bad is not None:
            try:
                # Extract HSV values from bad egg sample region
                x1, y1, x2, y2 = roi_bad
                sample = frame[y1:y2, x1:x2]
                if sample.size > 0:
                    hsv_bad = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)

                    # Calculate min/max values with some margin
                    h_min = max(0, np.percentile(hsv_bad[:, :, 0], 5) - 10)
                    h_max = min(180, np.percentile(hsv_bad[:, :, 0], 95) + 10)
                    s_min = max(0, np.percentile(hsv_bad[:, :, 1], 5) - 10)
                    s_max = min(255, np.percentile(hsv_bad[:, :, 1], 95) + 10)
                    v_min = max(0, np.percentile(hsv_bad[:, :, 2], 5) - 10)
                    v_max = min(255, np.percentile(hsv_bad[:, :, 2], 95) + 10)

                    self.bad_egg_lower = np.array([h_min, s_min, v_min])
                    self.bad_egg_upper = np.array([h_max, s_max, v_max])

                    print(f"Bad egg HSV range calibrated: {self.bad_egg_lower} to {self.bad_egg_upper}")
            except Exception as e:
                print(f"Error calibrating bad egg color: {e}")
                return False

        return True

    def optimize_frame(self, frame):
        """Resize frame for faster processing if needed."""
        if frame is None:
            return None

        try:
            height, width = frame.shape[:2]


            if width > self.resize_width:
                scale = self.resize_width / width
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

            return frame

        except Exception as e:
            print(f"Error optimizing frame: {e}")
            return frame  # Return original frame on error



    def process_frame(self, frame):

        start_time = time.time()
        result_frame = None
        egg_info = []

        try:
            if frame is None or frame.size == 0:
                return None, None

            # Optimize frame size for processing
            frame = self.optimize_frame(frame)
            if frame is None:
                return None, None

            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Apply color thresholding for both good and bad eggs
            mask_good = cv2.inRange(hsv, self.good_egg_lower, self.good_egg_upper)
            mask_bad = cv2.inRange(hsv, self.bad_egg_lower, self.bad_egg_upper)
            mask = cv2.bitwise_or(mask_good, mask_bad)

            # Reduce noise - Enhanced
            mask = cv2.GaussianBlur(mask, (self.blur_kernel_size, self.blur_kernel_size), 0)
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create result frame and gather egg info
            result_frame = frame.copy()
            egg_count = 0
            good_count = 0
            bad_count = 0

            for contour in contours:
                area = cv2.contourArea(contour)

                if area > self.min_area:
                    # Fit an ellipse if there are enough points
                    if len(contour) >= 5:
                        ellipse = cv2.fitEllipse(contour)

                        # Calculate dimensions and aspect ratio
                        (center, axes, angle) = ellipse
                        major_axis = max(axes)
                        minor_axis = min(axes)
                        aspect_ratio = minor_axis / major_axis

                        # Calculate confidence
                        x, y, w, h = cv2.boundingRect(contour)
                        mask_segment = np.zeros_like(mask)
                        cv2.drawContours(mask_segment, [contour], -1, 255, -1)

                        good_match = cv2.countNonZero(cv2.bitwise_and(mask_segment, mask_good))
                        bad_match = cv2.countNonZero(cv2.bitwise_and(mask_segment, mask_bad))

                        total_match = good_match + bad_match

                        # Confidence calculation
                        good_confidence = good_match / total_match if total_match > 0 else 0
                        bad_confidence = bad_match / total_match if total_match > 0 else 0

                        # Determine egg type based on higher confidence
                        if good_confidence >= bad_confidence and good_confidence >= self.confidence_threshold:
                            egg_type = "Good Egg"
                            confidence = good_confidence
                            good_count += 1
                            color = (0, 255, 0)  # Green for good eggs
                        elif bad_confidence > good_confidence and bad_confidence >= self.confidence_threshold:
                            egg_type = "Bad Egg"
                            confidence = bad_confidence
                            bad_count += 1
                            color = (0, 0, 255)  # Red for bad eggs
                        else:
                            egg_type = "Uncertain"
                            confidence = max(good_confidence, bad_confidence)
                            color = (255, 255, 0)  # Yellow for uncertain

                        # Draw the ellipse
                        cv2.ellipse(result_frame, ellipse, color, 2)

                        # Store egg information
                        egg_info.append({
                            'id': 0,  # Will be assigned in track_eggs
                            'type': egg_type,
                            'position': tuple(map(int, center)),
                            'dimensions': axes,
                            'angle': angle,
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'confidence': confidence,
                            'good_confidence': good_confidence,
                            'bad_confidence': bad_confidence
                        })

                        egg_count += 1

            # Track eggs across frames
            self.track_eggs(egg_info)

            # Visualize results
            result_frame = self.visualize_results(result_frame, egg_info, good_count, bad_count)

            # Calculate and store processing time - FPS calculation
            process_time = time.time() - start_time

            # Update frame processing times
            # If frame_times has reached its max length, remove the oldest time
            if len(self.frame_times) >= 30:
                self.frame_times.popleft()
            self.frame_times.append(process_time)

            # Calculate FPS - average of last 30 frame processing times
            # Protect against division by zero
            self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0

            return result_frame, egg_info

        except Exception as e:
            print(f"Error processing frame: {e}")
            if frame is not None:
                return frame, []  # Return original frame on error
            return None, []

    def visualize_results(self, frame, egg_info, good_count, bad_count):
        """Enhanced visualization with confidence levels."""
        try:

            total_eggs = len(egg_info) if egg_info else 1

            # Draw egg ellipses and labels
            for egg in egg_info:
                center = egg['position']
                axes = egg['dimensions']
                angle = egg['angle']
                egg_id = egg['id']

                # Set color based on egg type
                if egg['type'] == "Good Egg":
                    color = (0, 255, 0)
                elif egg['type'] == "Bad Egg":
                    color = (0, 0, 255)
                else:
                    color = (255, 255, 0)  # Yellow

                # Draw ellipse
                cv2.ellipse(frame, (center, axes, angle), color, 2)

                # Display detailed information
                info_text = (f"#{egg_id} {egg['type']} "
                             f"Conf: {egg['confidence']:.2f} "
                             # f":{egg['good_confidence']:.2f} "
                             # f"B:{egg['bad_confidence']:.2f}"
                             )

                cv2.putText(frame, info_text,
                            (int(center[0] - axes[0] / 2), int(center[1] - axes[1] / 2) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


                if self.show_tracks and egg_id in self.tracked_positions:
                    points = list(self.tracked_positions[egg_id])
                    if len(points) > 1:
                        for i in range(1, len(points)):
                            if points[i - 1] is not None and points[i] is not None:
                                cv2.line(frame, points[i - 1], points[i], color, 2)

            # Add statistics overlay
            overlay = frame.copy()
            stats_height = 150
            cv2.rectangle(overlay, (0, 0), (300, stats_height), (0, 0, 0), -1)


            alpha = 0.7
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


            cv2.putText(frame, f"Total Eggs: {len(egg_info)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Good: {good_count} ({good_count / total_eggs * 100:.1f}%)",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Bad: {bad_count} ({bad_count / total_eggs * 100:.1f}%)",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {self.fps:.1f}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            return frame

        except Exception as e:
            print(f"Error in visualization: {e}")
            return frame  # Return original frame on error

    def run_live_detection(self):


        if self.camera is None:
            if not self.select_camera():
                print("Failed to connect to camera")
                return

        calibration_mode = False
        roi_points = []
        calibration_type = None


        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    time.sleep(1)
                    continue

                # Handle calibration mode
                if calibration_mode:
                    temp_frame = frame.copy()


                    cv2.putText(temp_frame, "CALIBRATION MODE", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(temp_frame, f"Calibrating: {calibration_type}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(temp_frame, "Click to select region corners", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Draw selected points
                    for i, (x, y) in enumerate(roi_points):
                        cv2.circle(temp_frame, (x, y), 5, (0, 255, 255), -1)
                        if i > 0:  # Draw lines between points
                            cv2.line(temp_frame, roi_points[i - 1], (x, y), (0, 255, 255), 2)

                    # If we have 4 points, draw ROI
                    if len(roi_points) == 4:
                        cv2.polylines(temp_frame, [np.array(roi_points)], True, (0, 255, 255), 2)
                        cv2.putText(temp_frame, "Press ENTER to confirm", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    cv2.imshow("Egg Detection", temp_frame)
                else:
                    # Normal processing mode
                    result_frame, _ = self.process_frame(frame)
                    if result_frame is not None:
                        cv2.imshow("Egg Detection", result_frame)


                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                elif key == ord('c'):
                    # Toggle calibration mode
                    calibration_mode = not calibration_mode
                    roi_points = []
                    if calibration_mode:
                        # Ask for calibration type
                        print("\nEnter calibration type:")
                        print("1: Good Egg")
                        print("2: Bad Egg")
                        choice = input("Select (1/2): ")
                        calibration_type = "Good Egg" if choice == "1" else "Bad Egg"
                        print(f"Click to select 4 corners around a {calibration_type}")

                        # Define mouse callback function for ROI selection
                        def mouse_callback(event, x, y, flags, param):
                            if event == cv2.EVENT_LBUTTONDOWN:
                                if len(roi_points) < 4:
                                    roi_points.append((x, y))

                        cv2.setMouseCallback("Egg Detection", mouse_callback)
                    else:
                        cv2.setMouseCallback("Egg Detection", lambda *args: None)

                elif key == 13 and calibration_mode and len(roi_points) == 4:  # Enter key
                    # Process calibration
                    x_min = min(p[0] for p in roi_points)
                    y_min = min(p[1] for p in roi_points)
                    x_max = max(p[0] for p in roi_points)
                    y_max = max(p[1] for p in roi_points)

                    roi = (x_min, y_min, x_max, y_max)

                    if calibration_type == "Good Egg":
                        self.calibrate_colors(frame, roi_good=roi)
                    else:
                        self.calibrate_colors(frame, roi_bad=roi)

                    calibration_mode = False
                    cv2.setMouseCallback("Egg Detection", lambda *args: None)

                elif key == ord('s'):

                    self.save_config()

                elif key == ord('t'):
                    # Toggle tracking visualization
                    self.show_tracks = not self.show_tracks
                    print(f"Tracking visualization: {'ON' if self.show_tracks else 'OFF'}")

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nDetection stopped by user")
        except Exception as e:
            print(f"Error in detection loop: {e}")
        finally:
            if self.camera is not None:
                self.camera.release()
            cv2.destroyAllWindows()
            print("Detection system shutdown")


def main():

    try:

        detector = EggDetector()

        detector.run_live_detection()

    except Exception as e:
        print(f"Critical error in main: {e}")
        return 1

    return 0


main()
