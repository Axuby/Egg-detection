import cv2
import numpy as np

class EggDetector:
    def __init__(self):

        # This's Good eggs range - Satin Blossom White (convert to HSV range)
        self.good_egg_lower = np.array([0, 0, 220])  # Very light white
        self.good_egg_upper = np.array([180, 30, 255])  
        
        # Bad eggs - Satin Ivory Silk (convert to HSV range)
        self.bad_egg_lower = np.array([20, 20, 200])  # Ivory color
        self.bad_egg_upper = np.array([40, 60, 255])  # Yellowish white
        
    def detect_eggs(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # This will create masks for good and bad eggs
        good_mask = cv2.inRange(hsv, self.good_egg_lower, self.good_egg_upper)
        bad_mask = cv2.inRange(hsv, self.bad_egg_lower, self.bad_egg_upper)
        
        # Finding their contours
        good_contours, _ = cv2.findContours(good_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bad_contours, _ = cv2.findContours(bad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = {
            'good_eggs': [],
            'bad_eggs': [],
            'visualization': frame.copy()
        }
        
        # I'm Processing good eggs
        for contour in good_contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                results['good_eggs'].append({'position': (x, y), 'size': (w, h)})
                cv2.rectangle(results['visualization'], (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(results['visualization'], 'Good', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # This is Processing bad eggs
        for contour in bad_contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                results['bad_eggs'].append({'position': (x, y), 'size': (w, h)})
                cv2.rectangle(results['visualization'], (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(results['visualization'], 'Bad', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return results

def main():
    cap = cv2.VideoCapture(1)  # 0 for built-in camera, or change for USB camera
    detector = EggDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        results = detector.detect_eggs(frame)
        
        # Display results
        cv2.imshow('Egg Detection', results['visualization'])
        
        # Print counts
        print(f"Good eggs detected: {len(results['good_eggs'])}")
        print(f"Bad eggs detected: {len(results['bad_eggs'])}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


main()