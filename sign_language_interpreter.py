import cv2
import numpy as np
import os
import time

class SignLanguageInterpreter:
    def __init__(self):
        self.data_dir = "./data"
        self.signs = {}  # Dictionary to store sign templates and their meanings
        self.setup_data_directory()
        self.load_existing_signs()
        
    def setup_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data directory at {self.data_dir}")
    
    def load_existing_signs(self):
        """Load existing sign templates from data directory"""
        if not os.path.exists(self.data_dir):
            print("Data directory not found")
            return
        
        sign_count = 0
        print("\nLoading existing signs...")
        
        # Look for sign directories
        for dirname in os.listdir(self.data_dir):
            dir_path = os.path.join(self.data_dir, dirname)
            if os.path.isdir(dir_path) and not dirname.startswith('.'):
                # Get all frames for this sign
                frames = []
                for frame_file in os.listdir(dir_path):
                    if frame_file.endswith(('.jpg', '.png')):
                        frame_path = os.path.join(dir_path, frame_file)
                        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                        if frame is not None:
                            frames.append(frame)
                
                if frames:
                    # Use the first frame as template (or you could compute an average)
                    self.signs[dirname] = frames[0]
                    sign_count += 1
                    print(f"Loaded sign '{dirname}' with {len(frames)} frames")
        
        if sign_count > 0:
            print(f"\nSuccessfully loaded {sign_count} signs")
        else:
            print("No signs found in the data directory")
    
    def capture_training_data(self):
        """Capture multiple samples of a sign for training"""
        sign_name = input("Enter the name of the sign you want to add (e.g., 'hello'): ").lower()
        
        # Create a subdirectory for this sign
        sign_dir = os.path.join(self.data_dir, sign_name)
        if not os.path.exists(sign_dir):
            os.makedirs(sign_dir)
        
        print("\nControls:")
        print("'c' - Capture current frame")
        print("'d' - Done capturing")
        print("'q' - Quit without saving")
        
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("Could not open camera")
            return
        
        frame_count = 0
        frames = []
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            frame = cv2.flip(frame, 1)
            
            # Process frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            
            # Draw rectangle and add text
            cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
            cv2.putText(frame, f"Sign: {sign_name} | Frames: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Capture Sign", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Capture and save the frame
                roi = blurred[100:300, 100:300]
                filepath = os.path.join(sign_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(filepath, roi)
                frames.append(roi)  # Store frame in list
                frame_count += 1
                print(f"Captured frame {frame_count}")
                
            elif key == ord('d'):
                if frame_count > 0:
                    # Use average of all frames as template
                    average_template = np.mean(frames, axis=0).astype(np.uint8)
                    self.signs[sign_name] = average_template
                    print(f"\nSaved {frame_count} frames for sign '{sign_name}'")
                    break
                else:
                    print("No frames captured yet")
                
            elif key == ord('q'):
                print("\nQuitting without saving")
                break
        
        cam.release()
        cv2.destroyAllWindows()
        return frame_count > 0
    
    def process_frame(self, frame):
        """Process frame to extract hand region"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Add histogram equalization to normalize brightness
        equalized = cv2.equalizeHist(blurred)
        
        # Add thresholding to better segment the hand
        _, thresh = cv2.threshold(equalized, 127, 255, cv2.THRESH_BINARY)
        
        return thresh
    
    def match_sign(self, frame):
        """Match the current frame against known sign templates"""
        best_match = None
        best_score = float('inf')
        
        # Extract region of interest
        roi = frame[100:300, 100:300]
        
        try:
            for sign_name, template in self.signs.items():
                # Ensure template is not None
                if template is None:
                    continue
                
                # Resize template to match ROI size
                template_resized = cv2.resize(template, (roi.shape[1], roi.shape[0]))
                
                # Calculate difference between ROI and template
                diff = cv2.absdiff(roi, template_resized)
                score = np.mean(diff)
                
                # Add debug printing
                print(f"Sign: {sign_name}, Score: {score}")
                
                if score < best_score:
                    best_score = score
                    best_match = sign_name
            
            # The threshold of 50 might be too strict
            if best_score < 50:  # Adjust this threshold as needed
                return best_match, best_score
            return None, None
            
        except Exception as e:
            print(f"Error in matching: {str(e)}")
            return None, None
    
    def run(self):
        """Main loop for sign language interpretation"""
        while True:
            print("\nMain Menu:")
            print(f"Signs loaded: {len(self.signs)}")  # Add this line to show number of loaded signs
            print("1. Add new sign")
            print("2. Start recognition")
            print("3. List loaded signs")  # Add this option
            print("4. Exit")
            
            choice = input("Enter your choice (1-4): ")
            
            if choice == '1':
                self.capture_training_data()
                
            elif choice == '2':
                if len(self.signs) == 0:
                    print("No signs in database. Please add some signs first.")
                    continue
                
                print("\nStarting recognition mode...")
                print(f"Loaded signs: {', '.join(self.signs.keys())}")  # Show available signs
                print("Press 'q' to return to main menu")
                
                cam = cv2.VideoCapture(0)
                if not cam.isOpened():
                    print("Could not open camera")
                    continue
                
                while True:
                    ret, frame = cam.read()
                    if not ret:
                        continue
                    
                    frame = cv2.flip(frame, 1)
                    processed = self.process_frame(frame)
                    
                    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
                    
                    matched_sign, score = self.match_sign(processed)
                    if matched_sign:
                        cv2.putText(frame, f"Sign: {matched_sign} ({score:.2f})", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("Sign Language Interpreter", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                cam.release()
                cv2.destroyAllWindows()
                
            elif choice == '3':  # New option to list signs
                if len(self.signs) == 0:
                    print("No signs loaded")
                else:
                    print("\nLoaded signs:")
                    for i, sign in enumerate(self.signs.keys(), 1):
                        print(f"{i}. {sign}")
                
            elif choice == '4':
                print("Exiting program...")
                break
            
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    interpreter = SignLanguageInterpreter()
    interpreter.run() 