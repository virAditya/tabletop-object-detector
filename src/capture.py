"""
Image Acquisition Module
Handles camera connection and image capture from DroidCam
"""

import cv2
import numpy as np
import time


class ImageCapture:
    """Capture images from DroidCam or other camera sources"""
    
    def __init__(self, camera_source=1):
        """
        Initialize camera capture
        
        Args:
            camera_source: Can be int (0, 1, 2) or string (DroidCam URL)
        """
        self.camera_source = camera_source
        
    def capture_single_frame(self):
        """
        Capture a single frame from the camera
        
        Returns:
            numpy.ndarray: Captured BGR image
            
        Raises:
            RuntimeError: If camera cannot be opened or frame capture fails
        """
        cap = cv2.VideoCapture(self.camera_source)
        
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at source {self.camera_source}. "
                "Ensure DroidCam is running and connected."
            )
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")
        
        print(f"✓ Image captured: {frame.shape[1]}x{frame.shape[0]} pixels")
        return frame
    
    def capture_with_countdown(self, countdown_seconds=3, crop_top=0):
        """
        Capture image after countdown with live preview
        
        Args:
            countdown_seconds (int): Seconds to wait before capture
            crop_top (int): Pixels to crop from top (removes DroidCam overlay)
            
        Returns:
            numpy.ndarray: Captured image
        """
        cap = cv2.VideoCapture(self.camera_source)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera at source {self.camera_source}")
        
        print(f"Starting {countdown_seconds}-second countdown...")
        print("Position your objects now!")
        
        start_time = time.time()
        captured_frame = None
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("⚠ Failed to read frame")
                break
            
            # Calculate remaining time
            elapsed = time.time() - start_time
            remaining = countdown_seconds - elapsed
            
            if remaining <= 0:
                # Capture!
                captured_frame = frame.copy()
                
                # Crop top pixels to remove overlay
                if crop_top > 0:
                    captured_frame = captured_frame[crop_top:, :]
                    print(f"✓ Image captured: {captured_frame.shape[1]}x{captured_frame.shape[0]} pixels")
                    print(f"  (Cropped top {crop_top}px to remove overlay)")
                else:
                    print(f"✓ Image captured: {captured_frame.shape[1]}x{captured_frame.shape[0]} pixels")
                break
            
            # Draw countdown on frame
            display = frame.copy()
            countdown_text = f"Capturing in: {int(remaining) + 1}"
            
            # Add semi-transparent overlay
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (display.shape[1], 100), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
            
            # Add countdown text
            cv2.putText(display, countdown_text, 
                       (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       2, (0, 255, 0), 3)
            
            # Add instruction
            cv2.putText(display, "Adjust your view!", 
                       (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            cv2.imshow('Camera Preview - Countdown', display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Capture cancelled by user")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return captured_frame
    
    def capture_with_preview(self, window_name="Camera Preview"):
        """
        Capture image with live preview (press SPACE to capture, ESC to exit)
        
        Args:
            window_name (str): Name of preview window
            
        Returns:
            numpy.ndarray: Captured image
        """
        cap = cv2.VideoCapture(self.camera_source)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera at source {self.camera_source}")
        
        print("Camera preview started. Press SPACE to capture, ESC to exit.")
        
        captured_frame = None
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Display instructions on frame
            cv2.putText(frame, "Press SPACE to capture | ESC to exit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACE key
                captured_frame = frame.copy()
                print("✓ Frame captured!")
                break
            elif key == 27:  # ESC key
                print("Capture cancelled")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return captured_frame
    
    def validate_image(self, image):
        """
        Validate captured image
        
        Args:
            image (numpy.ndarray): Image to validate
            
        Returns:
            bool: True if image is valid
        """
        if image is None:
            return False
        
        if len(image.shape) != 3:
            return False
        
        if image.shape[2] != 3:
            return False
        
        return True
