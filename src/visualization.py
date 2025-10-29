"""
Visualization Module
Creates annotated images with bounding boxes, centroids, and labels
"""

import cv2
import numpy as np
import math


class ResultVisualizer:
    """Visualize detection results with annotations"""
    
    def __init__(self, bbox_color=(0, 255, 0), centroid_color=(0, 0, 255),
                 text_color=(255, 0, 0), orientation_color=(255, 0, 255)):
        """
        Initialize visualizer with color scheme
        
        Args:
            bbox_color (tuple): BGR color for bounding boxes
            centroid_color (tuple): BGR color for centroids
            text_color (tuple): BGR color for text labels
            orientation_color (tuple): BGR color for orientation lines
        """
        self.bbox_color = bbox_color
        self.centroid_color = centroid_color
        self.text_color = text_color
        self.orientation_color = orientation_color
        
    def draw_bounding_box(self, image, bbox, thickness=2):
        """
        Draw bounding box on image
        
        Args:
            image (numpy.ndarray): Image to draw on
            bbox (tuple): (x, y, width, height)
            thickness (int): Line thickness
        """
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), 
                     self.bbox_color, thickness)
    
    def draw_centroid(self, image, centroid, radius=5):
        """
        Draw centroid point
        
        Args:
            image (numpy.ndarray): Image to draw on
            centroid (tuple): (x, y) coordinates
            radius (int): Circle radius
        """
        cx, cy = centroid
        cv2.circle(image, (int(cx), int(cy)), radius, 
                  self.centroid_color, -1)
    
    def draw_orientation(self, image, centroid, angle, length=30, thickness=2):
        """
        Draw orientation line
        
        Args:
            image (numpy.ndarray): Image to draw on
            centroid (tuple): (x, y) coordinates
            angle (float): Orientation angle in degrees
            length (int): Line length in pixels
            thickness (int): Line thickness
        """
        cx, cy = centroid
        angle_rad = math.radians(angle)
        
        end_x = int(cx + length * math.cos(angle_rad))
        end_y = int(cy + length * math.sin(angle_rad))
        
        cv2.line(image, (int(cx), int(cy)), (end_x, end_y),
                self.orientation_color, thickness)
    
    def draw_label(self, image, position, text, font_scale=0.5, thickness=2):
        """
        Draw text label
        
        Args:
            image (numpy.ndarray): Image to draw on
            position (tuple): (x, y) coordinates for text
            text (str): Label text
            font_scale (float): Font size scale
            thickness (int): Text thickness
        """
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, self.text_color, thickness)
    
    def create_annotated_image(self, original_image, objects):
        """
        Create fully annotated image with all visualizations
        
        Args:
            original_image (numpy.ndarray): Original input image
            objects (list): List of object feature dictionaries
            
        Returns:
            numpy.ndarray: Annotated image
        """
        annotated = original_image.copy()
        
        for obj in objects:
            # Draw bounding box
            self.draw_bounding_box(annotated, obj['bbox'])
            
            # Draw centroid
            self.draw_centroid(annotated, obj['centroid'])
            
            # Draw orientation
            self.draw_orientation(annotated, obj['centroid'], 
                                obj['orientation'])
            
            # Draw ID label
            x, y, _, _ = obj['bbox']
            label = f"ID:{obj['id']}"
            self.draw_label(annotated, (x, y - 10), label)
        
        print(f"✓ Created annotated visualization with {len(objects)} objects")
        return annotated
    
    def create_info_overlay(self, image, objects):
        """
        Create image with detailed information overlay
        
        Args:
            image (numpy.ndarray): Base image
            objects (list): List of object feature dictionaries
            
        Returns:
            numpy.ndarray: Image with info overlay
        """
        overlay = image.copy()
        
        # Add summary at top
        summary = f"Objects Detected: {len(objects)}"
        cv2.putText(overlay, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return overlay
    
    def display_results(self, images_dict, window_prefix="Result"):
        """
        Display multiple images in separate windows
        
        Args:
            images_dict (dict): Dictionary of {name: image}
            window_prefix (str): Prefix for window names
        """
        for name, image in images_dict.items():
            cv2.imshow(f"{window_prefix} - {name}", image)
        
        print("Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def save_results(self, images_dict, output_dir='output'):
        """
        Save multiple images to disk
        
        Args:
            images_dict (dict): Dictionary of {name: image}
            output_dir (str): Output directory path
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, image in images_dict.items():
            filepath = os.path.join(output_dir, f"{name}.jpg")
            cv2.imwrite(filepath, image)
            print(f"✓ Saved: {filepath}")
