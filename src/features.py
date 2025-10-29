"""
Feature Extraction Module
Calculates geometric properties: centroid, orientation, dimensions
"""

import cv2
import numpy as np
import math


class FeatureExtractor:
    """Extract geometric features from segmented objects"""
    
    def __init__(self):
        """Initialize feature extractor"""
        pass
    
    def calculate_orientation(self, moments):
        """
        Calculate object orientation using second-order moments
        
        Args:
            moments (dict): OpenCV moments dictionary
            
        Returns:
            float: Orientation angle in degrees (-90° to +90°)
        """
        mu20 = moments['mu20']
        mu02 = moments['mu02']
        mu11 = moments['mu11']
        
        # Handle division by zero
        if (mu20 - mu02) == 0:
            angle = 0.0
        else:
            # θ = 0.5 * arctan(2*μ11 / (μ20 - μ02))
            angle = 0.5 * math.atan2(2 * mu11, (mu20 - mu02))
        
        # Convert to degrees
        angle_degrees = math.degrees(angle)
        
        return angle_degrees
    
    def calculate_orientation_from_mask(self, mask):
        """
        Calculate orientation from binary mask
        
        Args:
            mask (numpy.ndarray): Binary mask of object
            
        Returns:
            float: Orientation angle in degrees
        """
        moments = cv2.moments(mask, binaryImage=True)
        return self.calculate_orientation(moments)
    
    def extract_from_connected_components(self, segmentation_result, label_map):
        """
        Extract features from connected component segmentation
        
        Args:
            segmentation_result (dict): Output from ObjectSegmenter
            label_map (numpy.ndarray): Label map
            
        Returns:
            list: List of feature dictionaries for each object
        """
        objects_with_features = []
        
        for obj in segmentation_result['objects']:
            # Create mask for this object
            mask = np.uint8(label_map == obj['label_id']) * 255
            
            # Calculate moments
            moments = cv2.moments(mask, binaryImage=True)
            
            # Calculate orientation
            orientation = self.calculate_orientation(moments)
            
            # Compile features
            features = {
                'id': obj['label_id'],
                'centroid': (obj['centroid_x'], obj['centroid_y']),
                'area': obj['area'],
                'width': obj['width'],
                'height': obj['height'],
                'orientation': orientation,
                'bbox': (obj['left'], obj['top'], obj['width'], obj['height']),
                'aspect_ratio': obj['width'] / obj['height'] if obj['height'] > 0 else 0
            }
            
            objects_with_features.append(features)
        
        print(f"✓ Extracted features for {len(objects_with_features)} objects")
        return objects_with_features
    
    def extract_from_contours(self, segmentation_result):
        """
        Extract features from contour segmentation
        
        Args:
            segmentation_result (dict): Output from ObjectSegmenter
            
        Returns:
            list: List of feature dictionaries for each object
        """
        objects_with_features = []
        
        for obj in segmentation_result['objects']:
            # Calculate orientation
            orientation = self.calculate_orientation(obj['moments'])
            
            # Compile features
            features = {
                'id': obj['contour_id'] + 1,  # Start from 1
                'centroid': (obj['centroid_x'], obj['centroid_y']),
                'area': obj['area'],
                'width': obj['width'],
                'height': obj['height'],
                'orientation': orientation,
                'bbox': (obj['left'], obj['top'], obj['width'], obj['height']),
                'aspect_ratio': obj['width'] / obj['height'] if obj['height'] > 0 else 0,
                'perimeter': cv2.arcLength(obj['contour'], True)
            }
            
            objects_with_features.append(features)
        
        print(f"✓ Extracted features for {len(objects_with_features)} objects")
        return objects_with_features
