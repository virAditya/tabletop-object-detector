"""
Object Segmentation Module
Implements connected component analysis and contour detection with filtering
"""

import cv2
import numpy as np


class ObjectSegmenter:
    """Segment objects from binary images"""
    
    def __init__(self, min_area=500, connectivity=8):
        """
        Initialize segmenter
        
        Args:
            min_area (int): Minimum object area in pixels
            connectivity (int): Pixel connectivity (4 or 8)
        """
        self.min_area = min_area
        self.connectivity = connectivity
        
    def segment_connected_components(self, binary):
        """
        Segment objects using connected component analysis with intelligent filtering
        
        Args:
            binary (numpy.ndarray): Binary image
            
        Returns:
            dict: Segmentation results with labels, stats, and centroids
        """
        from config.config import config
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, 
            connectivity=self.connectivity,
            ltype=cv2.CV_32S
        )
        
        # Get image dimensions for region filtering
        image_height = binary.shape[0]
        exclude_top_pixels = int(image_height * (config.EXCLUDE_TOP_PERCENT / 100))
        
        # Filter objects
        valid_objects = []
        filtered_reasons = []
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            centroid_y = centroids[i][1]
            
            # Filter 1: Minimum area
            if area < self.min_area:
                filtered_reasons.append(f"Object {i}: Too small ({area}px)")
                continue
            
            # Filter 2: Position (top region - DroidCam overlay)
            if config.FILTER_BY_POSITION and centroid_y < exclude_top_pixels:
                filtered_reasons.append(f"Object {i}: In top region (Y={centroid_y:.0f})")
                continue
            
            # Filter 3: Aspect ratio (elongated text-like objects)
            if config.FILTER_BY_ASPECT_RATIO and height > 0:
                aspect_ratio = width / height
                
                if aspect_ratio > config.MAX_ASPECT_RATIO or aspect_ratio < config.MIN_ASPECT_RATIO:
                    filtered_reasons.append(f"Object {i}: Elongated (AR={aspect_ratio:.1f})")
                    continue
            
            # Object passed all filters
            valid_objects.append({
                'label_id': i,
                'area': area,
                'left': stats[i, cv2.CC_STAT_LEFT],
                'top': stats[i, cv2.CC_STAT_TOP],
                'width': width,
                'height': height,
                'centroid_x': centroids[i][0],
                'centroid_y': centroids[i][1]
            })
        
        # Print filtering summary
        total_detected = num_labels - 1  # Exclude background
        num_filtered = total_detected - len(valid_objects)
        
        print(f"✓ Detected {len(valid_objects)} valid objects")
        if num_filtered > 0:
            print(f"  ({num_filtered} objects filtered out)")
            for reason in filtered_reasons[:3]:  # Show first 3 reasons
                print(f"    • {reason}")
        
        return {
            'num_objects': len(valid_objects),
            'objects': valid_objects,
            'label_map': labels,
            'all_stats': stats,
            'all_centroids': centroids
        }
    
    def segment_contours(self, binary):
        """
        Segment objects using contour detection
        
        Args:
            binary (numpy.ndarray): Binary image
            
        Returns:
            dict: Segmentation results with contours and properties
        """
        contours, hierarchy = cv2.findContours(
            binary, 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and extract properties
        valid_objects = []
        
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area < self.min_area:
                continue
            
            # Calculate moments
            M = cv2.moments(contour)
            
            if M['m00'] == 0:
                continue
            
            # Extract properties
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            
            x, y, w, h = cv2.boundingRect(contour)
            
            valid_objects.append({
                'contour_id': idx,
                'contour': contour,
                'area': int(area),
                'left': x,
                'top': y,
                'width': w,
                'height': h,
                'centroid_x': cx,
                'centroid_y': cy,
                'moments': M
            })
        
        print(f"✓ Detected {len(valid_objects)} contours (min area: {self.min_area} px)")
        
        return {
            'num_objects': len(valid_objects),
            'objects': valid_objects,
            'all_contours': contours,
            'hierarchy': hierarchy
        }
    
    def create_object_mask(self, label_map, label_id):
        """
        Create binary mask for specific object
        
        Args:
            label_map (numpy.ndarray): Label map from connected components
            label_id (int): Specific label ID
            
        Returns:
            numpy.ndarray: Binary mask for the object
        """
        mask = np.uint8(label_map == label_id) * 255
        return mask
