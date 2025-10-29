"""
Image Preprocessing Module
Handles grayscale conversion, noise reduction, and binarization
"""

import cv2
import numpy as np


class ImagePreprocessor:
    """Preprocess images for object segmentation"""
    
    def __init__(self, blur_kernel_size=5, morph_kernel_size=3, 
                 morph_iterations=2):
        """
        Initialize preprocessor with parameters
        
        Args:
            blur_kernel_size (int): Gaussian blur kernel size
            morph_kernel_size (int): Morphological operation kernel size
            morph_iterations (int): Number of morphological operation iterations
        """
        self.blur_kernel_size = blur_kernel_size
        self.morph_kernel_size = morph_kernel_size
        self.morph_iterations = morph_iterations
        
    def convert_to_grayscale(self, image):
        """
        Convert BGR image to grayscale
        
        Args:
            image (numpy.ndarray): Input BGR image
            
        Returns:
            numpy.ndarray: Grayscale image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray
    
    def apply_blur(self, image):
        """
        Apply Gaussian blur for noise reduction
        
        Args:
            image (numpy.ndarray): Input grayscale image
            
        Returns:
            numpy.ndarray: Blurred image
        """
        blurred = cv2.GaussianBlur(
            image, 
            (self.blur_kernel_size, self.blur_kernel_size), 
            0
        )
        return blurred
    
    def binarize_otsu(self, image):
        """
        Apply Otsu's automatic thresholding
        
        Args:
            image (numpy.ndarray): Input grayscale image
            
        Returns:
            tuple: (threshold_value, binary_image)
        """
        threshold_value, binary = cv2.threshold(
            image, 0, 255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        print(f"✓ Otsu threshold value: {threshold_value:.2f}")
        return threshold_value, binary
    
    def binarize_adaptive(self, image, block_size=11, constant=2):
        """
        Apply adaptive thresholding for varying illumination
        
        Args:
            image (numpy.ndarray): Input grayscale image
            block_size (int): Size of neighborhood area
            constant (int): Constant subtracted from mean
            
        Returns:
            numpy.ndarray: Binary image
        """
        binary = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, constant
        )
        return binary
    
    def apply_morphology(self, binary):
        """
        Apply morphological operations to clean binary image
        
        Args:
            binary (numpy.ndarray): Input binary image
            
        Returns:
            numpy.ndarray: Cleaned binary image
        """
        kernel = np.ones(
            (self.morph_kernel_size, self.morph_kernel_size), 
            np.uint8
        )
        
        # Opening: Remove small noise
        opened = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel, 
            iterations=self.morph_iterations
        )
        
        # Closing: Fill small holes
        closed = cv2.morphologyEx(
            opened, cv2.MORPH_CLOSE, kernel, 
            iterations=self.morph_iterations
        )
        
        return closed
    
    def preprocess(self, image, method='otsu'):
        """
        Complete preprocessing pipeline
        
        Args:
            image (numpy.ndarray): Input BGR image
            method (str): Binarization method ('otsu' or 'adaptive')
            
        Returns:
            dict: Dictionary containing grayscale and binary images
        """
        print("Starting preprocessing...")
        
        # Step 1: Convert to grayscale
        gray = self.convert_to_grayscale(image)
        print("✓ Converted to grayscale")
        
        # Step 2: Apply blur
        blurred = self.apply_blur(gray)
        print("✓ Applied Gaussian blur")
        
        # Step 3: Binarization
        if method == 'otsu':
            _, binary = self.binarize_otsu(blurred)
        elif method == 'adaptive':
            binary = self.binarize_adaptive(blurred)
        else:
            raise ValueError(f"Unknown binarization method: {method}")
        
        print("✓ Image binarized")
        
        # Step 4: Morphological operations
        cleaned = self.apply_morphology(binary)
        print("✓ Morphological operations applied")
        
        return {
            'grayscale': gray,
            'blurred': blurred,
            'binary': cleaned
        }
