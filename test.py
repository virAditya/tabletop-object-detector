"""
Parameter Testing Script
Tests different MIN_OBJECT_AREA values to find optimal setting
"""

import cv2
from src.preprocess import ImagePreprocessor
from src.segmentation import ObjectSegmenter
from config.config import config

print("=" * 70)
print("PARAMETER TESTING - Finding Optimal MIN_OBJECT_AREA")
print("=" * 70)

# Capture image
print("\n[1] Capturing image...")
cap = cv2.VideoCapture(config.CAMERA_INDEX)

if not cap.isOpened():
    print("✗ Cannot open camera")
    exit()

ret, image = cap.read()
cap.release()

if not ret:
    print("✗ Failed to capture image")
    exit()

print(f"✓ Image captured: {image.shape[1]}x{image.shape[0]}")

# Preprocess
print("\n[2] Preprocessing...")
preprocessor = ImagePreprocessor(
    blur_kernel_size=config.BLUR_KERNEL_SIZE,
    morph_kernel_size=config.MORPH_KERNEL_SIZE,
    morph_iterations=config.MORPH_ITERATIONS
)

results = preprocessor.preprocess(image, method=config.BINARIZATION_METHOD)
binary = results['binary']

print("✓ Binary image created")

# Test different MIN_OBJECT_AREA values
print("\n[3] Testing different MIN_OBJECT_AREA values...")
print("-" * 70)

test_values = [100, 500, 1000, 1500, 2000, 3000, 5000, 8000, 10000]

for min_area in test_values:
    segmenter = ObjectSegmenter(min_area=min_area, connectivity=8)
    seg_results = segmenter.segment_connected_components(binary)
    
    num_objects = seg_results['num_objects']
    
    if num_objects > 0:
        areas = [obj['area'] for obj in seg_results['objects']]
        print(f"MIN_AREA = {min_area:5d} → {num_objects} objects detected | Areas: {areas}")
    else:
        print(f"MIN_AREA = {min_area:5d} → 0 objects (threshold too high!)")

print("-" * 70)

# Show binary image
print("\n[4] Check the binary image quality:")
print("  - White regions = detected areas")
print("  - Black = background")
print("  - Press any key to close")

cv2.imshow('Binary Image - Check segmentation quality', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)
print("Choose MIN_OBJECT_AREA value that gives you the correct object count")
print("Then update config/config.py with that value")
print("=" * 70)
