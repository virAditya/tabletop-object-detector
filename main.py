"""
Main Execution Script
Orchestrates the complete object detection pipeline
"""

import sys
from datetime import datetime

# Import modules
from src.capture import ImageCapture
from src.preprocess import ImagePreprocessor
from src.segmentation import ObjectSegmenter
from src.features import FeatureExtractor
from src.visualization import ResultVisualizer
from src.logger import DataLogger

from config.config import config
from utils.helpers import timer, print_header


@timer
def run_detection_pipeline():
    """Execute complete detection pipeline"""
    
    print_header("TABLETOP OBJECT DETECTION SYSTEM")
    start_time = datetime.now()
    
    try:
        # Step 1: Image Capture with Countdown
        print("\n[1/6] IMAGE ACQUISITION")
        print("-" * 70)
        capturer = ImageCapture(camera_source=config.CAMERA_INDEX)
        
        # Use countdown capture with cropping
        image = capturer.capture_with_countdown(
            countdown_seconds=config.COUNTDOWN_SECONDS,
            crop_top=config.CROP_TOP_PIXELS
        )
        
        if image is None:
            print("❌ Failed to capture image")
            return
        
        # Step 2: Preprocessing
        print("\n[2/6] PREPROCESSING & BINARIZATION")
        print("-" * 70)
        preprocessor = ImagePreprocessor(
            blur_kernel_size=config.BLUR_KERNEL_SIZE,
            morph_kernel_size=config.MORPH_KERNEL_SIZE,
            morph_iterations=config.MORPH_ITERATIONS
        )
        
        preprocess_results = preprocessor.preprocess(
            image, 
            method=config.BINARIZATION_METHOD
        )
        
        # Step 3: Segmentation with Filtering
        print("\n[3/6] OBJECT SEGMENTATION")
        print("-" * 70)
        segmenter = ObjectSegmenter(
            min_area=config.MIN_OBJECT_AREA,
            connectivity=config.CONNECTIVITY
        )
        
        if config.SEGMENTATION_METHOD == 'connected_components':
            seg_results = segmenter.segment_connected_components(
                preprocess_results['binary']
            )
        else:
            seg_results = segmenter.segment_contours(
                preprocess_results['binary']
            )
        
        if seg_results['num_objects'] == 0:
            print("⚠ No objects detected.")
            print("  Tips:")
            print("    • Check if there's enough contrast in the scene")
            print("    • Try adjusting MIN_OBJECT_AREA in config.py")
            print("    • Ensure good lighting conditions")
            return
        
        # Step 4: Feature Extraction
        print("\n[4/6] FEATURE EXTRACTION")
        print("-" * 70)
        extractor = FeatureExtractor()
        
        if config.SEGMENTATION_METHOD == 'connected_components':
            objects = extractor.extract_from_connected_components(
                seg_results,
                seg_results['label_map']
            )
        else:
            objects = extractor.extract_from_contours(seg_results)
        
        # Step 5: Visualization
        print("\n[5/6] VISUALIZATION")
        print("-" * 70)
        visualizer = ResultVisualizer(
            bbox_color=config.BBOX_COLOR,
            centroid_color=config.CENTROID_COLOR,
            text_color=config.TEXT_COLOR,
            orientation_color=config.ORIENTATION_COLOR
        )
        
        annotated = visualizer.create_annotated_image(image, objects)
        
        # Step 6: Logging & Output
        print("\n[6/6] DATA LOGGING & OUTPUT")
        print("-" * 70)
        logger = DataLogger(output_dir=config.OUTPUT_DIR)
        
        if config.GENERATE_LOGS:
            metadata = {
                'binarization_method': config.BINARIZATION_METHOD,
                'segmentation_method': config.SEGMENTATION_METHOD,
                'min_area_threshold': config.MIN_OBJECT_AREA,
                'image_dimensions': f"{image.shape[1]}x{image.shape[0]}",
                'filters_applied': {
                    'position_filter': config.FILTER_BY_POSITION,
                    'aspect_ratio_filter': config.FILTER_BY_ASPECT_RATIO
                }
            }
            logger.generate_all_logs(objects, metadata)
        
        # Save images
        if config.SAVE_IMAGES:
            images_to_save = {
                'original': image,
                'binary': preprocess_results['binary'],
                'annotated': annotated
            }
            visualizer.save_results(images_to_save, config.OUTPUT_DIR)
        
        # Display results
        if config.DISPLAY_RESULTS:
            visualizer.display_results({
                'Original': image,
                'Binary': preprocess_results['binary'],
                'Annotated': annotated
            })
        
        # Final Summary
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("✓ PIPELINE COMPLETE".center(70))
        print("=" * 70)
        print(f"Objects Detected:  {len(objects)}")
        print(f"Processing Time:   {processing_time:.3f} seconds")
        print(f"Output Directory:  {config.OUTPUT_DIR}/")
        print("=" * 70)
        
        # Print object summary
        print("\nDETECTED OBJECTS SUMMARY:")
        print("-" * 70)
        for obj in objects:
            print(f"  Object {obj['id']}: "
                  f"Centroid=({obj['centroid'][0]:.1f}, {obj['centroid'][1]:.1f}), "
                  f"Area={obj['area']}px, "
                  f"Angle={obj['orientation']:.1f}°")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_detection_pipeline()
