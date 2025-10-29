"""
Data Logging Module
Generates structured text logs and reports
"""

from datetime import datetime
import os
import json
import numpy as np


class DataLogger:
    """Log detection results to file"""
    
    def __init__(self, output_dir='output'):
        """
        Initialize logger
        
        Args:
            output_dir (str): Directory for output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    @staticmethod
    def convert_to_native(obj):
        """
        Convert numpy types to native Python types for JSON serialization
        
        Args:
            obj: Object to convert (can be numpy type, dict, list, etc.)
            
        Returns:
            Native Python type
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: DataLogger.convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [DataLogger.convert_to_native(item) for item in obj]
        else:
            return obj
        
    def generate_text_log(self, objects, filename='log.txt'):
        """
        Generate human-readable text log
        
        Args:
            objects (list): List of object feature dictionaries
            filename (str): Output filename
        """
        filepath = os.path.join(self.output_dir, filename)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(filepath, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("TABLETOP OBJECT DETECTION ANALYSIS LOG\n")
            f.write("=" * 70 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Objects Detected: {len(objects)}\n")
            f.write("=" * 70 + "\n\n")
            
            for obj in objects:
                f.write(f"--- Object ID: {obj['id']} ---\n")
                f.write(f"  Centroid (x, y):      ({float(obj['centroid'][0]):.2f}, {float(obj['centroid'][1]):.2f})\n")
                f.write(f"  Area (pixels):        {int(obj['area'])}\n")
                f.write(f"  Dimensions (W x H):   {int(obj['width'])} x {int(obj['height'])}\n")
                f.write(f"  Aspect Ratio:         {float(obj['aspect_ratio']):.2f}\n")
                f.write(f"  Orientation (deg):    {float(obj['orientation']):.2f}\n")
                f.write(f"  Bounding Box:         ({int(obj['bbox'][0])}, {int(obj['bbox'][1])}, {int(obj['bbox'][2])}, {int(obj['bbox'][3])})\n")
                f.write("\n")
            
            f.write("=" * 70 + "\n")
            f.write("END OF LOG\n")
            f.write("=" * 70 + "\n")
        
        print(f"✓ Text log saved: {filepath}")
    
    def generate_json_log(self, objects, metadata=None, filename='log.json'):
        """
        Generate machine-readable JSON log
        
        Args:
            objects (list): List of object feature dictionaries
            metadata (dict): Additional metadata
            filename (str): Output filename
        """
        filepath = os.path.join(self.output_dir, filename)
        timestamp = datetime.now().isoformat()
        
        # Prepare data structure with type conversion
        log_data = {
            'timestamp': timestamp,
            'total_objects': len(objects),
            'metadata': self.convert_to_native(metadata) if metadata else {},
            'objects': []
        }
        
        for obj in objects:
            obj_data = {
                'id': int(obj['id']),
                'centroid': {
                    'x': float(obj['centroid'][0]),
                    'y': float(obj['centroid'][1])
                },
                'area': int(obj['area']),
                'dimensions': {
                    'width': int(obj['width']),
                    'height': int(obj['height'])
                },
                'aspect_ratio': float(obj['aspect_ratio']),
                'orientation_degrees': float(obj['orientation']),
                'bounding_box': {
                    'x': int(obj['bbox'][0]),
                    'y': int(obj['bbox'][1]),
                    'width': int(obj['bbox'][2]),
                    'height': int(obj['bbox'][3])
                }
            }
            log_data['objects'].append(obj_data)
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"✓ JSON log saved: {filepath}")
    
    def generate_csv_log(self, objects, filename='log.csv'):
        """
        Generate CSV log for spreadsheet analysis
        
        Args:
            objects (list): List of object feature dictionaries
            filename (str): Output filename
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            # Header
            f.write("ID,Centroid_X,Centroid_Y,Area,Width,Height,Aspect_Ratio,Orientation,BBox_X,BBox_Y,BBox_W,BBox_H\n")
            
            # Data rows
            for obj in objects:
                f.write(f"{int(obj['id'])},")
                f.write(f"{float(obj['centroid'][0]):.2f},{float(obj['centroid'][1]):.2f},")
                f.write(f"{int(obj['area'])},")
                f.write(f"{int(obj['width'])},{int(obj['height'])},")
                f.write(f"{float(obj['aspect_ratio']):.2f},")
                f.write(f"{float(obj['orientation']):.2f},")
                f.write(f"{int(obj['bbox'][0])},{int(obj['bbox'][1])},{int(obj['bbox'][2])},{int(obj['bbox'][3])}\n")
        
        print(f"✓ CSV log saved: {filepath}")
    
    def generate_all_logs(self, objects, metadata=None):
        """
        Generate all log formats (text, JSON, CSV)
        
        Args:
            objects (list): List of object feature dictionaries
            metadata (dict): Additional metadata
        """
        self.generate_text_log(objects)
        self.generate_json_log(objects, metadata)
        self.generate_csv_log(objects)
