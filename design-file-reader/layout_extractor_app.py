#!/usr/bin/env python3
"""
Advanced Layout Extractor with Gradio Interface
- Better wall detection with gap analysis for doors
- Improved furniture detection using multiple methods
- Manual adjustment and pinning of unidentified items
- Export to multiple formats
"""

import cv2
import numpy as np
import json
try:
    import gradio as gr
except ImportError:
    gr = None  # Gradio not installed yet
try:
    import pandas as pd
except ImportError:
    pd = None  # pandas not installed yet
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import matplotlib.patches as mpatches
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
import io
import base64
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class LayoutExtractor:
    def __init__(self):
        self.image = None
        self.processed_image = None
        self.furniture_items = []
        self.walls = []
        self.doors = []
        self.manual_pins = []
        self.width = 0
        self.height = 0
        
    def process_image(self, image: np.ndarray) -> Dict:
        """Main processing pipeline with improved methods."""
        if image is None:
            return None
            
        self.image = image
        self.height, self.width = image.shape[:2]
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            self.gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            self.gray = image
        
        # Run detection pipeline
        results = {
            'furniture': self.detect_furniture_improved(),
            'walls': self.detect_walls_improved(),
            'doors': self.detect_doors_from_gaps(),
            'statistics': self.calculate_statistics()
        }
        
        return results
    
    def detect_furniture_improved(self) -> List[Dict]:
        """Improved furniture detection using multiple techniques."""
        furniture = []
        
        # Method 1: Adaptive threshold with multiple block sizes
        for block_size in [11, 15, 21]:
            adaptive = cv2.adaptiveThreshold(
                self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, block_size, 2
            )
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, hierarchy = cv2.findContours(
                cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for i, contour in enumerate(contours):
                # Skip if it's a hole (negative hierarchy)
                if hierarchy[0][i][3] != -1:
                    continue
                    
                area = cv2.contourArea(contour)
                
                # Furniture size filter
                if 200 < area < self.width * self.height * 0.15:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Get minimum area rectangle (handles rotation)
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    
                    # Calculate shape properties
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Approximate polygon
                    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                    
                    # Check if it's rectangular (4-6 vertices suggests rectangle)
                    is_rectangular = 4 <= len(approx) <= 6
                    
                    furniture_item = {
                        'id': len(furniture) + 1,
                        'center': [int(rect[0][0]), int(rect[0][1])],
                        'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                        'rotated_box': box.tolist(),
                        'area': float(area),
                        'perimeter': float(perimeter),
                        'circularity': float(circularity),
                        'is_rectangular': is_rectangular,
                        'vertices': len(approx),
                        'confidence': 0.8 if is_rectangular else 0.6
                    }
                    
                    # Check if similar item already exists
                    if not self.is_duplicate(furniture_item, furniture):
                        furniture.append(furniture_item)
        
        # Method 2: Connected components analysis
        _, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if 200 < area < self.width * self.height * 0.15:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                component_item = {
                    'id': len(furniture) + 1,
                    'center': [int(centroids[i][0]), int(centroids[i][1])],
                    'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                    'area': float(area),
                    'confidence': 0.7,
                    'method': 'connected_components'
                }
                
                if not self.is_duplicate(component_item, furniture):
                    furniture.append(component_item)
        
        # Sort by area (largest first) and renumber
        furniture.sort(key=lambda x: x['area'], reverse=True)
        for i, item in enumerate(furniture):
            item['id'] = i + 1
        
        self.furniture_items = furniture
        return furniture
    
    def detect_walls_improved(self) -> List[Dict]:
        """Improved wall detection focusing on strong edges."""
        walls = []
        
        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(self.gray, 9, 75, 75)
        
        # Multi-scale edge detection
        edges_list = []
        for threshold1, threshold2 in [(50, 150), (100, 200), (150, 250)]:
            edges = cv2.Canny(filtered, threshold1, threshold2, apertureSize=3)
            edges_list.append(edges)
        
        # Combine edges
        combined_edges = np.zeros_like(edges_list[0])
        for edges in edges_list:
            combined_edges = cv2.bitwise_or(combined_edges, edges)
        
        # Detect lines using Hough transform with different parameters
        all_lines = []
        for threshold in [80, 100, 120]:
            lines = cv2.HoughLinesP(
                combined_edges, 
                rho=1, 
                theta=np.pi/180, 
                threshold=threshold,
                minLineLength=50, 
                maxLineGap=20
            )
            if lines is not None:
                all_lines.extend(lines)
        
        # Process and merge similar lines
        merged_walls = self.merge_similar_lines(all_lines)
        
        for i, line in enumerate(merged_walls):
            x1, y1, x2, y2 = line
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            
            # Classify orientation
            if -10 < angle < 10 or angle > 170 or angle < -170:
                orientation = 'horizontal'
            elif 80 < abs(angle) < 100:
                orientation = 'vertical'
            else:
                orientation = 'diagonal'
            
            walls.append({
                'id': i + 1,
                'start': [int(x1), int(y1)],
                'end': [int(x2), int(y2)],
                'length': float(length),
                'angle': float(angle),
                'orientation': orientation,
                'thickness': self.estimate_wall_thickness(x1, y1, x2, y2)
            })
        
        self.walls = walls
        return walls
    
    def detect_doors_from_gaps(self) -> List[Dict]:
        """Detect doors by finding significant gaps in walls (max 2 doors)."""
        if not self.walls:
            return []
        
        doors = []
        
        # Separate walls by orientation
        h_walls = [w for w in self.walls if w['orientation'] == 'horizontal']
        v_walls = [w for w in self.walls if w['orientation'] == 'vertical']
        
        # Find gaps in horizontal walls
        h_gaps = self.find_gaps_in_wall_set(h_walls, 'horizontal')
        
        # Find gaps in vertical walls  
        v_gaps = self.find_gaps_in_wall_set(v_walls, 'vertical')
        
        # Combine all gaps
        all_gaps = h_gaps + v_gaps
        
        # Sort by gap size (larger gaps more likely to be doors)
        all_gaps.sort(key=lambda x: x['size'], reverse=True)
        
        # Take top 2 gaps as doors (as specified)
        for i, gap in enumerate(all_gaps[:2]):
            doors.append({
                'id': i + 1,
                'position': gap['position'],
                'size': gap['size'],
                'orientation': gap['orientation'],
                'confidence': gap['confidence']
            })
        
        self.doors = doors
        return doors
    
    def find_gaps_in_wall_set(self, walls: List[Dict], orientation: str) -> List[Dict]:
        """Find gaps in a set of walls with the same orientation."""
        if not walls:
            return []
        
        gaps = []
        
        # Sort walls by position
        if orientation == 'horizontal':
            walls.sort(key=lambda w: min(w['start'][1], w['end'][1]))
        else:
            walls.sort(key=lambda w: min(w['start'][0], w['end'][0]))
        
        # Group walls that are on the same line
        wall_lines = self.group_collinear_walls(walls, orientation)
        
        # Find gaps in each line of walls
        for line_walls in wall_lines:
            if len(line_walls) < 2:
                continue
                
            # Sort walls along the line
            if orientation == 'horizontal':
                line_walls.sort(key=lambda w: min(w['start'][0], w['end'][0]))
            else:
                line_walls.sort(key=lambda w: min(w['start'][1], w['end'][1]))
            
            # Check for gaps between consecutive walls
            for i in range(len(line_walls) - 1):
                w1 = line_walls[i]
                w2 = line_walls[i + 1]
                
                if orientation == 'horizontal':
                    end1 = max(w1['start'][0], w1['end'][0])
                    start2 = min(w2['start'][0], w2['end'][0])
                    gap_size = start2 - end1
                    gap_pos = [(end1 + start2) / 2, w1['start'][1]]
                else:
                    end1 = max(w1['start'][1], w1['end'][1])
                    start2 = min(w2['start'][1], w2['end'][1])
                    gap_size = start2 - end1
                    gap_pos = [w1['start'][0], (end1 + start2) / 2]
                
                # Door-sized gap (typically 70-150 pixels in floor plans)
                if 50 < gap_size < 200:
                    gaps.append({
                        'position': gap_pos,
                        'size': gap_size,
                        'orientation': orientation,
                        'confidence': 0.9 if 70 < gap_size < 150 else 0.7
                    })
        
        return gaps
    
    def group_collinear_walls(self, walls: List[Dict], orientation: str, tolerance: float = 20) -> List[List[Dict]]:
        """Group walls that are collinear (on the same line)."""
        if not walls:
            return []
        
        groups = []
        used = [False] * len(walls)
        
        for i, wall in enumerate(walls):
            if used[i]:
                continue
                
            group = [wall]
            used[i] = True
            
            # Find other walls on the same line
            for j, other_wall in enumerate(walls):
                if i == j or used[j]:
                    continue
                
                # Check if walls are collinear
                if orientation == 'horizontal':
                    y_diff = abs(wall['start'][1] - other_wall['start'][1])
                    if y_diff < tolerance:
                        group.append(other_wall)
                        used[j] = True
                else:
                    x_diff = abs(wall['start'][0] - other_wall['start'][0])
                    if x_diff < tolerance:
                        group.append(other_wall)
                        used[j] = True
            
            if group:
                groups.append(group)
        
        return groups
    
    def merge_similar_lines(self, lines: List) -> List:
        """Merge lines that are very similar or overlapping."""
        if not lines:
            return []
        
        merged = []
        used = [False] * len(lines)
        
        for i, line1 in enumerate(lines):
            if used[i]:
                continue
            
            x1, y1, x2, y2 = line1[0] if isinstance(line1, np.ndarray) else line1
            
            # Find similar lines
            similar_lines = [(x1, y1, x2, y2)]
            used[i] = True
            
            for j, line2 in enumerate(lines):
                if i == j or used[j]:
                    continue
                
                x3, y3, x4, y4 = line2[0] if isinstance(line2, np.ndarray) else line2
                
                # Check if lines are similar (close and parallel)
                if self.are_lines_similar(x1, y1, x2, y2, x3, y3, x4, y4):
                    similar_lines.append((x3, y3, x4, y4))
                    used[j] = True
            
            # Merge similar lines into one
            if similar_lines:
                merged_line = self.merge_line_group(similar_lines)
                merged.append(merged_line)
        
        return merged
    
    def are_lines_similar(self, x1, y1, x2, y2, x3, y3, x4, y4, angle_tol=10, dist_tol=30) -> bool:
        """Check if two lines are similar enough to merge."""
        # Calculate angles
        angle1 = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
        angle2 = np.arctan2(y4-y3, x4-x3) * 180 / np.pi
        
        # Normalize angles
        angle_diff = abs(angle1 - angle2)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        if angle_diff > angle_tol:
            return False
        
        # Check distance between lines
        # Distance from midpoint of line1 to line2
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Distance from point to line
        dist = self.point_to_line_distance(mid_x, mid_y, x3, y3, x4, y4)
        
        return dist < dist_tol
    
    def point_to_line_distance(self, px, py, x1, y1, x2, y2) -> float:
        """Calculate distance from point to line."""
        line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if line_length == 0:
            return np.sqrt((px-x1)**2 + (py-y1)**2)
        
        t = max(0, min(1, ((px-x1)*(x2-x1) + (py-y1)*(y2-y1)) / (line_length**2)))
        projection_x = x1 + t * (x2-x1)
        projection_y = y1 + t * (y2-y1)
        
        return np.sqrt((px-projection_x)**2 + (py-projection_y)**2)
    
    def merge_line_group(self, lines: List[Tuple]) -> Tuple:
        """Merge a group of similar lines into one representative line."""
        if len(lines) == 1:
            return lines[0]
        
        # Find the extremes of all lines
        all_points = []
        for x1, y1, x2, y2 in lines:
            all_points.append((x1, y1))
            all_points.append((x2, y2))
        
        # Find the line that best represents all points
        all_points = np.array(all_points)
        
        # Use PCA to find the principal direction
        mean = np.mean(all_points, axis=0)
        centered = all_points - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Get the principal direction
        principal_direction = eigenvectors[:, np.argmax(eigenvalues)]
        
        # Project all points onto this direction
        projections = np.dot(centered, principal_direction)
        
        # Find extremes
        min_proj_idx = np.argmin(projections)
        max_proj_idx = np.argmax(projections)
        
        # Get the extreme points
        point1 = all_points[min_proj_idx]
        point2 = all_points[max_proj_idx]
        
        return (int(point1[0]), int(point1[1]), int(point2[0]), int(point2[1]))
    
    def estimate_wall_thickness(self, x1, y1, x2, y2) -> float:
        """Estimate wall thickness by looking for parallel lines nearby."""
        # Simplified - look for parallel lines within 5-30 pixels
        thickness = 10.0  # Default thickness
        
        angle = np.arctan2(y2-y1, x2-x1)
        perpendicular_angle = angle + np.pi/2
        
        # Sample points perpendicular to the wall
        for dist in [5, 10, 15, 20, 25, 30]:
            sample_x = int((x1+x2)/2 + dist * np.cos(perpendicular_angle))
            sample_y = int((y1+y2)/2 + dist * np.sin(perpendicular_angle))
            
            if 0 <= sample_x < self.width and 0 <= sample_y < self.height:
                if self.gray[sample_y, sample_x] < 128:  # Found dark pixel (wall)
                    thickness = dist
                    break
        
        return thickness
    
    def is_duplicate(self, item: Dict, items: List[Dict], iou_threshold: float = 0.7) -> bool:
        """Check if an item is duplicate based on IoU."""
        for existing in items:
            iou = self.calculate_iou(item, existing)
            if iou > iou_threshold:
                return True
        return False
    
    def calculate_iou(self, item1: Dict, item2: Dict) -> float:
        """Calculate Intersection over Union for two items."""
        bbox1 = item1['bbox']
        bbox2 = item2['bbox']
        
        # Calculate intersection
        x1 = max(bbox1['x'], bbox2['x'])
        y1 = max(bbox1['y'], bbox2['y'])
        x2 = min(bbox1['x'] + bbox1['width'], bbox2['x'] + bbox2['width'])
        y2 = min(bbox1['y'] + bbox1['height'], bbox2['y'] + bbox2['height'])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = bbox1['width'] * bbox1['height']
        area2 = bbox2['width'] * bbox2['height']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def calculate_statistics(self) -> Dict:
        """Calculate statistics about the detection."""
        return {
            'furniture_count': len(self.furniture_items),
            'wall_count': len(self.walls),
            'door_count': len(self.doors),
            'total_furniture_area': sum(f['area'] for f in self.furniture_items),
            'image_dimensions': f"{self.width}x{self.height}",
            'wall_orientations': {
                'horizontal': sum(1 for w in self.walls if w['orientation'] == 'horizontal'),
                'vertical': sum(1 for w in self.walls if w['orientation'] == 'vertical'),
                'diagonal': sum(1 for w in self.walls if w['orientation'] == 'diagonal')
            }
        }
    
    def add_manual_pin(self, x: int, y: int, label: str = "Unknown") -> None:
        """Add a manual pin for unidentified items."""
        self.manual_pins.append({
            'id': len(self.manual_pins) + 1,
            'position': [x, y],
            'label': label
        })
    
    def update_item(self, item_type: str, item_id: int, updates: Dict) -> None:
        """Update an existing item with manual corrections."""
        if item_type == 'furniture':
            for item in self.furniture_items:
                if item['id'] == item_id:
                    item.update(updates)
                    break
        elif item_type == 'wall':
            for wall in self.walls:
                if wall['id'] == item_id:
                    wall.update(updates)
                    break
        elif item_type == 'door':
            for door in self.doors:
                if door['id'] == item_id:
                    door.update(updates)
                    break
    
    def export_to_json(self) -> str:
        """Export all data to JSON format."""
        data = {
            'furniture': self.furniture_items,
            'walls': self.walls,
            'doors': self.doors,
            'manual_pins': self.manual_pins,
            'statistics': self.calculate_statistics()
        }
        return json.dumps(data, indent=2)
    
    def export_to_csv(self) -> pd.DataFrame:
        """Export to CSV format."""
        rows = []
        
        # Add furniture
        for item in self.furniture_items:
            rows.append({
                'type': 'furniture',
                'id': item['id'],
                'center_x': item['center'][0],
                'center_y': item['center'][1],
                'width': item['bbox']['width'],
                'height': item['bbox']['height'],
                'area': item['area'],
                'confidence': item.get('confidence', 0)
            })
        
        # Add doors
        for door in self.doors:
            rows.append({
                'type': 'door',
                'id': door['id'],
                'center_x': door['position'][0],
                'center_y': door['position'][1],
                'width': door['size'],
                'height': 0,
                'area': 0,
                'confidence': door.get('confidence', 0)
            })
        
        # Add manual pins
        for pin in self.manual_pins:
            rows.append({
                'type': 'manual_pin',
                'id': pin['id'],
                'center_x': pin['position'][0],
                'center_y': pin['position'][1],
                'width': 0,
                'height': 0,
                'area': 0,
                'confidence': 1.0
            })
        
        return pd.DataFrame(rows)
    
    def visualize(self, show_walls=True, show_furniture=True, show_doors=True, show_pins=True) -> np.ndarray:
        """Create visualization of detected elements."""
        if self.image is None:
            return None
        
        # Create a copy for visualization
        vis_image = self.image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
        
        # Draw walls
        if show_walls:
            for wall in self.walls:
                color = (255, 0, 0) if wall['orientation'] == 'horizontal' else (0, 0, 255)
                cv2.line(vis_image, 
                        tuple(wall['start']), 
                        tuple(wall['end']), 
                        color, 2)
        
        # Draw furniture
        if show_furniture:
            for item in self.furniture_items:
                bbox = item['bbox']
                # Draw rectangle
                cv2.rectangle(vis_image,
                            (bbox['x'], bbox['y']),
                            (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                            (0, 255, 0), 2)
                
                # Draw ID
                cv2.putText(vis_image, f"F{item['id']}", 
                          (bbox['x'] + 5, bbox['y'] + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw doors
        if show_doors:
            for door in self.doors:
                pos = door['position']
                size = int(door['size'] / 2)
                # Draw door as a thick line segment
                if door['orientation'] == 'horizontal':
                    cv2.line(vis_image,
                           (int(pos[0] - size), int(pos[1])),
                           (int(pos[0] + size), int(pos[1])),
                           (255, 165, 0), 4)
                else:
                    cv2.line(vis_image,
                           (int(pos[0]), int(pos[1] - size)),
                           (int(pos[0]), int(pos[1] + size)),
                           (255, 165, 0), 4)
                
                # Add label
                cv2.putText(vis_image, f"D{door['id']}", 
                          (int(pos[0]) - 10, int(pos[1]) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
        
        # Draw manual pins
        if show_pins:
            for pin in self.manual_pins:
                pos = pin['position']
                # Draw pin as a circle with cross
                cv2.circle(vis_image, tuple(pos), 10, (255, 0, 255), 2)
                cv2.line(vis_image, (pos[0]-5, pos[1]), (pos[0]+5, pos[1]), (255, 0, 255), 2)
                cv2.line(vis_image, (pos[0], pos[1]-5), (pos[0], pos[1]+5), (255, 0, 255), 2)
                
                # Add label
                cv2.putText(vis_image, pin['label'], 
                          (pos[0] + 15, pos[1]),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        return vis_image


# Gradio Interface
extractor = LayoutExtractor()

def process_upload(image):
    """Process uploaded image."""
    if image is None:
        return None, "Please upload an image", "", ""
    
    results = extractor.process_image(image)
    vis = extractor.visualize()
    
    stats = results['statistics']
    stats_text = f"""
    📊 Detection Results:
    • Furniture: {stats['furniture_count']} items
    • Walls: {stats['wall_count']} segments
    • Doors: {stats['door_count']} detected
    • Image: {stats['image_dimensions']}
    """
    
    json_output = extractor.export_to_json()
    csv_df = extractor.export_to_csv()
    
    return vis, stats_text, json_output, csv_df

def add_pin(image, evt: gr.SelectData):
    """Add manual pin where user clicks."""
    if image is None:
        return image
    
    x, y = evt.index
    extractor.add_manual_pin(x, y, f"Pin_{len(extractor.manual_pins)+1}")
    vis = extractor.visualize()
    return vis

def update_visualization(show_walls, show_furniture, show_doors, show_pins):
    """Update visualization based on toggles."""
    vis = extractor.visualize(show_walls, show_furniture, show_doors, show_pins)
    return vis

def export_files():
    """Export to downloadable formats."""
    json_str = extractor.export_to_json()
    csv_df = extractor.export_to_csv()
    
    # Create downloadable files
    json_file = "layout_data.json"
    csv_file = "layout_data.csv"
    
    with open(json_file, 'w') as f:
        f.write(json_str)
    
    csv_df.to_csv(csv_file, index=False)
    
    return json_file, csv_file

# Create Gradio interface
with gr.Blocks(title="Layout Extractor", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🏠 Advanced Layout Extractor
    
    Upload a floor plan image to extract furniture, walls, and doors. 
    - **Automatic detection** of furniture and walls
    - **Door detection** from wall gaps (max 2 doors)
    - **Manual adjustment** - click to add pins for missed items
    - **Export** to JSON/CSV formats
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Floor Plan", type="numpy")
            process_btn = gr.Button("🔍 Process Image", variant="primary")
            
            gr.Markdown("### Display Options")
            show_walls = gr.Checkbox(label="Show Walls", value=True)
            show_furniture = gr.Checkbox(label="Show Furniture", value=True)
            show_doors = gr.Checkbox(label="Show Doors", value=True)
            show_pins = gr.Checkbox(label="Show Manual Pins", value=True)
            
            update_btn = gr.Button("🔄 Update Display")
            
        with gr.Column(scale=2):
            output_image = gr.Image(label="Detection Results", interactive=True)
            stats_output = gr.Textbox(label="Statistics", lines=5)
            
            with gr.Row():
                export_json_btn = gr.Button("📥 Export JSON")
                export_csv_btn = gr.Button("📥 Export CSV")
            
            with gr.Row():
                json_file = gr.File(label="JSON Export", visible=False)
                csv_file = gr.File(label="CSV Export", visible=False)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### JSON Output")
            json_output = gr.Textbox(label="JSON Data", lines=10, max_lines=20)
        
        with gr.Column():
            gr.Markdown("### CSV Preview")
            csv_output = gr.Dataframe(label="CSV Data", height=300)
    
    gr.Markdown("""
    ### Instructions:
    1. **Upload** a floor plan image (TIFF, PNG, JPG)
    2. **Process** to detect furniture, walls, and doors
    3. **Click on image** to add manual pins for missed items
    4. **Toggle display** options to show/hide elements
    5. **Export** results as JSON or CSV
    
    ### Notes:
    - Doors are detected from gaps in walls (max 2 per image)
    - Green boxes = Furniture (F1, F2, ...)
    - Red/Blue lines = Walls (horizontal/vertical)
    - Orange lines = Doors (D1, D2)
    - Purple pins = Manual markers
    """)
    
    # Connect events
    process_btn.click(
        process_upload,
        inputs=[input_image],
        outputs=[output_image, stats_output, json_output, csv_output]
    )
    
    output_image.select(
        add_pin,
        inputs=[output_image],
        outputs=[output_image]
    )
    
    update_btn.click(
        update_visualization,
        inputs=[show_walls, show_furniture, show_doors, show_pins],
        outputs=[output_image]
    )
    
    export_json_btn.click(
        lambda: (extractor.export_to_json(), gr.update(visible=True)),
        outputs=[json_file, json_file]
    )
    
    export_csv_btn.click(
        lambda: (extractor.export_to_csv().to_csv(index=False), gr.update(visible=True)),
        outputs=[csv_file, csv_file]
    )

if __name__ == "__main__":
    app.launch(share=False, inbrowser=True)