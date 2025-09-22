import numpy as np
import cv2
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import confusion_matrix, precision_recall_curve
from ultralytics import YOLO
import os
import glob

# Try to import ByteTrack - use fallback if not available
try:
    from yolox.tracker.byte_tracker import BYTETracker
    BYTETRACK_AVAILABLE = True
    print("ByteTracker imported successfully")
except ImportError:
    try:
        # Alternative import paths
        from byte_tracker import BYTETracker
        BYTETRACK_AVAILABLE = True
        print("ByteTracker imported from byte_tracker")
    except ImportError:
        BYTETRACK_AVAILABLE = False
        print("Warning: ByteTracker not available. Install with: pip install yolox")
        
        # Fallback simple tracker
        class SimpleBYTETracker:
            def __init__(self, *args, **kwargs):
                self.tracks = {}
                self.next_id = 0
                
            def update(self, output_results, img_size):
                # Simple tracking fallback
                current_tracks = []
                for det in output_results:
                    track_id = self.next_id
                    self.next_id += 1
                    current_tracks.append({
                        'track_id': track_id,
                        'bbox': det[:4],
                        'score': det[4],
                        'class': int(det[5]) if len(det) > 5 else 0
                    })
                return current_tracks
        
        BYTETracker = SimpleBYTETracker


class PhysicsBasedMotionDetector:
    def __init__(self, config: Optional[Dict] = None):
        self.config = {
            'flow_params': {
                'pyr_scale': 0.5,
                'levels': 3,
                'winsize': 15,
                'iterations': 3,
                'poly_n': 5,
                'poly_sigma': 1.2,
                'flags': 0
            },
            'temporal_window': 5,
            'lambda_weight': 0.7,
            'high_motion_percentile': 95,
            'grid_size': (4, 5),
            'weights': [0.6, 0.3, 0.1],
            'output_params': {
                'graph_height': 120,
                'window_size': 150,
                'step_size': 16
            },
            'flow_visualization_params': {
                'threshold': 1.0,
                'step': 16,
                'scale': 1.5,
                'color': (0, 255, 0)
            },
            'yolo_params': {
                'conf_threshold': 0.25,
                'target_classes': ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck'],
                'model_path': 'yolov8n.pt'
            },
            'physics_params': {
                'pixel_to_meter_ratio': 0.03,
                'fps': 25.0,
                'normal_walking_speed': 1.5,
                'normal_running_speed': 4.0,
                'max_human_speed': 8.0,
                'normal_acceleration': 1.5,
                'abnormal_acceleration': 3.0,
                'max_human_acceleration': 6.0,
                'speed_thresholds': {
                    'stationary': (0.0, 0.5),
                    'walking': (0.5, 2.5),
                    'jogging': (2.5, 4.5),
                    'running': (4.5, 7.0),
                    'abnormal': 7.0
                },
                'acceleration_thresholds': {
                    'normal': 1.5,
                    'suspicious': 2.5,
                    'abnormal': 4.0
                },
                'detection_sensitivity': {
                    'motion_threshold': 2.0,
                    'velocity_threshold': 1.0,
                    'acceleration_threshold': 1.5,
                    'consistency_threshold': 0.6
                }
            },
            'noise_filtering': {
                'gaussian_blur_kernel': 5,
                'median_filter_size': 5,
                'temporal_smoothing_window': 7,
                'min_motion_area': 50,
                'motion_consistency_frames': 3
            },
            'tracker_params': {
                'frame_rate': 25,
                'track_thresh': 0.6,
                'track_buffer': 30,
                'match_thresh': 0.8,
                'min_box_area': 200
            }
        }
        if config:
            self.config.update(config)
                
        # Initialize components
        self.motion_history = []
        self.background_model = None
        self.background_std = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=16, detectShadows=False
        )
        
        # Physics tracking variables
        self.prev_flow = None
        self.prev_velocity_grid = None
        self.velocity_history = []
        self.motion_consistency_tracker = []
        
        # Noise filtering
        self.score_history = []
        
        # Initialize YOLO
        try:
            self.yolo_model = YOLO(self.config['yolo_params']['model_path'])
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Installing ultralytics...")
            os.system('pip install -q ultralytics')
            self.yolo_model = YOLO(self.config['yolo_params']['model_path'])

        # Initialize ByteTracker
        if BYTETRACK_AVAILABLE:
            try:
                # Configure ByteTracker arguments
                class TrackerArgs:
                    def __init__(self):
                        self.track_thresh = self.config['tracker_params']['track_thresh']
                        self.track_buffer = self.config['tracker_params']['track_buffer']
                        self.match_thresh = self.config['tracker_params']['match_thresh']
                        self.min_box_area = self.config['tracker_params']['min_box_area']
                        self.mot20 = False
                
                self.tracker_args = TrackerArgs()
                self.byte_tracker = BYTETracker(frame_rate=self.config['tracker_params']['frame_rate'])
                self.track_histories = {}
                print("ByteTracker initialized successfully")
            except Exception as e:
                print(f"Error initializing ByteTracker: {e}")
                self.byte_tracker = None
        else:
            self.byte_tracker = None
            self.track_histories = {}

    def pixels_to_meters(self, pixel_distance: float) -> float:
        """แปลงระยะทางจาก pixels เป็น meters"""
        return pixel_distance * self.config['physics_params']['pixel_to_meter_ratio']

    def pixels_per_frame_to_meters_per_second(self, pixel_velocity: float) -> float:
        """แปลงความเร็วจาก pixels/frame เป็น m/s"""
        meters_per_frame = self.pixels_to_meters(pixel_velocity)
        return meters_per_frame * self.config['physics_params']['fps']

    def detect_and_track_objects(self, frame: np.ndarray, frame_idx: int) -> Tuple[List[Dict], float]:
        """ตรวจจับและติดตามวัตถุด้วย YOLO + ByteTracker"""
        # YOLO Detection
        results = self.yolo_model(frame, verbose=False)
        detections = []
        max_confidence = 0
        
        # Convert YOLO results to detection format
        yolo_detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_name = result.names[int(box.cls)]
                confidence = float(box.conf)
                bbox = box.xyxy[0].cpu().numpy()
                
                if class_name in self.config['yolo_params']['target_classes']:
                    if confidence > self.config['yolo_params']['conf_threshold']:
                        max_confidence = max(max_confidence, confidence)
                        # Format: [x1, y1, x2, y2, conf, class_id]
                        yolo_detections.append([
                            bbox[0], bbox[1], bbox[2], bbox[3], 
                            confidence, int(box.cls)
                        ])
        
        tracked_objects = []
        
        if self.byte_tracker is not None and len(yolo_detections) > 0:
            try:
                # Convert to numpy array for ByteTracker
                output_results = np.array(yolo_detections)
                
                # Update ByteTracker
                online_targets = self.byte_tracker.update(
                    output_results, 
                    img_info=(frame.shape[0], frame.shape[1]),
                    img_size=(frame.shape[0], frame.shape[1])
                )
                
                # Process tracked objects
                for track in online_targets:
                    track_id = track.track_id
                    bbox = np.array(track.tlbr)  # Ensure bbox is numpy array
                    score = track.score
                    
                    # Get class info from original detections
                    class_id = 0  # Default to person
                    class_name = 'person'
                    for det in yolo_detections:
                        det_bbox = det[:4]
                        # Simple bbox matching
                        if (abs(det_bbox[0] - bbox[0]) < 10 and 
                            abs(det_bbox[1] - bbox[1]) < 10):
                            class_id = int(det[5])
                            class_name = self.yolo_model.names[class_id]
                            break
                    
                    # Calculate motion data for this track
                    motion_data = self.calculate_track_motion(track_id, bbox, frame_idx)
                    
                    tracked_objects.append({
                        'track_id': track_id,
                        'bbox': bbox,  # Already numpy array
                        'confidence': score,
                        'class': class_name,
                        'class_id': class_id,
                        'motion_data': motion_data
                    })
                    
            except Exception as e:
                print(f"ByteTracker error: {e}")
                # Fallback to simple detection without tracking
                for det in yolo_detections:
                    bbox = np.array(det[:4])  # Ensure bbox is numpy array
                    confidence = det[4]
                    class_id = int(det[5])
                    class_name = self.yolo_model.names[class_id]
                    
                    tracked_objects.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': bbox,
                        'track_id': -1,
                        'motion_data': {}
                    })
                return tracked_objects, max_confidence
        else:
            # Fallback without tracking
            for det in yolo_detections:
                bbox = np.array(det[:4])  # Ensure bbox is numpy array
                confidence = det[4]
                class_id = int(det[5])
                class_name = self.yolo_model.names[class_id]
                
                tracked_objects.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': bbox,
                    'track_id': -1,
                    'motion_data': {}
                })
        
        return tracked_objects, max_confidence

    def calculate_track_motion(self, track_id: int, bbox: np.ndarray, frame_idx: int) -> Dict:
        """คำนวณการเคลื่อนไหวของ track"""
        motion_data = {
            'velocity_mps': 0.0,
            'acceleration_mps2': 0.0,
            'direction': 0.0,
            'motion_type': 'stationary',
            'is_fast_motion': False
        }
        
        # Get center point
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        current_pos = np.array([center_x, center_y])
        
        # Initialize track history if not exists
        if track_id not in self.track_histories:
            self.track_histories[track_id] = {
                'positions': [],
                'timestamps': [],
                'velocities': [],
                'accelerations': []
            }
        
        track_history = self.track_histories[track_id]
        track_history['positions'].append(current_pos)
        track_history['timestamps'].append(frame_idx)
        
        # Keep only recent history
        max_history = 10
        if len(track_history['positions']) > max_history:
            track_history['positions'] = track_history['positions'][-max_history:]
            track_history['timestamps'] = track_history['timestamps'][-max_history:]
            track_history['velocities'] = track_history['velocities'][-max_history:]
            track_history['accelerations'] = track_history['accelerations'][-max_history:]
        
        # Calculate motion if we have enough history
        if len(track_history['positions']) >= 2:
            # Calculate velocity
            prev_pos = track_history['positions'][-2]
            displacement_pixels = np.linalg.norm(current_pos - prev_pos)
            velocity_pixels_per_frame = displacement_pixels
            velocity_mps = self.pixels_per_frame_to_meters_per_second(velocity_pixels_per_frame)
            
            motion_data['velocity_mps'] = velocity_mps
            track_history['velocities'].append(velocity_mps)
            
            # Calculate direction
            if displacement_pixels > 0:
                direction_vec = current_pos - prev_pos
                motion_data['direction'] = np.arctan2(direction_vec[1], direction_vec[0])
            
            # Calculate acceleration if we have velocity history
            if len(track_history['velocities']) >= 2:
                prev_velocity = track_history['velocities'][-2]
                acceleration = (velocity_mps - prev_velocity) * self.config['physics_params']['fps']
                motion_data['acceleration_mps2'] = acceleration
                track_history['accelerations'].append(acceleration)
            
            # Classify motion type
            motion_data['motion_type'] = self.classify_motion_type(
                velocity_mps, motion_data['acceleration_mps2']
            )
            
            # Check if fast motion
            speed_thresholds = self.config['physics_params']['speed_thresholds']
            motion_data['is_fast_motion'] = velocity_mps > speed_thresholds['jogging'][0]
        
        return motion_data

    def calculate_track_based_anomaly_score(self, tracked_objects: List[Dict]) -> float:
        """คำนวณคะแนนความผิดปกติจาก tracking data"""
        if not tracked_objects:
            return 0.0
        
        anomaly_score = 0.0
        person_tracks = [obj for obj in tracked_objects if obj['class'] == 'person']
        
        for obj in person_tracks:
            motion_data = obj.get('motion_data', {})
            velocity = motion_data.get('velocity_mps', 0.0)
            acceleration = abs(motion_data.get('acceleration_mps2', 0.0))
            
            # Score based on velocity
            speed_thresholds = self.config['physics_params']['speed_thresholds']
            if velocity > speed_thresholds['abnormal']:
                anomaly_score = max(anomaly_score, 0.9)
            elif velocity > speed_thresholds['running'][1]:
                anomaly_score = max(anomaly_score, 0.7)
            elif velocity > speed_thresholds['running'][0]:
                anomaly_score = max(anomaly_score, 0.5)
            elif velocity > speed_thresholds['jogging'][0]:
                anomaly_score = max(anomaly_score, 0.3)
            
            # Score based on acceleration
            acc_thresholds = self.config['physics_params']['acceleration_thresholds']
            if acceleration > acc_thresholds['abnormal']:
                anomaly_score = max(anomaly_score, 0.8)
            elif acceleration > acc_thresholds['suspicious']:
                anomaly_score = max(anomaly_score, 0.4)
        
        return anomaly_score

    def apply_noise_filtering(self, flow: np.ndarray) -> np.ndarray:
        """ใช้ noise filtering กับ optical flow"""
        kernel_size = self.config['noise_filtering']['gaussian_blur_kernel']
        flow_x_filtered = cv2.GaussianBlur(flow[..., 0], (kernel_size, kernel_size), 0)
        flow_y_filtered = cv2.GaussianBlur(flow[..., 1], (kernel_size, kernel_size), 0)
        
        median_size = self.config['noise_filtering']['median_filter_size']
        flow_x_filtered = cv2.medianBlur(flow_x_filtered.astype(np.float32), median_size)
        flow_y_filtered = cv2.medianBlur(flow_y_filtered.astype(np.float32), median_size)
        
        filtered_flow = np.dstack([flow_x_filtered, flow_y_filtered])
        return filtered_flow

    def calculate_motion_consistency(self, flow: np.ndarray, cell_bounds: Tuple) -> float:
        """คำนวณความสม่ำเสมอของการเคลื่อนไหวใน cell"""
        y_start, y_end, x_start, x_end = cell_bounds
        cell_flow_x = flow[y_start:y_end, x_start:x_end, 0]
        cell_flow_y = flow[y_start:y_end, x_start:x_end, 1]
        
        if cell_flow_x.size == 0 or cell_flow_y.size == 0:
            return 0.0
        
        magnitude = np.sqrt(cell_flow_x**2 + cell_flow_y**2)
        
        motion_threshold = self.config['physics_params']['detection_sensitivity']['motion_threshold']
        moving_pixels = magnitude > motion_threshold
        
        if np.sum(moving_pixels) < 10:
            return 0.0
        
        directions = np.arctan2(cell_flow_y[moving_pixels], cell_flow_x[moving_pixels])
        direction_consistency = 1.0 - np.std(directions) / np.pi
        
        magnitude_consistency = 1.0 - (np.std(magnitude[moving_pixels]) / 
                                      (np.mean(magnitude[moving_pixels]) + 1e-6))
        
        overall_consistency = 0.6 * direction_consistency + 0.4 * magnitude_consistency
        return max(0.0, min(1.0, overall_consistency))

    def calculate_physics_based_motion(self, flow: np.ndarray, frame_size: Tuple[int, int]) -> Dict:
        """คำนวณการเคลื่อนที่ตามหลักฟิสิกส์"""
        h, w = frame_size
        grid_h, grid_w = self.config['grid_size']
        cell_h, cell_w = h // grid_h, w // grid_w
        
        filtered_flow = self.apply_noise_filtering(flow)
        magnitude, angle = cv2.cartToPolar(filtered_flow[..., 0], filtered_flow[..., 1])
        
        velocity_grid = np.zeros((grid_h, grid_w))
        acceleration_grid = np.zeros((grid_h, grid_w))
        consistency_grid = np.zeros((grid_h, grid_w))
        physics_score_grid = np.zeros((grid_h, grid_w))
        
        abnormal_cells = []
        
        for i in range(grid_h):
            for j in range(grid_w):
                y_start, y_end = i * cell_h, (i + 1) * cell_h
                x_start, x_end = j * cell_w, (j + 1) * cell_w
                
                cell_bounds = (y_start, y_end, x_start, x_end)
                
                cell_magnitude = magnitude[y_start:y_end, x_start:x_end]
                avg_pixel_velocity = np.percentile(cell_magnitude[cell_magnitude > 0], 75) if np.any(cell_magnitude > 0) else 0
                
                velocity_m_per_s = self.pixels_per_frame_to_meters_per_second(avg_pixel_velocity)
                velocity_grid[i, j] = velocity_m_per_s
                
                if self.prev_velocity_grid is not None:
                    prev_velocity = self.prev_velocity_grid[i, j]
                    time_delta = 1.0 / self.config['physics_params']['fps']
                    acceleration = (velocity_m_per_s - prev_velocity) / time_delta
                    acceleration_grid[i, j] = acceleration
                else:
                    acceleration_grid[i, j] = 0.0
                
                consistency = self.calculate_motion_consistency(filtered_flow, cell_bounds)
                consistency_grid[i, j] = consistency
                
                physics_score = self.calculate_improved_physics_score(
                    velocity_m_per_s, acceleration_grid[i, j], consistency
                )
                physics_score_grid[i, j] = physics_score
                
                is_abnormal = self.is_motion_abnormal(
                    velocity_m_per_s, acceleration_grid[i, j], consistency, avg_pixel_velocity
                )
                
                if is_abnormal:
                    abnormal_cells.append({
                        'grid_pos': (i, j),
                        'pixel_bounds': {
                            'y': (y_start, y_end),
                            'x': (x_start, x_end)
                        },
                        'velocity_ms': velocity_m_per_s,
                        'acceleration_ms2': acceleration_grid[i, j],
                        'consistency': consistency,
                        'physics_score': physics_score,
                        'motion_type': self.classify_motion_type(velocity_m_per_s, acceleration_grid[i, j]),
                        'pixel_velocity': avg_pixel_velocity
                    })
        
        self.prev_velocity_grid = velocity_grid.copy()
        
        overall_stats = {
            'mean_velocity': float(np.mean(velocity_grid)),
            'max_velocity': float(np.max(velocity_grid)),
            'mean_acceleration': float(np.mean(acceleration_grid)),
            'max_acceleration': float(np.max(np.abs(acceleration_grid))),
            'mean_consistency': float(np.mean(consistency_grid)),
            'abnormal_cell_count': len(abnormal_cells),
            'abnormal_cell_ratio': len(abnormal_cells) / (grid_h * grid_w)
        }
        
        return {
            'velocity_grid': velocity_grid,
            'acceleration_grid': acceleration_grid,
            'consistency_grid': consistency_grid,
            'physics_score_grid': physics_score_grid,
            'abnormal_cells': abnormal_cells,
            'overall_stats': overall_stats,
            'grid_size': (grid_h, grid_w),
            'cell_size': (cell_h, cell_w)
        }

    def is_motion_abnormal(self, velocity: float, acceleration: float, consistency: float, pixel_velocity: float) -> bool:
        """ตรวจสอบว่าการเคลื่อนไหวผิดปกติหรือไม่"""
        speed_thresholds = self.config['physics_params']['speed_thresholds']
        acc_thresholds = self.config['physics_params']['acceleration_thresholds']
        sensitivity = self.config['physics_params']['detection_sensitivity']
        
        if pixel_velocity < sensitivity['motion_threshold']:
            return False
        
        min_velocity_for_running = 0.8
        if velocity < min_velocity_for_running:
            return False
        
        if velocity > speed_thresholds['jogging'][0]:
            consistency_threshold = 0.4
        else:
            consistency_threshold = sensitivity['consistency_threshold']
        
        if consistency < consistency_threshold:
            return False
        
        speed_abnormal = velocity > speed_thresholds['abnormal']
        acceleration_abnormal = abs(acceleration) > acc_thresholds['abnormal']
        suspicious_motion = (
            abs(acceleration) > acc_thresholds['suspicious'] and 
            velocity > speed_thresholds['walking'][1]
        )
        fast_running = (
            velocity > speed_thresholds['jogging'][0] and
            consistency > 0.3
        )
        burst_motion = (
            pixel_velocity > sensitivity['motion_threshold'] * 2 and
            velocity > speed_thresholds['walking'][1]
        )
        
        return speed_abnormal or acceleration_abnormal or suspicious_motion or fast_running or burst_motion

    def calculate_improved_physics_score(self, velocity: float, acceleration: float, consistency: float) -> float:
        """คำนวณคะแนนความผิดปกติจากฟิสิกส์"""
        score = 0.0
        
        speed_thresholds = self.config['physics_params']['speed_thresholds']
        acc_thresholds = self.config['physics_params']['acceleration_thresholds']
        
        if velocity > speed_thresholds['abnormal']:
            score += 0.9
        elif velocity > speed_thresholds['running'][1]:
            score += 0.7
        elif velocity > speed_thresholds['running'][0]:
            score += 0.5
        elif velocity > speed_thresholds['jogging'][1]:
            score += 0.4
        elif velocity > speed_thresholds['jogging'][0]:
            score += 0.3
        elif velocity > speed_thresholds['walking'][1]:
            score += 0.2
        
        abs_acceleration = abs(acceleration)
        if abs_acceleration > acc_thresholds['abnormal']:
            score += 0.6
        elif abs_acceleration > acc_thresholds['suspicious']:
            score += 0.4
        elif abs_acceleration > acc_thresholds['normal']:
            score += 0.2
        
        if velocity > speed_thresholds['jogging'][0]:
            if consistency < 0.2:
                score *= 0.7
            elif consistency < 0.4:
                score *= 0.9
            else:
                score *= 1.1
        else:
            if consistency < 0.3:
                score *= 0.5
            elif consistency < 0.6:
                score *= 0.8
            else:
                score *= 1.0
        
        if velocity > speed_thresholds['jogging'][0] and abs_acceleration > acc_thresholds['normal']:
            score += 0.1
        
        return min(1.0, score)

    def classify_motion_type(self, velocity: float, acceleration: float) -> str:
        """จำแนกประเภทการเคลื่อนไหว"""
        speed_thresholds = self.config['physics_params']['speed_thresholds']
        acc_thresholds = self.config['physics_params']['acceleration_thresholds']
        
        if velocity > speed_thresholds['abnormal']:
            return "abnormally_fast"
        elif abs(acceleration) > acc_thresholds['abnormal']:
            if acceleration > 0:
                return "sudden_acceleration"
            else:
                return "sudden_deceleration"
        elif abs(acceleration) > acc_thresholds['suspicious']:
            return "suspicious_motion"
        elif velocity > speed_thresholds['running'][0]:
            return "running"
        elif velocity > speed_thresholds['jogging'][0]:
            return "jogging"
        elif velocity > speed_thresholds['walking'][0]:
            return "walking"
        else:
            return "stationary"

    def calculate_combined_physics_score(self, physics_data: Dict) -> float:
        """คำนวณคะแนนความผิดปกติรวม"""
        stats = physics_data['overall_stats']
        
        cell_ratio_score = min(1.0, stats['abnormal_cell_ratio'] * 5.0)
        
        max_velocity_score = 0.0
        speed_thresholds = self.config['physics_params']['speed_thresholds']
        if stats['max_velocity'] > speed_thresholds['abnormal']:
            max_velocity_score = 0.9
        elif stats['max_velocity'] > speed_thresholds['running'][1]:
            max_velocity_score = 0.6
        elif stats['max_velocity'] > speed_thresholds['running'][0]:
            max_velocity_score = 0.3
        
        max_acceleration_score = 0.0
        acc_thresholds = self.config['physics_params']['acceleration_thresholds']
        if stats['max_acceleration'] > acc_thresholds['abnormal']:
            max_acceleration_score = 0.7
        elif stats['max_acceleration'] > acc_thresholds['suspicious']:
            max_acceleration_score = 0.4
        
        consistency_penalty = 0.0
        if stats['mean_consistency'] < 0.5:
            consistency_penalty = 0.3
        elif stats['mean_consistency'] < 0.7:
            consistency_penalty = 0.1
        
        combined_score = (
            0.4 * cell_ratio_score +
            0.3 * max_velocity_score +
            0.2 * max_acceleration_score +
            0.1 * consistency_penalty
        )
        
        return min(1.0, combined_score)

    def temporal_smoothing(self, current_score: float) -> float:
        """ทำ temporal smoothing เพื่อลด noise"""
        window_size = self.config['noise_filtering']['temporal_smoothing_window']
        
        self.score_history.append(current_score)
        if len(self.score_history) > window_size:
            self.score_history = self.score_history[-window_size:]
        
        weights = np.exp(np.linspace(-1.5, 0, len(self.score_history)))
        weights /= weights.sum()
        
        smoothed_score = np.average(self.score_history, weights=weights)
        
        if current_score > 0.7 and current_score > smoothed_score * 1.5:
            recent_high_scores = sum(1 for s in self.score_history[-3:] if s > 0.4)
            if recent_high_scores >= 2:
                return current_score
        
        if smoothed_score < 0.3:
            smoothed_score *= 0.8
        
        return smoothed_score

    def detect_objects(self, frame: np.ndarray) -> Tuple[List[Dict], float]:
        """ตรวจจับวัตถุด้วย YOLO"""
        results = self.yolo_model(frame, verbose=False)
        detections = []
        max_confidence = 0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_name = result.names[int(box.cls)]
                confidence = float(box.conf)
                bbox = box.xyxy[0].cpu().numpy()
                
                if class_name in self.config['yolo_params']['target_classes']:
                    if confidence > self.config['yolo_params']['conf_threshold']:
                        max_confidence = max(max_confidence, confidence)
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': np.array(bbox) if isinstance(bbox, list) else bbox
                        })
        
        return detections, max_confidence

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """แปลงเฟรมเป็น Grayscale และทำการ preprocessing"""
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        fg_mask = self.bg_subtractor.apply(frame)
        
        kernel = np.ones((3,3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        processed_with_bg = cv2.bitwise_and(frame, frame, mask=fg_mask)
        processed = cv2.addWeighted(frame, 0.5, processed_with_bg, 0.5, 0)
        
        return processed

    def analyze_video_from_frames(self, frames_path: str) -> Tuple[np.ndarray, List[np.ndarray], List[List[Dict]]]:
        """วิเคราะห์วิดีโอจากโฟลเดอร์ที่มีไฟล์เฟรม - ใช้ ByteTrack"""
        frame_files = sorted(glob.glob(os.path.join(frames_path, "*.jpg")))
        total_frames = len(frame_files)
        
        if total_frames == 0:
            raise ValueError(f"No frame files found in {frames_path}")

        anomaly_scores = []
        flows = []
        all_detections = []
        prev_processed = None
        
        # Reset state
        self.score_history = []
        self.prev_velocity_grid = None
        self.track_histories = {}
        
        with tqdm(total=total_frames, desc=f'Analyzing {os.path.basename(frames_path)} with ByteTrack') as pbar:
            for frame_idx, frame_file in enumerate(frame_files):
                frame = cv2.imread(frame_file)
                if frame is None:
                    continue

                # YOLO Detection + ByteTrack
                tracked_objects, yolo_confidence = self.detect_and_track_objects(frame, frame_idx)
                all_detections.append(tracked_objects)

                # Analyze tracked objects for suspicious activity
                suspicious_score = 0.0
                person_count = 0
                fast_moving_persons = 0
                
                for obj in tracked_objects:
                    if obj['class'] == 'person':
                        person_count += 1
                        motion_data = obj.get('motion_data', {})
                        if motion_data.get('is_fast_motion', False):
                            fast_moving_persons += 1
                            velocity = motion_data.get('velocity_mps', 0.0)
                            if velocity > self.config['physics_params']['speed_thresholds']['running'][0]:
                                suspicious_score = max(suspicious_score, 0.8)
                            elif velocity > self.config['physics_params']['speed_thresholds']['jogging'][0]:
                                suspicious_score = max(suspicious_score, 0.6)
                    else:
                        if obj['confidence'] > 0.5:
                            suspicious_score = max(suspicious_score, obj['confidence'])

                # คำนวณคะแนนจาก track-based analysis
                track_anomaly_score = self.calculate_track_based_anomaly_score(tracked_objects)

                # Optical Flow และ Physics Analysis
                processed = self.preprocess_frame(frame)
                
                if prev_processed is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_processed, processed, None, **self.config['flow_params']
                    )
                    
                    physics_data = self.calculate_physics_based_motion(flow, processed.shape[:2])
                    physics_score = self.calculate_combined_physics_score(physics_data)
                    
                    # คำนวณคะแนนรวม - ผสมระหว่าง track-based และ physics-based
                    if track_anomaly_score > 0.5:
                        current_score = track_anomaly_score
                    elif suspicious_score > 0.6:
                        current_score = min(0.9, suspicious_score + 0.1 * physics_score)
                    elif fast_moving_persons > 0:
                        current_score = max(track_anomaly_score, 0.7 * physics_score + 0.3 * suspicious_score)
                    elif physics_score > 0.6:
                        current_score = physics_score
                    elif physics_score > 0.3:
                        if person_count > 0:
                            current_score = 0.9 * physics_score + 0.1 * yolo_confidence
                        else:
                            current_score = 0.7 * physics_score
                    elif person_count > 0 and physics_score > 0.2:
                        current_score = 0.7 * physics_score
                    elif physics_score > 0.15:
                        current_score = 0.5 * physics_score
                    else:
                        current_score = 0.05 * physics_score

                    smoothed_score = self.temporal_smoothing(current_score)
                    anomaly_scores.append(smoothed_score)
                    flows.append(flow)

                prev_processed = processed
                pbar.update(1)

        # Post-processing: ลด noise ด้วย median filter
        anomaly_scores = np.array(anomaly_scores)
        if len(anomaly_scores) > 5:
            try:
                from scipy.signal import medfilt
                anomaly_scores = medfilt(anomaly_scores, kernel_size=5)
            except ImportError:
                print("Warning: scipy not available, skipping median filtering")
        
        return anomaly_scores, flows, all_detections

    def load_avenue_labels(self, label_path: str) -> np.ndarray:
        """โหลด Avenue dataset labels จากไฟล์ .npy"""
        try:
            labels = np.load(label_path)
            if labels.shape[0] == 1:
                labels = labels[0]
            print(f"Avenue labels loaded: {labels.shape} frames")
            print(f"Anomaly frames: {np.sum(labels)} ({np.sum(labels)/len(labels)*100:.2f}%)")
            return labels
        except Exception as e:
            print(f"Error loading Avenue labels: {e}")
            return None

    def get_video_frame_mapping(self, testing_path: str) -> Dict[str, Tuple[int, int]]:
        """สร้าง mapping ระหว่างชื่อวิดีโอกับช่วง frame index ใน label array"""
        video_folders = sorted([d for d in os.listdir(testing_path) 
                               if os.path.isdir(os.path.join(testing_path, d))])
        
        frame_mapping = {}
        current_index = 0
        
        for video_folder in video_folders:
            video_path = os.path.join(testing_path, video_folder)
            frame_files = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
            frame_count = len(frame_files)
            
            frame_mapping[video_folder] = (current_index, current_index + frame_count)
            current_index += frame_count
            
        return frame_mapping

    def get_video_labels(self, video_name: str, all_labels: np.ndarray, frame_mapping: Dict[str, Tuple[int, int]]) -> np.ndarray:
        """ดึง labels ที่เกี่ยวข้องกับวิดีโอที่ระบุ"""
        if video_name in frame_mapping:
            start_idx, end_idx = frame_mapping[video_name]
            return all_labels[start_idx:end_idx]
        else:
            print(f"Warning: Video {video_name} not found in mapping")
            return None

    def draw_tracked_objects(self, frame: np.ndarray, tracked_objects: List[Dict]) -> np.ndarray:
        """วาดกรอบและข้อมูล tracked objects"""
        viz_frame = frame.copy()
        
        for obj in tracked_objects:
            bbox = obj['bbox']
            if isinstance(bbox, list):
                bbox = np.array(bbox)
            bbox = bbox.astype(int)
            track_id = obj.get('track_id', -1)
            motion_data = obj.get('motion_data', {})
            
            velocity = motion_data.get('velocity_mps', 0.0)
            motion_type = motion_data.get('motion_type', 'stationary')
            
            if motion_type in ['abnormally_fast', 'very_fast_running']:
                color = (0, 0, 255)  # แดง
                thickness = 3
            elif motion_type == 'running':
                color = (0, 165, 255)  # ส้ม
                thickness = 2
            elif motion_type in ['jogging', 'fast_walking']:
                color = (0, 255, 255)  # เหลือง
                thickness = 2
            elif obj['class'] == 'person':
                color = (0, 255, 0)  # เขียวสำหรับคนปกติ
                thickness = 2
            else:
                color = (255, 0, 0)  # น้ำเงินสำหรับวัตถุอื่น
                thickness = 2
            
            cv2.rectangle(viz_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
            
            if track_id >= 0:
                if obj['class'] == 'person' and velocity > 0.5:
                    label = f"ID:{track_id} {motion_type} {velocity:.1f}m/s"
                else:
                    label = f"ID:{track_id} {obj['class']} {obj['confidence']:.2f}"
            else:
                label = f"{obj['class']} {obj['confidence']:.2f}"
            
            font_scale = 0.5
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            
            cv2.rectangle(viz_frame, 
                        (bbox[0], bbox[1] - text_size[1] - 8),
                        (bbox[0] + text_size[0] + 4, bbox[1]),
                        color, -1)
            
            cv2.putText(viz_frame, label,
                       (bbox[0] + 2, bbox[1] - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            
            # วาดลูกศรทิศทาง
            if velocity > 1.0 and 'direction' in motion_data:
                direction = motion_data['direction']
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                arrow_length = min(30, int(velocity * 5))
                
                end_x = int(center_x + arrow_length * np.cos(direction))
                end_y = int(center_y + arrow_length * np.sin(direction))
                
                cv2.arrowedLine(viz_frame, (center_x, center_y), (end_x, end_y), 
                              color, 2, tipLength=0.3)
        
        return viz_frame

    def draw_physics_analysis(self, frame: np.ndarray, physics_data: Dict) -> np.ndarray:
        """วาดการวิเคราะห์ทางฟิสิกส์บนเฟรม"""
        viz_frame = frame.copy()
        h, w = frame.shape[:2]
        
        grid_h, grid_w = physics_data['grid_size']
        cell_h, cell_w = physics_data['cell_size']
        
        # วาดเส้นแบ่ง grid
        for i in range(1, grid_h):
            y = i * cell_h
            cv2.line(viz_frame, (0, y), (w, y), (100, 100, 100), 1, cv2.LINE_AA)
        
        for j in range(1, grid_w):
            x = j * cell_w
            cv2.line(viz_frame, (x, 0), (x, h), (100, 100, 100), 1, cv2.LINE_AA)
        
        # วาดข้อมูลใน abnormal cells
        for cell in physics_data['abnormal_cells']:
            y_start, y_end = cell['pixel_bounds']['y']
            x_start, x_end = cell['pixel_bounds']['x']
            center_x = (x_start + x_end) // 2
            center_y = (y_start + y_end) // 2
            
            motion_colors = {
                "abnormally_fast": (0, 0, 255),
                "sudden_acceleration": (0, 165, 255),
                "sudden_deceleration": (0, 165, 255),
                "suspicious_motion": (0, 255, 255),
                "running": (255, 100, 0),
                "jogging": (255, 150, 0)
            }
            
            color = motion_colors.get(cell['motion_type'], (255, 0, 0))
            
            if cell['motion_type'] in ["abnormally_fast", "sudden_acceleration", "sudden_deceleration"]:
                cv2.circle(viz_frame, (center_x, center_y), 15, color, 2)
                cv2.putText(viz_frame, "!", (center_x-4, center_y+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            else:
                cv2.arrowedLine(viz_frame, 
                              (center_x-10, center_y), (center_x+10, center_y),
                              color, 2, tipLength=0.3)
            
            label = {
                "abnormally_fast": "FAST",
                "sudden_acceleration": "ACC",
                "sudden_deceleration": "DEC", 
                "suspicious_motion": "SUS",
                "running": "RUN",
                "jogging": "JOG"
            }.get(cell['motion_type'], "!")
            
            font_scale = 0.4
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            
            cv2.rectangle(viz_frame, 
                        (center_x - text_size[0]//2 - 2, center_y + 20),
                        (center_x + text_size[0]//2 + 2, center_y + 20 + text_size[1] + 4),
                        color, -1)
            
            cv2.putText(viz_frame, label,
                       (center_x - text_size[0]//2, center_y + 20 + text_size[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        
        # แสดงสถิติโดยรวม
        stats = physics_data['overall_stats']
        info_lines = [
            f"Max Speed: {stats['max_velocity']:.1f} m/s",
            f"Alerts: {stats['abnormal_cell_count']}/{grid_h*grid_w}",
            f"Consistency: {stats['mean_consistency']:.2f}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(viz_frame, line, (10, 20 + i * 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # แสดงสถานะโดยรวม
        if stats['abnormal_cell_count'] > 2:
            status = "ALERT"
            status_color = (0, 0, 255)
        elif stats['abnormal_cell_count'] > 0:
            status = "WATCH"
            status_color = (0, 255, 255)
        else:
            status = "NORMAL"
            status_color = (0, 255, 0)
        
        cv2.putText(viz_frame, status, (w - 80, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2, cv2.LINE_AA)
        
        return viz_frame

    def evaluate(self, scores: np.ndarray, ground_truth: np.ndarray) -> Dict:
        """ประเมินผลการตรวจจับความผิดปกติ"""
        min_len = min(len(scores), len(ground_truth))
        scores = scores[:min_len]
        ground_truth = ground_truth[:min_len]

        optimal_threshold = 0.35

        predictions = (scores >= optimal_threshold).astype(int)
        accuracy = np.mean(predictions == ground_truth)
        precision = np.sum((predictions == 1) & (ground_truth == 1)) / (np.sum(predictions == 1) + 1e-10)
        recall = np.sum((predictions == 1) & (ground_truth == 1)) / (np.sum(ground_truth == 1) + 1e-10)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

        fpr, tpr, _ = roc_curve(ground_truth, scores)
        roc_auc = auc(fpr, tpr)

        return {
            'scores': scores,
            'ground_truth': ground_truth,
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'optimal_threshold': optimal_threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

    def plot_results(self, results: Dict, video_name: str = "") -> None:
        """แสดงผลลัพธ์ในรูปแบบกราฟ"""
        
        title_prefix = f"{video_name} - " if video_name else ""
        
        # 1. Anomaly Scores Over Time
        plt.figure(figsize=(20, 8))
        plt.plot(results['scores'], label='Enhanced Physics + ByteTrack Score', alpha=0.8, linewidth=2, color='blue')
        plt.plot(results['ground_truth'], 'r--', label='Ground Truth', alpha=0.8, linewidth=2)
        plt.axhline(y=results['optimal_threshold'], color='g', linestyle=':', 
                   label=f'Threshold ({results["optimal_threshold"]})', linewidth=2)
        plt.title(f'{title_prefix}Enhanced Physics + ByteTrack Anomaly Detection', fontsize=16)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('Anomaly Score', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 2. ROC Curve
        plt.figure(figsize=(20, 8))
        plt.plot(results['fpr'], results['tpr'], linewidth=3, 
                label=f"Enhanced Physics + ByteTrack (AUC = {results['auc']:.3f})", color='blue')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5)
        plt.title(f'{title_prefix}ROC Curve - Enhanced Physics + ByteTrack Analysis', fontsize=16)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 3. Performance Metrics
        plt.figure(figsize=(20, 8))
        metrics = {
            'AUC': results['auc'],
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1 Score': results['f1_score']
        }
        
        colors = ['steelblue', 'lightcoral', 'gold', 'lightgreen', 'plum']
        bars = plt.bar(metrics.keys(), metrics.values(), color=colors, width=0.6, alpha=0.8)
        
        plt.title(f'{title_prefix}Enhanced Physics + ByteTrack Performance Metrics', fontsize=16)
        plt.ylim(0, 1)
        
        for bar, (name, value) in zip(bars, metrics.items()):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', fontsize=11, fontweight='bold')
        
        plt.grid(True, axis='y', alpha=0.3)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()

    def create_visualization_video_from_frames(self, frames_path: str, scores: np.ndarray, 
                                             flows: List[np.ndarray], ground_truth: np.ndarray, 
                                             output_path: str) -> None:
        """สร้างวิดีโอแสดงผลจากโฟลเดอร์เฟรม - with ByteTrack visualization"""
        try:
            frame_files = sorted(glob.glob(os.path.join(frames_path, "*.jpg")))
            if not frame_files:
                raise ValueError(f"No frame files found in {frames_path}")

            sample_frame = cv2.imread(frame_files[0])
            if sample_frame is None:
                raise ValueError(f"Cannot read sample frame: {frame_files[0]}")
                
            height, width = sample_frame.shape[:2]
            fps = 25
            total_frames = len(frame_files)

            graph_height = self.config['output_params']['graph_height']
            output_height = height + graph_height + 20
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, output_height))

            max_frames = min(total_frames, len(scores), len(flows) + 1)

            # Reset physics state และ tracking
            self.prev_velocity_grid = None
            self.track_histories = {}

            with tqdm(total=max_frames, desc='Creating ByteTrack visualization') as pbar:
                gt_history = []
                for frame_idx in range(max_frames):
                    if frame_idx >= len(frame_files):
                        break
                        
                    frame = cv2.imread(frame_files[frame_idx])
                    if frame is None:
                        continue

                    current_score = scores[frame_idx] if frame_idx < len(scores) else 0.0

                    # Object Detection + Tracking และวาด
                    tracked_objects, _ = self.detect_and_track_objects(frame, frame_idx)
                    frame = self.draw_tracked_objects(frame, tracked_objects)

                    # Physics Analysis และวาด
                    if frame_idx > 0 and frame_idx - 1 < len(flows):
                        flow = flows[frame_idx - 1]
                        physics_data = self.calculate_physics_based_motion(flow, frame.shape[:2])
                        frame = self.draw_physics_analysis(frame, physics_data)

                    # เก็บประวัติสำหรับกราฟ
                    if frame_idx < len(ground_truth):
                        gt_history.append((ground_truth[frame_idx], current_score))

                        # สร้าง GT bar และ graph
                        gt_bar = self.draw_gt_bar(width, gt_history, current_score, max_frames)
                        graph_img = self.create_graph_background(width, graph_height, frame_idx, gt_history, max_frames)

                    # วาดกราฟ
                    self.draw_graph_elements(graph_img, gt_history, frame_idx, max_frames, width, graph_height, current_score)

                    # เพิ่มข้อมูล ByteTrack status
                    track_count = len([obj for obj in tracked_objects if obj.get('track_id', -1) >= 0])
                    fast_tracks = len([obj for obj in tracked_objects if obj.get('motion_data', {}).get('is_fast_motion', False)])
                    
                    cv2.putText(frame, f"Tracks: {track_count} | Fast: {fast_tracks}", 
                               (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    # รวมเฟรมทั้งหมด
                    combined_frame = np.vstack([frame, gt_bar, graph_img])
                    out.write(combined_frame)

                    pbar.update(1)

            out.release()
            print(f"ByteTrack visualization video saved: {output_path}")

        except Exception as e:
            print(f"Error creating ByteTrack visualization video: {str(e)}")
            raise

    def draw_gt_bar(self, width, gt_history, current_score, max_frames):
        """วาดแถบ ground truth และการทำนาย"""
        gt_bar = np.ones((20, width, 3), dtype=np.uint8) * 50
        
        for i, (gt, score) in enumerate(gt_history):
            x = int(i * width / max_frames)
            x_next = int((i + 1) * width / max_frames)
            pred = score >= 0.35

            if gt == 0:
                if pred == 0:
                    cv2.rectangle(gt_bar, (x, 0), (x_next, 20), (0, 255, 0), -1)  # True Negative
                else:
                    cv2.rectangle(gt_bar, (x, 0), (x_next, 20), (0, 0, 255), -1)   # False Positive
            else:
                if pred == 0:
                    cv2.rectangle(gt_bar, (x, 0), (x_next, 20), (255, 0, 0), -1)   # False Negative
                else:
                    cv2.rectangle(gt_bar, (x, 0), (x_next, 20), (255, 255, 0), -1) # True Positive
        
        return gt_bar

    def create_graph_background(self, width, height, frame_idx, gt_history, max_frames):
        """สร้างพื้นหลังกราฟ"""
        graph_img = np.ones((height, width, 3), dtype=np.uint8) * 50
        
        # เน้น anomaly regions
        for i, (gt, score) in enumerate(gt_history):
            if gt == 1:
                x = int(i * width / max_frames)
                x_next = int((i + 1) * width / max_frames)
                cv2.rectangle(graph_img, (x, 0), (x_next, height), (80, 80, 80), -1)
        
        return graph_img

    def draw_graph_elements(self, graph_img, gt_history, frame_idx, max_frames, width, graph_height, current_score):
        """วาดองค์ประกอบของกราฟ"""
        # วาดเส้น threshold
        threshold_y = int((1 - 0.5) * (graph_height - 20)) + 10
        self.draw_dashed_line(graph_img, (0, threshold_y), (width, threshold_y), (0, 255, 0), thickness=2)

        # วาดเส้นกราฟ score
        points = []
        for i in range(min(frame_idx + 1, len(gt_history))):
            x = int(i * width / max_frames)
            y = int((1 - gt_history[i][1]) * (graph_height - 20)) + 10
            points.append((x, y))

        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(graph_img, points[i], points[i + 1], (0, 150, 255), 3, cv2.LINE_AA)

        # Current frame indicator
        if frame_idx < max_frames:
            current_x = int(frame_idx * width / max_frames)
            cv2.line(graph_img, (current_x, 0), (current_x, graph_height), (255, 255, 255), 2)

        # แสดงข้อมูล score
        score_text = f"Enhanced Physics + ByteTrack Score: {current_score:.3f}"
        cv2.putText(graph_img, score_text, (10, 25), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # แสดงค่าแกน Y
        for i in range(0, 6):
            y_val = i * 0.2
            y_pos = int((1 - y_val) * (graph_height - 20)) + 10
            cv2.putText(graph_img, f"{y_val:.1f}", (width - 35, y_pos + 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    def draw_dashed_line(self, img, start_point, end_point, color, thickness=1, gap=10):
        """วาดเส้นประ"""
        x1, y1 = start_point
        x2, y2 = end_point
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx*dx + dy*dy)
        dashes = int(dist/gap)
        for i in range(0, dashes, 2):
            start_ratio = i / dashes
            end_ratio = min((i + 1) / dashes, 1.0)
            start = (int(x1 + dx*start_ratio), int(y1 + dy*start_ratio))
            end = (int(x1 + dx*end_ratio), int(y1 + dy*end_ratio))
            cv2.line(img, start, end, color, thickness)


def analyze_single_video_enhanced(frames_path: str, video_name: str, labels: np.ndarray, 
                                frame_mapping: Dict[str, Tuple[int, int]], 
                                detector: PhysicsBasedMotionDetector) -> Dict:
    """วิเคราะห์วิดีโอเดี่ยวด้วยการวิเคราะห์ทางฟิสิกส์ที่ปรับปรุงแล้ว + ByteTrack"""
    try:
        print(f"\n=== Enhanced Physics + ByteTrack Analysis: {video_name} ===")
        
        video_labels = detector.get_video_labels(video_name, labels, frame_mapping)
        if video_labels is None:
            print(f"Warning: No labels found for {video_name}")
            return None
            
        print(f"Video frames: {len(video_labels)}")
        print(f"Anomaly frames: {np.sum(video_labels)} ({np.sum(video_labels)/len(video_labels)*100:.2f}%)")
        
        # วิเคราะห์วิดีโอ
        scores, flows, detections = detector.analyze_video_from_frames(frames_path)
        
        # ประเมินผล
        results = detector.evaluate(scores, video_labels)
        
        # แสดงผลลัพธ์
        print(f"\nEnhanced Physics + ByteTrack Results for {video_name}:")
        print(f"ROC AUC: {results['auc']:.4f}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"Threshold: {results['optimal_threshold']:.2f}")
        
        # สร้างกราฟ
        detector.plot_results(results, f"Enhanced Physics + ByteTrack - {video_name}")
        
        # สร้างวิดีโอแสดงผล
        output_video_path = os.path.join(os.path.dirname(frames_path), f"{video_name}_enhanced_physics_bytetrack.avi")
        print(f"\nCreating Enhanced Physics + ByteTrack visualization: {output_video_path}")
        detector.create_visualization_video_from_frames(frames_path, results['scores'], flows, results['ground_truth'], output_video_path)
        
        return {
            'video_name': video_name,
            'results': results,
            'output_video': output_video_path
        }
        
    except Exception as e:
        print(f"Error analyzing {video_name}: {str(e)}")
        return None


def main_enhanced_physics_avenue(testing_path: str, label_path: str, video_name: Optional[str] = None):
    """Main function สำหรับรัน enhanced physics + ByteTrack anomaly detection"""
    try:
        print("=== Enhanced Physics + ByteTrack Avenue Dataset Analysis ===")
        print("Features:")
        print("- YOLO object detection with ByteTrack tracking")
        print("- Individual object motion analysis")
        print("- Track-based velocity and acceleration calculation")
        print("- Enhanced noise filtering and temporal smoothing")
        print("- Physics-based anomaly scoring")
        print("- Combined tracking + physics anomaly detection")
        print("- Real-time visualization with motion indicators")
        
        if BYTETRACK_AVAILABLE:
            print("✓ ByteTrack available - Full tracking features enabled")
        else:
            print("⚠ ByteTrack not available - Using fallback simple tracking")
        
        # สร้าง Enhanced Physics + ByteTrack Detector
        detector = PhysicsBasedMotionDetector()
        
        # โหลด Avenue labels
        print(f"\nLoading Avenue labels from: {label_path}")
        labels = detector.load_avenue_labels(label_path)
        if labels is None:
            raise ValueError("Failed to load Avenue labels")
        
        # ตรวจสอบ path
        if not os.path.exists(testing_path):
            raise FileNotFoundError(f"Testing path not found: {testing_path}")
        
        if video_name:
            # วิเคราะห์วิดีโอเดี่ยว
            frames_path = os.path.join(testing_path, video_name)
            if not os.path.exists(frames_path):
                raise FileNotFoundError(f"Video frames path not found: {frames_path}")
            
            frame_mapping = detector.get_video_frame_mapping(testing_path)
            result = analyze_single_video_enhanced(frames_path, video_name, labels, frame_mapping, detector)
            
            if result:
                print(f"\n✅ Enhanced analysis complete for {video_name}!")
                print(f"📊 Output video: {result['output_video']}")
                print(f"🎯 Performance Summary:")
                print(f"   - AUC: {result['results']['auc']:.3f}")
                print(f"   - Precision: {result['results']['precision']:.3f}")
                print(f"   - Recall: {result['results']['recall']:.3f}")
                print(f"   - F1-Score: {result['results']['f1_score']:.3f}")
                
                if BYTETRACK_AVAILABLE:
                    print("🚀 ByteTrack features used:")
                    print("   - Continuous object tracking across frames")
                    print("   - Motion trajectory analysis")
                    print("   - Individual velocity/acceleration calculation")
                    print("   - Track-based anomaly detection")
        else:
            print("Multiple video analysis not implemented yet. Please specify a video_name.")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    # กำหนด path สำหรับ Avenue dataset
    testing_path = "./data/avenue/testing/frames"
    label_path = "./data/frame_labels_avenue.npy"
    
    # Installation instructions
    print("🔧 Installation Requirements:")
    print("pip install ultralytics opencv-python scikit-learn matplotlib tqdm numpy")
    print("pip install yolox  # For ByteTrack (optional but recommended)")
    print("pip install scipy  # For enhanced filtering (optional)")
    print()
    
    # ตัวอย่างการใช้งาน Enhanced Physics + ByteTrack Detection:
    print("🎯 Enhanced Physics + ByteTrack Motion Detector")
    print("Key Features:")
    print("• YOLO object detection with ByteTrack multi-object tracking")
    print("• Individual trajectory analysis for each tracked object")
    print("• Real-time velocity & acceleration calculation per track")
    print("• Motion pattern classification (walking, jogging, running, etc.)")
    print("• Track-based anomaly scoring combined with physics analysis")
    print("• Enhanced noise filtering and temporal smoothing")
    print("• Visual motion arrows and track IDs with status indicators")
    print("• Comprehensive performance evaluation with ROC curves")
    print()
    print("Expected Improvements:")
    print("• More accurate detection of fast-moving persons")
    print("• Reduced false positives through tracking continuity")
    print("• Better motion metrics from individual object tracking")
    print("• Enhanced anomaly localization with track IDs")
    print("• Improved temporal consistency through ByteTrack")
    
    # เรียกใช้งาน
    main_enhanced_physics_avenue(testing_path, label_path, video_name="17")
