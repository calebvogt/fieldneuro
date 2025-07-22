"""
Behavioral Classification Module for UWB Animal Tracking Data

This module provides behavioral analysis capabilities for ultrawideband (UWB) tracking data,
designed to integrate with the existing uwb_animate.py and uwb_plots.py functions.

Author: AI Assistant
Date: 2025-07-21
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import circstd
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import timedelta

@dataclass
class BehaviorConfig:
    """Configuration for behavioral classification parameters"""
    window_size_seconds: float = 10.0
    overlap_seconds: float = 5.0
    min_behavior_duration: float = 2.0
    speed_threshold_rest: float = 0.05  # m/s
    speed_threshold_active: float = 0.3  # m/s
    distance_threshold_huddle: float = 0.15  # meters (about 6 inches)
    distance_threshold_social: float = 0.5  # meters for social interactions
    turning_angle_threshold: float = 45.0  # degrees for directional changes
    
    # Behavior display configuration
    behavior_colors: Dict[str, str] = None
    behavior_labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.behavior_colors is None:
            self.behavior_colors = {
                'resting': '#808080',      # Gray
                'slow_exploration': '#90EE90',  # Light green
                'active_movement': '#00FF00',   # Bright green
                'huddling': '#FF69B4',     # Hot pink
                'social_approach': '#87CEEB',  # Sky blue
                'social_avoidance': '#FFA500', # Orange
                'following': '#1E90FF',    # Dodge blue
                'chasing': '#FF4500',      # Orange red
                'unknown': '#D3D3D3'       # Light gray
            }
        
        if self.behavior_labels is None:
            self.behavior_labels = {
                'resting': 'Resting',
                'slow_exploration': 'Exploring',
                'active_movement': 'Active',
                'huddling': 'Huddling',
                'social_approach': 'Approaching',
                'social_avoidance': 'Avoiding',
                'following': 'Following',
                'chasing': 'Chasing',
                'unknown': 'Unknown'
            }

class FeatureExtractor:
    """Extract behavioral features from movement data"""
    
    def __init__(self, config: BehaviorConfig):
        self.config = config
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract behavioral features from movement data
        
        Args:
            data: DataFrame with columns including Timestamp, shortid, location_x/y or smoothed_x/y
            
        Returns:
            DataFrame with extracted features for each time window
        """
        print("Extracting behavioral features...")
        
        # Determine which coordinate columns to use
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        features_list = []
        
        # Get time bounds
        start_time = data['Timestamp'].min()
        end_time = data['Timestamp'].max()
        
        # Create sliding time windows
        current_time = start_time
        window_timedelta = pd.Timedelta(seconds=self.config.window_size_seconds)
        step_timedelta = pd.Timedelta(seconds=self.config.overlap_seconds)
        
        # Calculate total expected windows for progress tracking
        total_duration = (end_time - start_time).total_seconds()
        expected_windows = int(total_duration / (self.config.window_size_seconds - self.config.overlap_seconds)) + 1
        print(f"Expected to process approximately {expected_windows} time windows...")
        
        window_count = 0
        progress_interval = max(1, expected_windows // 20)  # Show progress every ~5%
        
        import time
        start_processing_time = time.time()
        
        while current_time < end_time:
            window_end = current_time + window_timedelta
            
            # Get data for this time window
            window_data = data[
                (data['Timestamp'] >= current_time) & 
                (data['Timestamp'] < window_end)
            ].copy()
            
            if not window_data.empty and len(window_data['shortid'].unique()) > 0:
                window_features = self._extract_window_features(
                    window_data, current_time, x_col, y_col
                )
                if window_features:
                    features_list.extend(window_features)
            
            current_time += step_timedelta
            window_count += 1
            
            # Show progress periodically
            if window_count % progress_interval == 0 or window_count == 1:
                elapsed_time = time.time() - start_processing_time
                progress_pct = min(100, (window_count / expected_windows) * 100)
                print(f"  Progress: {window_count}/{expected_windows} windows ({progress_pct:.1f}%) - "
                      f"Elapsed: {elapsed_time:.1f}s")
        
        processing_time = time.time() - start_processing_time
        
        if features_list:
            features_df = pd.DataFrame(features_list)
            print(f"✅ Feature extraction complete! Processed {len(features_df)} time windows in {processing_time:.2f} seconds")
            return features_df
        else:
            print("No features extracted - returning empty DataFrame")
            return pd.DataFrame()
    
    def _extract_window_features(self, window_data: pd.DataFrame, timestamp: pd.Timestamp, 
                                x_col: str, y_col: str) -> List[Dict]:
        """Extract features for a single time window"""
        features_list = []
        
        # Get unique animals in this window
        animals = window_data['shortid'].unique()
        
        # Calculate individual features for each animal
        individual_features = {}
        for animal_id in animals:
            animal_data = window_data[window_data['shortid'] == animal_id].copy()
            if len(animal_data) > 1:  # Need at least 2 points for movement features
                individual_features[animal_id] = self._calculate_individual_features(
                    animal_data, x_col, y_col
                )
        
        # Calculate pairwise features
        pairwise_features = {}
        for i, animal1 in enumerate(animals):
            for animal2 in animals[i+1:]:
                if animal1 in individual_features and animal2 in individual_features:
                    pair_key = f"{min(animal1, animal2)}_{max(animal1, animal2)}"
                    pairwise_features[pair_key] = self._calculate_pairwise_features(
                        window_data, animal1, animal2, x_col, y_col
                    )
        
        # Create feature records for each animal
        for animal_id in animals:
            if animal_id in individual_features:
                features = {
                    'timestamp': timestamp,
                    'animal_id': animal_id,
                    **individual_features[animal_id]
                }
                
                # Add relevant pairwise features
                for pair_key, pair_feats in pairwise_features.items():
                    animal_ids = [int(x) for x in pair_key.split('_')]
                    if animal_id in animal_ids:
                        partner_id = animal_ids[1] if animal_ids[0] == animal_id else animal_ids[0]
                        for feat_name, feat_value in pair_feats.items():
                            features[f"{feat_name}_partner_{partner_id}"] = feat_value
                
                features_list.append(features)
        
        return features_list
    
    def _calculate_individual_features(self, animal_data: pd.DataFrame, x_col: str, y_col: str) -> Dict:
        """Calculate individual movement features for one animal"""
        features = {}
        
        # Sort by timestamp to ensure correct order
        animal_data = animal_data.sort_values('Timestamp')
        
        # Position data
        x_coords = animal_data[x_col].values
        y_coords = animal_data[y_col].values
        timestamps = animal_data['Timestamp'].values
        
        # Calculate distances between consecutive points
        distances = []
        time_diffs = []
        for i in range(1, len(x_coords)):
            dist = euclidean([x_coords[i-1], y_coords[i-1]], [x_coords[i], y_coords[i]])
            # Handle numpy timedelta64 properly
            time_diff_td = timestamps[i] - timestamps[i-1]
            if hasattr(time_diff_td, 'total_seconds'):
                # Python timedelta
                time_diff = time_diff_td.total_seconds()
            else:
                # numpy timedelta64 - convert to seconds
                time_diff = time_diff_td / pd.Timedelta(seconds=1)
            
            if time_diff > 0:  # Avoid division by zero
                distances.append(dist)
                time_diffs.append(time_diff)
        
        if distances:
            # Speed features
            speeds = [d/t for d, t in zip(distances, time_diffs)]
            features['speed_mean'] = np.mean(speeds)
            features['speed_max'] = np.max(speeds)
            features['speed_std'] = np.std(speeds)
            
            # Distance features
            features['total_distance'] = np.sum(distances)
            features['displacement'] = euclidean([x_coords[0], y_coords[0]], [x_coords[-1], y_coords[-1]])
            
            # Path complexity
            if features['displacement'] > 0:
                features['path_efficiency'] = features['displacement'] / features['total_distance']
            else:
                features['path_efficiency'] = 0.0
            
            # Spatial area (convex hull area approximation)
            if len(x_coords) > 2:
                features['area_covered'] = self._calculate_area_covered(x_coords, y_coords)
            else:
                features['area_covered'] = 0.0
            
            # Turning angles
            turning_angles = self._calculate_turning_angles(x_coords, y_coords)
            if turning_angles:
                features['turning_angle_mean'] = np.mean(np.abs(turning_angles))
                features['turning_angle_std'] = np.std(turning_angles)
            else:
                features['turning_angle_mean'] = 0.0
                features['turning_angle_std'] = 0.0
        else:
            # No movement detected
            features.update({
                'speed_mean': 0.0, 'speed_max': 0.0, 'speed_std': 0.0,
                'total_distance': 0.0, 'displacement': 0.0, 'path_efficiency': 0.0,
                'area_covered': 0.0, 'turning_angle_mean': 0.0, 'turning_angle_std': 0.0
            })
        
        return features
    
    def _calculate_pairwise_features(self, window_data: pd.DataFrame, animal1: int, animal2: int,
                                   x_col: str, y_col: str) -> Dict:
        """Calculate pairwise features between two animals"""
        features = {}
        
        # Get data for both animals
        data1 = window_data[window_data['shortid'] == animal1].sort_values('Timestamp')
        data2 = window_data[window_data['shortid'] == animal2].sort_values('Timestamp')
        
        if len(data1) == 0 or len(data2) == 0:
            return {'distance_mean': np.inf, 'distance_min': np.inf, 'distance_change_rate': 0.0}
        
        # Find overlapping time points (or nearest points)
        distances = []
        distance_changes = []
        
        # Simple approach: calculate distances at all timepoints where both animals have data
        for _, row1 in data1.iterrows():
            # Find closest data point for animal2 in time
            time_diffs = np.abs((data2['Timestamp'] - row1['Timestamp']).dt.total_seconds())
            closest_idx = time_diffs.idxmin()
            row2 = data2.loc[closest_idx]
            
            # Only use if time difference is reasonable (within window)
            if time_diffs[closest_idx] <= self.config.window_size_seconds / 2:
                distance = euclidean([row1[x_col], row1[y_col]], [row2[x_col], row2[y_col]])
                distances.append(distance)
        
        if distances:
            features['distance_mean'] = np.mean(distances)
            features['distance_min'] = np.min(distances)
            features['distance_max'] = np.max(distances)
            features['distance_std'] = np.std(distances)
            
            # Calculate rate of distance change
            if len(distances) > 1:
                distance_changes = np.diff(distances)
                features['distance_change_rate'] = np.mean(distance_changes)
            else:
                features['distance_change_rate'] = 0.0
        else:
            features.update({
                'distance_mean': np.inf, 'distance_min': np.inf, 'distance_max': np.inf,
                'distance_std': 0.0, 'distance_change_rate': 0.0
            })
        
        return features
    
    def _calculate_area_covered(self, x_coords: np.ndarray, y_coords: np.ndarray) -> float:
        """Calculate area covered using convex hull approximation"""
        from scipy.spatial import ConvexHull
        
        try:
            points = np.column_stack([x_coords, y_coords])
            # Remove duplicate points
            unique_points = np.unique(points, axis=0)
            
            if len(unique_points) >= 3:
                hull = ConvexHull(unique_points)
                return hull.volume  # In 2D, volume is area
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_turning_angles(self, x_coords: np.ndarray, y_coords: np.ndarray) -> List[float]:
        """Calculate turning angles between consecutive movement vectors"""
        angles = []
        
        for i in range(1, len(x_coords) - 1):
            # Vector from point i-1 to point i
            v1 = np.array([x_coords[i] - x_coords[i-1], y_coords[i] - y_coords[i-1]])
            # Vector from point i to point i+1
            v2 = np.array([x_coords[i+1] - x_coords[i], y_coords[i+1] - y_coords[i]])
            
            # Calculate angle between vectors
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Prevent numerical errors
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)
        
        return angles

class BehavioralClassifier:
    """Main behavioral classification system"""
    
    def __init__(self, config: Optional[BehaviorConfig] = None):
        self.config = config or BehaviorConfig()
        self.feature_extractor = FeatureExtractor(self.config)
        self.behavior_timeline = None
        
    def analyze_session(self, data: pd.DataFrame, method: str = 'rule_based') -> pd.DataFrame:
        """
        Analyze behavioral patterns in a complete session
        
        Args:
            data: Movement data with required columns
            method: 'rule_based', 'ml', or 'hybrid'
            
        Returns:
            DataFrame with behavioral classifications over time
        """
        print(f"Starting behavioral analysis using {method} method...")
        
        # Extract features
        features = self.feature_extractor.extract_features(data)
        
        if features.empty:
            print("No features extracted - returning empty behavior timeline")
            return pd.DataFrame()
        
        # Classify behaviors
        if method == 'rule_based':
            behaviors = self._rule_based_classification(features)
        elif method == 'ml':
            behaviors = self._ml_classification(features)
        else:  # hybrid
            behaviors = self._hybrid_classification(features)
        
        # Post-process to ensure minimum durations
        behaviors = self._apply_minimum_duration_filter(behaviors)
        
        # Store for later use
        self.behavior_timeline = behaviors
        
        print(f"Behavioral analysis complete. Classified {len(behaviors)} time windows.")
        return behaviors
    
    def _rule_based_classification(self, features: pd.DataFrame) -> pd.DataFrame:
        """Classify behaviors using hardcoded rules"""
        print("Applying rule-based behavioral classification...")
        
        behaviors = []
        total_windows = len(features)
        progress_interval = max(1, total_windows // 10)  # Show progress every ~10%
        
        import time
        start_time = time.time()
        
        for i, (_, row) in enumerate(features.iterrows()):
            animal_id = row['animal_id']
            timestamp = row['timestamp']
            
            # Extract relevant features
            speed_mean = row.get('speed_mean', 0.0)
            speed_max = row.get('speed_max', 0.0)
            path_efficiency = row.get('path_efficiency', 0.0)
            
            # Get minimum distance to any partner
            partner_distances = []
            for col in row.index:
                if col.startswith('distance_mean_partner_'):
                    dist = row[col]
                    if not np.isnan(dist) and not np.isinf(dist):
                        partner_distances.append(dist)
            
            min_partner_distance = min(partner_distances) if partner_distances else np.inf
            
            # Apply rules in order of specificity
            behavior = 'unknown'
            confidence = 0.5
            
            # Rule 1: Resting - very low speed
            if speed_mean < self.config.speed_threshold_rest:
                if min_partner_distance < self.config.distance_threshold_huddle:
                    behavior = 'huddling'
                    confidence = 0.9
                else:
                    behavior = 'resting'
                    confidence = 0.8
            
            # Rule 2: High speed behaviors
            elif speed_mean > self.config.speed_threshold_active:
                # Check for social context
                if min_partner_distance < self.config.distance_threshold_social:
                    # Look at distance change to determine approach vs. avoidance
                    distance_change_rates = []
                    for col in row.index:
                        if col.startswith('distance_change_rate_partner_'):
                            rate = row[col]
                            if not np.isnan(rate):
                                distance_change_rates.append(rate)
                    
                    if distance_change_rates:
                        avg_distance_change = np.mean(distance_change_rates)
                        if avg_distance_change < -0.05:  # Decreasing distance
                            behavior = 'chasing'
                            confidence = 0.7
                        elif avg_distance_change > 0.05:  # Increasing distance
                            behavior = 'social_avoidance'
                            confidence = 0.7
                        else:
                            behavior = 'following'
                            confidence = 0.6
                    else:
                        behavior = 'active_movement'
                        confidence = 0.6
                else:
                    behavior = 'active_movement'
                    confidence = 0.7
            
            # Rule 3: Moderate speed - exploration
            else:
                if path_efficiency < 0.3:  # Tortuous path indicates exploration
                    behavior = 'slow_exploration'
                    confidence = 0.6
                else:
                    behavior = 'active_movement'
                    confidence = 0.5
            
            behaviors.append({
                'timestamp': timestamp,
                'animal_id': animal_id,
                'behavior': behavior,
                'confidence': confidence,
                'speed_mean': speed_mean,
                'min_partner_distance': min_partner_distance,
                'path_efficiency': path_efficiency
            })
            
            # Show progress periodically
            if (i + 1) % progress_interval == 0 or i == 0:
                elapsed_time = time.time() - start_time
                progress_pct = ((i + 1) / total_windows) * 100
                print(f"  Classification progress: {i + 1}/{total_windows} windows ({progress_pct:.1f}%) - "
                      f"Elapsed: {elapsed_time:.1f}s")
        
        classification_time = time.time() - start_time
        behavior_df = pd.DataFrame(behaviors)
        
        print(f"Rule-based classification complete. Behavior distribution:")
        for behavior, count in behavior_df['behavior'].value_counts().items():
            print(f"  {behavior}: {count} windows ({100*count/len(behavior_df):.1f}%)")
        print(f"Classification completed in {classification_time:.2f} seconds")
        
        return behavior_df
    
    def _ml_classification(self, features: pd.DataFrame) -> pd.DataFrame:
        """Classify behaviors using machine learning (placeholder for future implementation)"""
        print("ML classification not yet implemented, using rule-based fallback...")
        return self._rule_based_classification(features)
    
    def _hybrid_classification(self, features: pd.DataFrame) -> pd.DataFrame:
        """Combine rule-based and ML approaches (placeholder for future implementation)"""
        print("Hybrid classification not yet implemented, using rule-based fallback...")
        return self._rule_based_classification(features)
    
    def _apply_minimum_duration_filter(self, behaviors: pd.DataFrame) -> pd.DataFrame:
        """Filter out behaviors that don't meet minimum duration requirements"""
        if behaviors.empty:
            return behaviors
        
        print("Applying minimum duration filter...")
        filtered_behaviors = []
        
        import time
        start_time = time.time()
        
        # Group by animal
        for animal_id in behaviors['animal_id'].unique():
            animal_behaviors = behaviors[behaviors['animal_id'] == animal_id].sort_values('timestamp')
            
            # Group consecutive behaviors
            current_behavior = None
            current_start = None
            current_records = []
            
            for _, row in animal_behaviors.iterrows():
                if row['behavior'] != current_behavior:
                    # Save previous behavior group if it meets duration requirement
                    if current_behavior is not None and current_records:
                        duration = (current_records[-1]['timestamp'] - current_start).total_seconds()
                        if duration >= self.config.min_behavior_duration:
                            filtered_behaviors.extend(current_records)
                        else:
                            # Change short behaviors to 'unknown'
                            for record in current_records:
                                record['behavior'] = 'unknown'
                                record['confidence'] = 0.3
                            filtered_behaviors.extend(current_records)
                    
                    # Start new behavior group
                    current_behavior = row['behavior']
                    current_start = row['timestamp']
                    current_records = [row.to_dict()]
                else:
                    current_records.append(row.to_dict())
            
            # Handle final group
            if current_behavior is not None and current_records:
                duration = (current_records[-1]['timestamp'] - current_start).total_seconds()
                if duration >= self.config.min_behavior_duration:
                    filtered_behaviors.extend(current_records)
                else:
                    for record in current_records:
                        record['behavior'] = 'unknown'
                        record['confidence'] = 0.3
                    filtered_behaviors.extend(current_records)
        
        result = pd.DataFrame(filtered_behaviors)
        
        filtering_time = time.time() - start_time
        print(f"Duration filtering complete. Final behavior distribution:")
        for behavior, count in result['behavior'].value_counts().items():
            print(f"  {behavior}: {count} windows ({100*count/len(result):.1f}%)")
        print(f"Duration filtering completed in {filtering_time:.2f} seconds")
        
        return result
    
    def get_behavior_at_time(self, timestamp: pd.Timestamp, animal_id: int) -> Tuple[str, float]:
        """Get behavior and confidence for a specific animal at a specific time"""
        if self.behavior_timeline is None:
            return 'unknown', 0.0
        
        # Find the closest behavior record in time
        animal_behaviors = self.behavior_timeline[
            self.behavior_timeline['animal_id'] == animal_id
        ]
        
        if animal_behaviors.empty:
            return 'unknown', 0.0
        
        # Find closest timestamp
        time_diffs = np.abs((animal_behaviors['timestamp'] - timestamp).dt.total_seconds())
        closest_idx = time_diffs.idxmin()
        closest_record = animal_behaviors.loc[closest_idx]
        
        # Only return if within reasonable time window
        if time_diffs[closest_idx] <= self.config.window_size_seconds:
            return closest_record['behavior'], closest_record['confidence']
        else:
            return 'unknown', 0.0
    
    def get_current_behaviors(self, timestamp: pd.Timestamp) -> Dict[int, Tuple[str, float]]:
        """Get current behaviors for all animals at a specific timestamp"""
        if self.behavior_timeline is None:
            return {}
        
        current_behaviors = {}
        for animal_id in self.behavior_timeline['animal_id'].unique():
            behavior, confidence = self.get_behavior_at_time(timestamp, animal_id)
            current_behaviors[animal_id] = (behavior, confidence)
        
        return current_behaviors
    
    def save_config(self, filepath: str):
        """Save current configuration to JSON file"""
        config_dict = {
            'window_size_seconds': self.config.window_size_seconds,
            'overlap_seconds': self.config.overlap_seconds,
            'min_behavior_duration': self.config.min_behavior_duration,
            'speed_threshold_rest': self.config.speed_threshold_rest,
            'speed_threshold_active': self.config.speed_threshold_active,
            'distance_threshold_huddle': self.config.distance_threshold_huddle,
            'distance_threshold_social': self.config.distance_threshold_social,
            'turning_angle_threshold': self.config.turning_angle_threshold,
            'behavior_colors': self.config.behavior_colors,
            'behavior_labels': self.config.behavior_labels
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_config(cls, filepath: str) -> 'BehavioralClassifier':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = BehaviorConfig(**config_dict)
        return cls(config)

# Convenience functions for integration with existing code
def analyze_behaviors(data: pd.DataFrame, config_file: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to analyze behaviors in movement data
    
    Args:
        data: Movement data DataFrame
        config_file: Optional path to configuration file
        
    Returns:
        DataFrame with behavioral classifications
    """
    if config_file and os.path.exists(config_file):
        classifier = BehavioralClassifier.load_config(config_file)
    else:
        classifier = BehavioralClassifier()
    
    return classifier.analyze_session(data)

def get_behavior_colors() -> Dict[str, str]:
    """Get default behavior color mapping for plotting"""
    config = BehaviorConfig()
    return config.behavior_colors

def get_behavior_labels() -> Dict[str, str]:
    """Get default behavior label mapping for plotting"""
    config = BehaviorConfig()
    return config.behavior_labels

if __name__ == "__main__":
    # Example usage and testing
    print("Behavioral Classifier Module")
    print("This module provides behavioral analysis for UWB tracking data")
    print("Import this module in your analysis scripts to use behavioral classification")
    
    # Create and save a default configuration
    classifier = BehavioralClassifier()
    config_path = os.path.join(os.path.dirname(__file__), 'default_behavior_config.json')
    classifier.save_config(config_path)
    print(f"Default configuration saved to: {config_path}")
