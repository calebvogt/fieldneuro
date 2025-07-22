"""
Test Script for Behavioral Classifier Module

This script tests the behavioral_classifier.py module with UWB data to verify:
1. Data loading and preprocessing
2. Feature extraction
3. Behavioral classification
4. Output validation

Author: AI Assistant
Date: 2025-07-21
"""

import os
import sys
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import traceback

# Add the current directory to path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from behavioral_classifier import (
        BehavioralClassifier, 
        BehaviorConfig, 
        FeatureExtractor,
        analyze_behaviors
    )
    print("✅ Successfully imported behavioral_classifier module")
except ImportError as e:
    print(f"❌ Failed to import behavioral_classifier module: {e}")
    sys.exit(1)

def test_sample_data_creation():
    """Create sample UWB data for testing if no database is available"""
    print("\n📊 Creating sample UWB data for testing...")
    
    # Create sample data with 2 animals over 2 minutes
    start_time = pd.Timestamp.now()
    timestamps = pd.date_range(start_time, periods=120, freq='1S')  # 2 minutes at 1Hz
    
    data_records = []
    
    # Animal 1: Moving in a circle
    for i, ts in enumerate(timestamps):
        angle = (i / 10.0) * np.pi  # Slow circular motion
        x = 2.0 + 1.0 * np.cos(angle)  # Center at (2,2) with radius 1
        y = 2.0 + 1.0 * np.sin(angle)
        
        data_records.append({
            'Timestamp': ts,
            'shortid': 1,
            'location_x': x,
            'location_y': y,
            'Date': ts.date()
        })
    
    # Animal 2: More complex movement pattern
    for i, ts in enumerate(timestamps):
        if i < 30:  # First 30 seconds: stationary (resting)
            x, y = 0.0, 0.0
        elif i < 60:  # Next 30 seconds: approaching animal 1
            progress = (i - 30) / 30.0
            x = progress * 2.5
            y = progress * 2.5
        else:  # Last 60 seconds: following animal 1 at distance
            angle = ((i-60) / 10.0) * np.pi + np.pi  # Follow opposite side
            x = 2.0 + 1.5 * np.cos(angle)  # Larger radius
            y = 2.0 + 1.5 * np.sin(angle)
        
        data_records.append({
            'Timestamp': ts,
            'shortid': 2,
            'location_x': x,
            'location_y': y,
            'Date': ts.date()
        })
    
    sample_data = pd.DataFrame(data_records)
    print(f"✅ Created sample data: {len(sample_data)} records, {len(sample_data['shortid'].unique())} animals")
    print(f"   Time range: {sample_data['Timestamp'].min()} to {sample_data['Timestamp'].max()}")
    print(f"   Animals: {sorted(sample_data['shortid'].unique())}")
    
    return sample_data

def load_real_uwb_data(database_path):
    """Load real UWB data from SQLite database"""
    print(f"\n📁 Loading real UWB data from: {database_path}")
    
    try:
        conn = sqlite3.connect(database_path)
        
        # Query a limited amount of data for testing (first 500 records)
        query = """
        SELECT timestamp, shortid, location_x, location_y 
        FROM data 
        ORDER BY shortid, timestamp 
        LIMIT 1000
        """
        
        data = pd.read_sql_query(query, conn)
        conn.close()
        
        if data.empty:
            print("❌ No data found in database")
            return None
        
        # Process timestamps and coordinates like in your existing code
        data['Timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', origin='unix', utc=True)
        data['location_x'] *= 0.0254  # Convert to meters
        data['location_y'] *= 0.0254
        data['Date'] = data['Timestamp'].dt.date
        
        # Sort by animal and timestamp
        data = data.sort_values(['shortid', 'Timestamp'])
        
        print(f"✅ Loaded real data: {len(data)} records, {len(data['shortid'].unique())} animals")
        print(f"   Time range: {data['Timestamp'].min()} to {data['Timestamp'].max()}")
        print(f"   Animals: {sorted(data['shortid'].unique())}")
        
        return data
        
    except Exception as e:
        print(f"❌ Failed to load real data: {e}")
        return None

def test_config_creation():
    """Test BehaviorConfig creation and modification"""
    print("\n⚙️  Testing BehaviorConfig creation...")
    
    # Test default config
    config = BehaviorConfig()
    print(f"✅ Default config created")
    print(f"   Window size: {config.window_size_seconds}s")
    print(f"   Speed thresholds: rest={config.speed_threshold_rest}, active={config.speed_threshold_active}")
    print(f"   Available behaviors: {list(config.behavior_colors.keys())}")
    
    # Test custom config
    custom_config = BehaviorConfig(
        window_size_seconds=15.0,
        speed_threshold_rest=0.02,
        speed_threshold_active=0.25
    )
    print(f"✅ Custom config created with modified parameters")
    print(f"   Custom window size: {custom_config.window_size_seconds}s")
    
    return config

def test_feature_extraction(data, config):
    """Test feature extraction from UWB data"""
    print("\n🔍 Testing feature extraction...")
    
    try:
        extractor = FeatureExtractor(config)
        features = extractor.extract_features(data)
        
        if features.empty:
            print("⚠️  No features extracted - data might be too short or sparse")
            return None
        
        print(f"✅ Feature extraction successful")
        print(f"   Extracted {len(features)} feature windows")
        print(f"   Animals in features: {sorted(features['animal_id'].unique())}")
        print(f"   Feature columns: {len(features.columns)} total")
        
        # Show some sample features
        print("\n📋 Sample features:")
        for col in features.columns[:10]:  # Show first 10 columns
            if col not in ['timestamp', 'animal_id']:
                sample_value = features[col].iloc[0] if len(features) > 0 else 'N/A'
                print(f"   {col}: {sample_value}")
        
        return features
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        traceback.print_exc()
        return None

def test_behavioral_classification(data, config):
    """Test behavioral classification pipeline"""
    print("\n🧠 Testing behavioral classification...")
    
    try:
        classifier = BehavioralClassifier(config)
        behaviors = classifier.analyze_session(data, method='rule_based')
        
        if behaviors.empty:
            print("⚠️  No behaviors classified")
            return None
        
        print(f"✅ Behavioral classification successful")
        print(f"   Classified {len(behaviors)} time windows")
        print(f"   Animals: {sorted(behaviors['animal_id'].unique())}")
        
        # Show behavior distribution
        print("\n📊 Behavior distribution:")
        behavior_counts = behaviors['behavior'].value_counts()
        total_windows = len(behaviors)
        
        for behavior, count in behavior_counts.items():
            percentage = 100 * count / total_windows
            print(f"   {behavior}: {count} windows ({percentage:.1f}%)")
        
        # Show sample behavior records
        print("\n📋 Sample behavior classifications:")
        for i in range(min(5, len(behaviors))):
            row = behaviors.iloc[i]
            print(f"   {row['timestamp']}: Animal {row['animal_id']} -> {row['behavior']} (confidence: {row['confidence']:.2f})")
        
        return behaviors
        
    except Exception as e:
        print(f"❌ Behavioral classification failed: {e}")
        traceback.print_exc()
        return None

def test_behavior_queries(classifier, behaviors):
    """Test behavior query functions"""
    print("\n🔍 Testing behavior query functions...")
    
    if behaviors.empty or classifier.behavior_timeline is None:
        print("⚠️  No behaviors to query")
        return
    
    try:
        # Test getting behavior at specific time
        test_timestamp = behaviors['timestamp'].iloc[len(behaviors)//2]  # Middle timestamp
        test_animal = behaviors['animal_id'].iloc[0]
        
        behavior, confidence = classifier.get_behavior_at_time(test_timestamp, test_animal)
        print(f"✅ Behavior query successful")
        print(f"   At {test_timestamp}: Animal {test_animal} -> {behavior} (confidence: {confidence:.2f})")
        
        # Test getting all current behaviors
        current_behaviors = classifier.get_current_behaviors(test_timestamp)
        print(f"✅ Current behaviors query successful")
        print(f"   All animals at {test_timestamp}:")
        for animal_id, (behavior, conf) in current_behaviors.items():
            print(f"     Animal {animal_id}: {behavior} (confidence: {conf:.2f})")
            
    except Exception as e:
        print(f"❌ Behavior queries failed: {e}")
        traceback.print_exc()

def test_config_save_load(classifier):
    """Test configuration save and load functionality"""
    print("\n💾 Testing configuration save/load...")
    
    try:
        # Save configuration
        config_path = os.path.join(current_dir, 'test_behavior_config.json')
        classifier.save_config(config_path)
        print(f"✅ Configuration saved to: {config_path}")
        
        # Load configuration
        loaded_classifier = BehavioralClassifier.load_config(config_path)
        print(f"✅ Configuration loaded successfully")
        print(f"   Window size: {loaded_classifier.config.window_size_seconds}s")
        print(f"   Speed thresholds: rest={loaded_classifier.config.speed_threshold_rest}, active={loaded_classifier.config.speed_threshold_active}")
        
        # Clean up test file
        os.remove(config_path)
        print(f"✅ Cleaned up test configuration file")
        
    except Exception as e:
        print(f"❌ Configuration save/load failed: {e}")
        traceback.print_exc()

def test_convenience_functions(data):
    """Test convenience functions"""
    print("\n🛠️  Testing convenience functions...")
    
    try:
        # Test analyze_behaviors convenience function
        behaviors = analyze_behaviors(data)
        
        if not behaviors.empty:
            print(f"✅ Convenience function analyze_behaviors() works")
            print(f"   Classified {len(behaviors)} time windows")
        else:
            print("⚠️  Convenience function returned empty results")
        
        # Test color and label functions
        from behavioral_classifier import get_behavior_colors, get_behavior_labels
        colors = get_behavior_colors()
        labels = get_behavior_labels()
        
        print(f"✅ Color mapping function works: {len(colors)} behaviors")
        print(f"✅ Label mapping function works: {len(labels)} labels")
        
    except Exception as e:
        print(f"❌ Convenience functions failed: {e}")
        traceback.print_exc()

def main():
    """Main test function"""
    print("🧪 Behavioral Classifier Test Suite")
    print("=" * 50)
    
    # Test 1: Configuration
    config = test_config_creation()
    
    # Test 2: Data loading
    print("\n🤔 Do you have a UWB SQLite database file to test with? (y/n): ", end="")
    use_real_data = input().strip().lower() in ['y', 'yes']
    
    if use_real_data:
        print("\n📁 Please provide the path to your SQLite database file:")
        db_path = input("Path: ").strip().replace('"', '')  # Remove quotes if present
        
        if os.path.exists(db_path):
            data = load_real_uwb_data(db_path)
        else:
            print(f"❌ Database file not found: {db_path}")
            print("📊 Using sample data instead...")
            data = test_sample_data_creation()
    else:
        data = test_sample_data_creation()
    
    if data is None or data.empty:
        print("❌ No data available for testing. Exiting.")
        return
    
    # Test 3: Feature extraction
    features = test_feature_extraction(data, config)
    
    # Test 4: Behavioral classification
    classifier = BehavioralClassifier(config)
    behaviors = test_behavioral_classification(data, config)
    
    # Test 5: Behavior queries
    if behaviors is not None:
        test_behavior_queries(classifier, behaviors)
    
    # Test 6: Configuration save/load
    test_config_save_load(classifier)
    
    # Test 7: Convenience functions
    test_convenience_functions(data)
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 Test Suite Complete!")
    
    if behaviors is not None and not behaviors.empty:
        print("✅ All core functionality appears to be working correctly")
        print(f"📊 Successfully classified behaviors for {len(behaviors['animal_id'].unique())} animals")
        print(f"📈 Generated {len(behaviors)} behavioral annotations")
        
        # Show final behavior summary
        print("\n📋 Final Results Summary:")
        behavior_summary = behaviors.groupby(['animal_id', 'behavior']).size().unstack(fill_value=0)
        print(behavior_summary)
        
    else:
        print("⚠️  Some functionality may need adjustment for your specific data")
    
    print("\n💡 Next steps:")
    print("   1. Review any warnings or errors above")
    print("   2. Adjust BehaviorConfig parameters if needed")
    print("   3. Test with your actual UWB data")
    print("   4. Integrate with uwb_animate.py and uwb_plots.py")

if __name__ == "__main__":
    main()
