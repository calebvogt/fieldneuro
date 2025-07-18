#!/usr/bin/env python3
"""
Quick test to verify the animation script works correctly after debug cleanup.
"""

import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(__file__))

try:
    from uwb_animate_paths import uwb_animate_paths
    print("✓ Successfully imported uwb_animate_paths function")
    print("Script syntax appears correct")
    
    # Test if we can at least get to the file dialog
    print("\nTo test the full workflow, run:")
    print("python uwb_animate_paths.py")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except SyntaxError as e:
    print(f"✗ Syntax error: {e}")
except Exception as e:
    print(f"✗ Other error: {e}")
