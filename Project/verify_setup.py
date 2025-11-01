#!/usr/bin/env python3
"""
Setup Verification Script
=========================

Verifies that the project skeleton is correctly set up and all modules
can be imported.

Run this before starting implementation to ensure everything is configured.
"""

import sys
from pathlib import Path

def check_structure():
    """Check if all directories exist."""
    print("=" * 60)
    print("CHECKING PROJECT STRUCTURE")
    print("=" * 60)
    
    expected_dirs = [
        'src',
        'data',
        'outputs',
        'outputs/figures'
    ]
    
    all_good = True
    for dir_path in expected_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ - MISSING")
            all_good = False
    
    return all_good


def check_modules():
    """Check if all modules can be imported."""
    print("\n" + "=" * 60)
    print("CHECKING MODULE IMPORTS")
    print("=" * 60)
    
    modules = [
        'src',
        'src.data_processing',
        'src.state_action',
        'src.mdp',
        'src.success_modeling',
        'src.analysis',
        'src.visualization'
    ]
    
    all_good = True
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except ImportError as e:
            print(f"✗ {module_name} - FAILED: {e}")
            all_good = False
    
    return all_good


def check_dependencies():
    """Check if required external libraries are available."""
    print("\n" + "=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60)
    
    required = [
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scipy',
        'mplsoccer'
    ]
    
    all_good = True
    for lib in required:
        try:
            __import__(lib)
            print(f"✓ {lib}")
        except ImportError:
            print(f"✗ {lib} - NOT INSTALLED")
            all_good = False
    
    return all_good


def test_basic_functionality():
    """Test basic functionality of key modules."""
    print("\n" + "=" * 60)
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 60)
    
    try:
        from src.state_action import FieldGrid, ACTION_NAMES
        
        # Test grid creation
        grid = FieldGrid(n_rows=17, n_cols=22)
        assert grid.n_states == 374, "Grid should have 374 states"
        print(f"✓ FieldGrid: {grid.n_rows}×{grid.n_cols} = {grid.n_states} states")
        
        # Test coordinate conversion
        x, y = 52.5, 34.0
        state = grid.xy_to_state(x, y)
        x_back, y_back = grid.state_to_xy(state)
        assert 0 <= state < 374, "State should be in valid range"
        print(f"✓ Coordinate conversion: ({x}, {y}) → {state} → ({x_back:.2f}, {y_back:.2f})")
        
        # Test action names
        assert len(ACTION_NAMES) == 9, "Should have 9 actions"
        print(f"✓ Action space: {len(ACTION_NAMES)} actions defined")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("\n🚀 PROJECT SETUP VERIFICATION\n")
    
    results = []
    results.append(("Structure", check_structure()))
    results.append(("Modules", check_modules()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Functionality", test_basic_functionality()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = all(result for _, result in results)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s}: {status}")
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All checks passed! Ready to start implementation.")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
