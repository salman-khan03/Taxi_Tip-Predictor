"""
Environment check script.
Verifies that all required dependencies are installed and GPU is available.
"""

import sys

def check_environment():
    """Check environment setup and dependencies."""
    print("=" * 60)
    print("🔍 ENVIRONMENT CHECK")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check Python version
    print("📌 Python Version:")
    print(f"   {sys.version}")
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("   ⚠️  Warning: Python 3.8+ recommended")
        all_ok = False
    else:
        print("   ✅ Python version OK")
    print()
    
    # Check NumPy
    print("📌 NumPy:")
    try:
        import numpy as np
        print(f"   ✅ Version: {np.__version__}")
    except ImportError:
        print("   ❌ Not installed")
        all_ok = False
    print()
    
    # Check Pandas
    print("📌 Pandas:")
    try:
        import pandas as pd
        print(f"   ✅ Version: {pd.__version__}")
    except ImportError:
        print("   ❌ Not installed")
        all_ok = False
    print()
    
    # Check cuDF (optional)
    print("📌 cuDF (RAPIDS - GPU-accelerated Pandas):")
    try:
        import cudf
        print(f"   ✅ Version: {cudf.__version__}")
        print("   ✅ GPU-accelerated data processing available")
    except ImportError:
        print("   ⚠️  Not installed (will use pandas CPU fallback)")
    print()
    
    # Check RMM (optional)
    print("📌 RMM (RAPIDS Memory Manager):")
    try:
        import rmm
        print(f"   ✅ Available")
    except ImportError:
        print("   ⚠️  Not installed (memory management will be handled by system)")
    print()
    
    # Check XGBoost
    print("📌 XGBoost:")
    try:
        import xgboost as xgb
        print(f"   ✅ Version: {xgb.__version__}")
        
        # Check GPU support
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("   ✅ NVIDIA GPU detected")
                # Check if XGBoost can use GPU
                print("   ℹ️  XGBoost GPU support: Available (set device='cuda' in config)")
            else:
                print("   ⚠️  NVIDIA GPU not detected or nvidia-smi not available")
        except:
            print("   ⚠️  Could not check GPU status")
    except ImportError:
        print("   ❌ Not installed - REQUIRED")
        all_ok = False
    print()
    
    # Check scikit-learn
    print("📌 scikit-learn:")
    try:
        import sklearn
        print(f"   ✅ Version: {sklearn.__version__}")
    except ImportError:
        print("   ❌ Not installed - REQUIRED")
        all_ok = False
    print()
    
    # Check Matplotlib
    print("📌 Matplotlib:")
    try:
        import matplotlib
        print(f"   ✅ Version: {matplotlib.__version__}")
    except ImportError:
        print("   ⚠️  Not installed (visualizations will not work)")
    print()
    
    # Check Seaborn
    print("📌 Seaborn:")
    try:
        import seaborn as sns
        print(f"   ✅ Version: {sns.__version__}")
    except ImportError:
        print("   ⚠️  Not installed (some visualizations may not work)")
    print()
    
    # Summary
    print("=" * 60)
    if all_ok:
        print("✅ All required dependencies are installed!")
        print("\n💡 Next steps:")
        print("   1. Download the NYC Taxi dataset")
        print("   2. Place it in the 'data/' directory")
        print("   3. Update config.py if needed")
        print("   4. Run: python main.py")
    else:
        print("❌ Some required dependencies are missing!")
        print("\n💡 Install missing packages:")
        print("   pip install -r requirements.txt")
    print("=" * 60)


if __name__ == "__main__":
    check_environment()
