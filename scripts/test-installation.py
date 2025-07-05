#!/usr/bin/env python3
"""Test script to verify DeepSentinel SDK package installation."""

import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def run_command(cmd: list[str], cwd: Path = None) -> tuple[bool, str, str]:
    """Run a command and return success status with output."""
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            check=True, 
            capture_output=True, 
            text=True
        )
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr


def create_test_venv(venv_path: Path) -> bool:
    """Create a test virtual environment."""
    print(f"🔧 Creating test virtual environment at {venv_path}")
    
    try:
        venv.create(venv_path, with_pip=True)
        print("✓ Virtual environment created")
        return True
    except Exception as e:
        print(f"✗ Failed to create virtual environment: {e}")
        return False


def get_venv_python(venv_path: Path) -> Path:
    """Get the Python executable path in the virtual environment."""
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"


def install_package(venv_python: Path, package_path: Path) -> bool:
    """Install the package in the test environment."""
    print(f"📦 Installing package from {package_path}")
    
    success, stdout, stderr = run_command([
        str(venv_python), 
        "-m", 
        "pip", 
        "install", 
        str(package_path)
    ])
    
    if success:
        print("✓ Package installed successfully")
        return True
    else:
        print("✗ Package installation failed")
        print(f"  stdout: {stdout}")
        print(f"  stderr: {stderr}")
        return False


def test_basic_import(venv_python: Path) -> bool:
    """Test basic package import."""
    print("🧪 Testing basic import...")
    
    test_code = """
import deepsentinel
print(f"DeepSentinel version: {deepsentinel.__version__}")
print("✓ Basic import successful")
"""
    
    success, stdout, stderr = run_command([
        str(venv_python), 
        "-c", 
        test_code
    ])
    
    if success:
        print(stdout.strip())
        return True
    else:
        print("✗ Basic import failed")
        print(f"  stderr: {stderr}")
        return False


def test_main_components(venv_python: Path) -> bool:
    """Test importing main components."""
    print("🧪 Testing main component imports...")
    
    test_code = """
try:
    from deepsentinel import (
        SentinelClient,
        ComplianceEngine,
        OpenAIProvider,
        AnthropicProvider,
        SentinelConfig
    )
    print("✓ Main components import successful")
    
    # Test basic configuration
    config = SentinelConfig()
    print(f"✓ Configuration object created: {type(config).__name__}")
    
    print("✓ All main component tests passed")
    
except Exception as e:
    print(f"✗ Component import failed: {e}")
    exit(1)
"""
    
    success, stdout, stderr = run_command([
        str(venv_python), 
        "-c", 
        test_code
    ])
    
    if success:
        print(stdout.strip())
        return True
    else:
        print("✗ Component import failed")
        print(f"  stderr: {stderr}")
        return False


def test_optional_dependencies(venv_python: Path) -> bool:
    """Test that optional dependencies work correctly."""
    print("🧪 Testing optional dependencies...")
    
    # Test that core functionality works without optional deps
    test_code = """
try:
    import deepsentinel
    from deepsentinel import SentinelClient, SentinelConfig
    
    # This should work without optional dependencies
    config = SentinelConfig()
    print("✓ Core functionality works without optional dependencies")
    
except Exception as e:
    print(f"✗ Core functionality failed: {e}")
    exit(1)
"""
    
    success, stdout, stderr = run_command([
        str(venv_python), 
        "-c", 
        test_code
    ])
    
    if success:
        print(stdout.strip())
        return True
    else:
        print("✗ Optional dependency test failed")
        print(f"  stderr: {stderr}")
        return False


def main():
    """Main test process."""
    print("🧪 Testing DeepSentinel SDK package installation")
    print("=" * 60)
    
    # Get project root and find the built package
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    dist_dir = project_root / "dist"
    
    if not dist_dir.exists():
        print("❌ No dist directory found. Please build the package first.")
        print("Run: python scripts/build-package.py")
        sys.exit(1)
    
    # Find the wheel file
    wheel_files = list(dist_dir.glob("*.whl"))
    if not wheel_files:
        print("❌ No wheel file found in dist directory")
        sys.exit(1)
    
    package_path = wheel_files[0]  # Use the first wheel file
    print(f"📦 Testing package: {package_path.name}")
    
    # Create temporary directory for test environment
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_path = Path(temp_dir) / "test_env"
        
        # Step 1: Create test virtual environment
        if not create_test_venv(venv_path):
            print("❌ Virtual environment creation failed")
            sys.exit(1)
        
        venv_python = get_venv_python(venv_path)
        
        # Step 2: Install the package
        if not install_package(venv_python, package_path):
            print("❌ Package installation failed")
            sys.exit(1)
        
        # Step 3: Test basic import
        if not test_basic_import(venv_python):
            print("❌ Basic import test failed")
            sys.exit(1)
        
        # Step 4: Test main components
        if not test_main_components(venv_python):
            print("❌ Main component test failed")
            sys.exit(1)
        
        # Step 5: Test optional dependencies
        if not test_optional_dependencies(venv_python):
            print("❌ Optional dependency test failed")
            sys.exit(1)
    
    print("=" * 60)
    print("🎉 All installation tests passed!")
    print("✅ Package is ready for PyPI distribution")


if __name__ == "__main__":
    main()