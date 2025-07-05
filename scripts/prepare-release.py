#!/usr/bin/env python3
"""Comprehensive script to prepare DeepSentinel SDK for PyPI release."""

import subprocess
import sys
from pathlib import Path


def run_script(script_name: str, project_root: Path) -> bool:
    """Run a script and return success status."""
    script_path = project_root / "scripts" / script_name
    
    print(f"🔄 Running {script_name}...")
    try:
        subprocess.run([
            sys.executable,
            str(script_path)
        ], cwd=project_root, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {script_name} failed")
        return False


def check_version_consistency(project_root: Path) -> bool:
    """Check that version is consistent across files."""
    print("🔍 Checking version consistency...")
    
    init_file = project_root / "src" / "deepsentinel" / "__init__.py"
    if not init_file.exists():
        print("✗ __init__.py not found")
        return False
    
    # Read version from __init__.py
    with open(init_file, 'r') as f:
        content = f.read()
        
    import re
    version_pattern = r'__version__\s*=\s*["\']([^"\']+)["\']'
    version_match = re.search(version_pattern, content)
    if not version_match:
        print("✗ No version found in __init__.py")
        return False
    
    version = version_match.group(1)
    print(f"✓ Found version: {version}")
    
    # Check that version follows semantic versioning pattern
    if not re.match(r'^\d+\.\d+\.\d+(-\w+)?$', version):
        print(f"⚠️  Version '{version}' doesn't follow semantic versioning")
        return False
    
    return True


def main():
    """Main preparation process."""
    print("🚀 Preparing DeepSentinel SDK for PyPI release")
    print("=" * 60)
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print(f"Project root: {project_root}")
    
    # Step 1: Check version consistency
    if not check_version_consistency(project_root):
        print("❌ Version consistency check failed")
        sys.exit(1)
    
    # Step 2: Build package
    if not run_script("build-package.py", project_root):
        sys.exit(1)
    
    # Step 3: Test installation
    if not run_script("test-installation.py", project_root):
        sys.exit(1)
    
    print("=" * 60)
    print("🎉 Package preparation completed successfully!")
    print()
    print("📋 Next steps for PyPI publication:")
    print("  1. Review the built package in dist/ directory")
    print("  2. Test upload to TestPyPI:")
    print("     python -m twine upload --repository testpypi dist/*")
    print("  3. Test installation from TestPyPI:")
    print("     pip install --index-url https://test.pypi.org/simple/ "
          "deepsentinel")
    print("  4. Upload to PyPI:")
    print("     python -m twine upload dist/*")
    print()
    print("🔐 Make sure you have configured your PyPI credentials:")
    print("  - Create ~/.pypirc with your API token")
    print("  - Or set TWINE_USERNAME and TWINE_PASSWORD environment variables")


if __name__ == "__main__":
    main()