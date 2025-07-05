#!/usr/bin/env python3
"""Build script for DeepSentinel SDK package."""

import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path = None) -> bool:
    """Run a command and return success status."""
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"✓ {' '.join(cmd)}")
        if result.stdout.strip():
            print(f"  {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {' '.join(cmd)}")
        if e.stdout:
            print(f"  stdout: {e.stdout.strip()}")
        if e.stderr:
            print(f"  stderr: {e.stderr.strip()}")
        return False


def clean_build_artifacts(project_root: Path) -> None:
    """Clean previous build artifacts."""
    print("🧹 Cleaning build artifacts...")
    
    artifacts = [
        project_root / "build",
        project_root / "dist",
        project_root / "src" / "deepsentinel.egg-info",
        project_root / "deepsentinel.egg-info",
    ]
    
    for artifact in artifacts:
        if artifact.exists():
            if artifact.is_dir():
                shutil.rmtree(artifact)
            else:
                artifact.unlink()
            print(f"  Removed: {artifact}")


def verify_package_structure(project_root: Path) -> bool:
    """Verify the package structure is correct."""
    print("🔍 Verifying package structure...")
    
    required_files = [
        project_root / "pyproject.toml",
        project_root / "README.md",
        project_root / "LICENSE",
        project_root / "src" / "deepsentinel" / "__init__.py",
    ]
    
    for file_path in required_files:
        if not file_path.exists():
            print(f"✗ Missing required file: {file_path}")
            return False
        print(f"✓ Found: {file_path}")
    
    return True


def build_package(project_root: Path) -> bool:
    """Build the package using hatchling."""
    print("🔨 Building package...")
    
    return run_command(
        [sys.executable, "-m", "build"],
        cwd=project_root
    )


def verify_build_output(project_root: Path) -> bool:
    """Verify the build produced expected artifacts."""
    print("✅ Verifying build output...")
    
    dist_dir = project_root / "dist"
    if not dist_dir.exists():
        print("✗ No dist directory found")
        return False
    
    files = list(dist_dir.glob("*"))
    if not files:
        print("✗ No files in dist directory")
        return False
    
    # Should have both wheel and source distribution
    wheel_files = list(dist_dir.glob("*.whl"))
    tar_files = list(dist_dir.glob("*.tar.gz"))
    
    if not wheel_files:
        print("✗ No wheel file found")
        return False
    
    if not tar_files:
        print("✗ No source distribution found")
        return False
    
    print(f"✓ Wheel: {wheel_files[0].name}")
    print(f"✓ Source dist: {tar_files[0].name}")
    
    return True


def main():
    """Main build process."""
    print("🚀 Building DeepSentinel SDK package")
    print("=" * 50)
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print(f"Project root: {project_root}")
    
    # Step 1: Verify package structure
    if not verify_package_structure(project_root):
        print("❌ Package structure verification failed")
        sys.exit(1)
    
    # Step 2: Clean previous builds
    clean_build_artifacts(project_root)
    
    # Step 3: Build package
    if not build_package(project_root):
        print("❌ Package build failed")
        sys.exit(1)
    
    # Step 4: Verify build output
    if not verify_build_output(project_root):
        print("❌ Build output verification failed")
        sys.exit(1)
    
    print("=" * 50)
    print("🎉 Package build completed successfully!")
    print(f"📦 Artifacts available in: {project_root / 'dist'}")


if __name__ == "__main__":
    main()