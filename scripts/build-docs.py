#!/usr/bin/env python3
"""Build script for DeepSentinel documentation."""

import argparse
import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Optional


def run_command(cmd: list, cwd: Path = None) -> bool:
    """Run a command and return success status."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        return False


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    required_packages = [
        "mkdocs",
        "mkdocs-material",
        "mkdocstrings[python]",
        "mkdocs-gen-files",
        "mkdocs-literate-nav",
        "mkdocs-section-index",
        "pytest-check-links"
    ]
    
    print("Checking documentation dependencies...")
    
    for package in required_packages:
        try:
            __import__(package.split('[')[0].replace('-', '_'))
        except ImportError:
            print(f"❌ Missing dependency: {package}")
            print(f"Install with: pip install {package}")
            return False
    
    print("✅ All dependencies are installed")
    return True


def check_links(docs_dir: Path) -> bool:
    """Check links in documentation to ensure they are valid."""
    print("Checking links in documentation...")
    
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "--check-links", str(docs_dir), "-v"
            ],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print("✅ All links are valid")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Some links are broken:")
        print(e.stdout)
        return False


def validate_docs(docs_dir: Path) -> bool:
    """Validate documentation structure and formatting."""
    print("Validating documentation structure...")
    
    # Check for required files
    required_files = [
        docs_dir / "index.md",
        docs_dir / "quickstart.md",
        docs_dir / "faq.md"
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("❌ Missing required documentation files:")
        for file in missing_files:
            print(f"  - {file.relative_to(docs_dir.parent)}")
        return False
    
    # Validate structure exists for all major sections
    required_dirs = [
        docs_dir / "concepts",
        docs_dir / "reference",
        docs_dir / "guides",
        docs_dir / "tutorials"
    ]
    
    missing_dirs = [
        d for d in required_dirs
        if not d.exists() or not any(d.iterdir())
    ]
    if missing_dirs:
        print("❌ Missing or empty required documentation directories:")
        for directory in missing_dirs:
            print(f"  - {directory.relative_to(docs_dir.parent)}")
        return False
    
    print("✅ Documentation structure validation passed")
    return True


def build_docs(
    serve: bool = False,
    clean: bool = False,
    validate: bool = False,
    check_links_flag: bool = False,
    port: int = 8000,
    open_browser: bool = False,
    strict: bool = False
) -> bool:
    """Build the documentation."""
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"
    
    if not check_dependencies():
        return False
    
    # Validate documentation if requested
    if validate:
        if not validate_docs(docs_dir):
            print("Documentation validation failed.")
            return False
    
    # Check links if requested
    if check_links_flag:
        if not check_links(docs_dir):
            print("Link checking failed.")
            if strict:
                return False
            else:
                print(
                    "Continuing despite link errors "
                    "(strict mode not enabled)..."
                )
    
    # Clean previous build if requested
    if clean:
        print("Cleaning previous build...")
        site_dir = project_root / "site"
        if site_dir.exists():
            import shutil
            shutil.rmtree(site_dir)
    
    # Change to project directory
    os.chdir(project_root)
    
    if serve:
        print(f"Starting development server on port {port}...")
        print(f"Documentation will be available at http://127.0.0.1:{port}")
        print("Press Ctrl+C to stop the server")
        
        if open_browser:
            webbrowser.open(f"http://127.0.0.1:{port}")
        
        return run_command([
            sys.executable, "-m", "mkdocs", "serve",
            "--dev-addr", f"127.0.0.1:{port}",
            "--dirtyreload"  # Enable faster reloading
        ])
    else:
        print("Building documentation...")
        
        cmd = [sys.executable, "-m", "mkdocs", "build"]
        if strict:
            cmd.append("--strict")
        
        success = run_command(cmd)
        
        if success:
            print("✅ Documentation built successfully!")
            print(f"📁 Output directory: {project_root / 'site'}")
            print("🌐 Open site/index.html to view documentation")
        
        return success


def deploy_docs(version: Optional[str] = None) -> bool:
    """Deploy documentation to GitHub Pages using mike for versioning."""
    project_root = Path(__file__).parent.parent
    
    # Change to project directory
    os.chdir(project_root)
    
    cmd = [sys.executable, "-m", "mike", "deploy"]
    
    if version:
        cmd.extend([version])
        # If it's a release version, make it the latest
        if not version.startswith("dev"):
            cmd.extend(["latest"])
    else:
        # If no version specified, deploy as dev
        cmd.extend(["dev"])
    
    cmd.extend(["--update-aliases", "--push"])
    
    version_str = f' as version {version}' if version else ''
    print(f"Deploying documentation{version_str} to GitHub Pages...")
    
    success = run_command(cmd)
    
    if success:
        print("✅ Documentation deployed successfully!")
        print("Visit https://deepsentinel-ai.github.io/deepsentinel-python to view the documentation")
    
    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build and manage DeepSentinel documentation"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start development server instead of building"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean previous build before building"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate documentation structure and content"
    )
    parser.add_argument(
        "--check-links",
        action="store_true",
        help="Check for broken links in the documentation"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to use for the development server (default: 8000)"
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open browser after starting the development server"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict mode for building (fails on warnings)"
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy documentation to GitHub Pages"
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Version to deploy (for use with --deploy)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.deploy:
            success = deploy_docs(version=args.version)
        else:
            success = build_docs(
                serve=args.serve,
                clean=args.clean,
                validate=args.validate,
                check_links_flag=args.check_links,
                port=args.port,
                open_browser=args.open,
                strict=args.strict
            )
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🔍 Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()