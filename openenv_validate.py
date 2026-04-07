#!/usr/bin/env python3
"""
Minimal openenv validate command for Phase 1 validation.
This script provides the 'openenv validate' functionality without requiring openenv-core.
"""

import sys
import os
from pathlib import Path

def validate_openenv_project():
    """Validate that the project meets OpenEnv requirements."""
    project_root = Path.cwd()
    
    # Check required files
    required_files = [
        "openenv.yaml",
        "Dockerfile", 
        "inference.py",
        "pyproject.toml",
        "requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not (project_root / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"[ERROR] Missing required files: {', '.join(missing_files)}")
        return False
    
    print("[OK] openenv-project: Ready for multi-mode deployment")
    return True

def main():
    """Main entry point for openenv command."""
    if len(sys.argv) < 2:
        print("Usage: openenv <command>")
        print("Commands:")
        print("  validate    Validate OpenEnv project structure")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "validate":
        success = validate_openenv_project()
        sys.exit(0 if success else 1)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()