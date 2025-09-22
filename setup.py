#!/usr/bin/env python3
"""
Setup script for Semantic Role Labeling project
Handles installation, configuration, and initial setup
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def setup_database():
    """Initialize the database"""
    print("\n🗄️ Setting up database...")
    
    try:
        from mock_database import MockDatabase
        db = MockDatabase()
        stats = db.get_statistics()
        print(f"✅ Database initialized with {stats['total_sentences']} sample sentences")
        return True
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        from modern_srl import SemanticRoleLabeler
        from visualization import SRLVisualizer
        from mock_database import MockDatabase
        
        print("✅ All modules imported successfully")
        
        # Test basic functionality
        db = MockDatabase()
        sentences = db.get_sentences(limit=1)
        if sentences:
            print("✅ Database functionality working")
        
        visualizer = SRLVisualizer()
        print("✅ Visualization tools loaded")
        
        return True
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "logs",
        "models",
        "data",
        "outputs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def setup_git_hooks():
    """Setup git hooks for development"""
    print("\n🔧 Setting up git hooks...")
    
    git_hooks_dir = Path(".git/hooks")
    if git_hooks_dir.exists():
        # Pre-commit hook
        pre_commit_hook = git_hooks_dir / "pre-commit"
        pre_commit_content = """#!/bin/sh
# Run tests before commit
python test_srl.py
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
"""
        pre_commit_hook.write_text(pre_commit_content)
        pre_commit_hook.chmod(0o755)
        print("✅ Pre-commit hook installed")
    else:
        print("ℹ️ Not a git repository, skipping git hooks")

def print_next_steps():
    """Print next steps for the user"""
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the basic demo:")
    print("   python 0108.py")
    print("\n2. Launch the web interface:")
    print("   streamlit run app.py")
    print("\n3. Run tests:")
    print("   python test_srl.py")
    print("\n4. Explore the code:")
    print("   - modern_srl.py (Core SRL implementation)")
    print("   - visualization.py (Visualization tools)")
    print("   - mock_database.py (Database management)")
    print("\n5. Read the documentation:")
    print("   - README.md (Main documentation)")
    print("   - CONTRIBUTING.md (Contributing guidelines)")

def main():
    """Main setup function"""
    print("🚀 Semantic Role Labeling Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed at requirements installation")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Setup database
    if not setup_database():
        print("\n⚠️ Database setup failed, but continuing...")
    
    # Test installation
    if not test_installation():
        print("\n❌ Setup failed at installation test")
        sys.exit(1)
    
    # Setup git hooks
    setup_git_hooks()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
