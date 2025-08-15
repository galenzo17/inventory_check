#!/usr/bin/env python3
"""
Medical Inventory Checker - Hugging Face Spaces App
Main entry point for the Gradio application
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and run the main app
from src.app import create_interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch()