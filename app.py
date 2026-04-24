"""
HuggingFace Spaces Entry Point
This file is required for HF Spaces deployment
"""

import os
import sys

# Setup environment
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress warnings

# Import the main app
from app.gradio_app import DevDocsApp

def main():
    """Launch the Gradio app for HuggingFace Spaces"""
    
    print("🚀 Starting DevDocs AI on HuggingFace Spaces...")
    
    # Initialize app
    app = DevDocsApp()
    demo = app.create_interface()
    
    # Launch with HF Spaces settings
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Not needed on HF Spaces
        show_error=True
    )

if __name__ == "__main__":
    main()