# Medical Inventory Checker

An AI-powered application for checking medical case inventory using computer vision and multimodal models. Designed for medical equipment cases containing screws, tools, and instruments (similar to Medtronic products).

## Features

- **Visual Inventory Comparison**: Compare before/after images of medical cases
- **AI-Powered Detection**: Uses Hugging Face models with free GPU resources
- **Difference Highlighting**: Visually marks differences between inventory states
- **Database Integration**: Tracks inventory stock with detailed product information
- **Multiple Model Support**: Test different vision models for optimal accuracy

## Tech Stack

- **Python 3.9+**
- **Hugging Face Transformers**: For multimodal AI models
- **Gradio**: Web interface for easy interaction
- **OpenCV**: Image processing and visualization
- **SQLite**: Lightweight database for inventory tracking
- **Pillow**: Image manipulation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/inventory_check.git
cd inventory_check
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Hugging Face token (optional, for private models):
```bash
export HF_TOKEN=your_token_here
```

## Usage

### Quick Start

```bash
python src/app.py
```

This will launch a Gradio interface where you can:
1. Upload a "before" image of the medical case
2. Upload an "after" image of the medical case
3. Select the AI model to use
4. View the analysis results with highlighted differences

### API Usage

```python
from src.models.inventory_checker import InventoryChecker

checker = InventoryChecker(model_name="microsoft/Florence-2-base")
results = checker.compare_inventory(before_image_path, after_image_path)
```

## Project Structure

```
inventory_check/
├── src/
│   ├── app.py                 # Main Gradio application
│   ├── models/
│   │   ├── inventory_checker.py  # Core inventory checking logic
│   │   └── model_loader.py      # Hugging Face model management
│   ├── utils/
│   │   ├── image_processor.py   # Image preprocessing utilities
│   │   └── visualization.py     # Difference visualization tools
│   └── data/
│       ├── inventory_db.py      # Database operations
│       └── sample_inventory.json # Sample inventory data
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## Models Supported

The application supports state-of-the-art vision-language models from Hugging Face:

- **Qwen2.5-VL (3B/7B)**: Latest state-of-the-art multimodal model with excellent object localization
- **Llama 3.2 Vision (11B)**: Meta's powerful vision-language model with strong reasoning
- **Kosmos-2**: Microsoft's grounding-capable model for precise object detection
- **Microsoft Florence-2**: Lightweight model for basic object detection (legacy support)

## Database Schema

The inventory database tracks:
- Product ID
- Product name
- Category (screws, tools, instruments)
- Expected quantity
- Current quantity
- Location in case
- Part number
- Description

## Development

### Running Tests
```bash
pytest tests/
```

### Adding New Models
1. Add model configuration to `src/models/model_loader.py`
2. Implement model-specific processing in `inventory_checker.py`
3. Update the UI in `app.py` to include the new model option

## Roadmap

- [ ] MVP: Basic image comparison with single model
- [ ] Multi-model comparison interface
- [ ] Export reports (PDF, Excel)
- [ ] Real-time inventory tracking
- [ ] Mobile app integration
- [ ] Barcode/QR code support
- [ ] Historical tracking and analytics

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Hugging Face for providing free GPU resources
- Medical professionals who provided domain expertise
- Open source community for the amazing tools