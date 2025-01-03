# LoRA Image Processor v1.0

A GUI tool for preparing image datasets for LoRA training. Automatically resizes images and generates captions using the BLIP model.

## Features

- Batch process images with automatic resizing
- Generate AI captions using BLIP model
- Maintain aspect ratios with smart padding
- User-friendly GUI interface
- Progress tracking and logging
- Compatible with most LoRA training workflows

## Installation

### Prerequisites
```bash
# Using pip
pip install Pillow torch transformers[torch] accelerate sentencepiece tk

# Using conda
conda install tk pillow pytorch torchvision -c pytorch
conda install transformers -c huggingface
conda install accelerate sentencepiece -c conda-forge
```

### Running from Source
```bash
python app.py
```

## Usage

1. Launch the application
2. Select input directory containing images
3. Choose output directory for processed files
4. Set desired image dimensions (default: 512x512)
5. Click "Start Processing"

## Output Format

For each image (e.g., `image.jpg`), the tool creates:
- Processed image: `image.jpg`
- Caption file: `image.txt`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- BLIP image captioning model by Salesforce
- Transformers library by Hugging Face
