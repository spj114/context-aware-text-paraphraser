<div align="center">

# 🔄 Context-Aware Text Paraphrasing Tool

### AI-Powered Text Transformation using T5 Transformer

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Hugging_Face-yellow)](https://huggingface.co/spaces/shreyas114/context-aware-text-paraphraser)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

# 🔄 AI Text Paraphraser & Rewriter

A sophisticated context-aware text paraphrasing and rewriting tool powered by the **Humarin/chatgpt_paraphraser_on_T5_base** model from HuggingFace Transformers. This application provides an intuitive web interface for transforming text with AI-powered paraphrasing capabilities.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.30+-green.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)
![Model](https://img.shields.io/badge/Model-Humarin%20T5%20Paraphraser-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

### 🎯 Core Functionality
- **AI-Powered Paraphrasing**: Uses Humarin's fine-tuned T5-base model specifically trained for paraphrasing
- **Style Options**: Choose from Formal, Casual, Academic, or Simple writing styles (UI preserved for user preference)
- **Creativity Control**: Adjustable temperature slider (0.3-1.5) for controlling output creativity and variation
- **Real-time Processing**: Fast paraphrasing with GPU acceleration support
- **Clean Output**: Model generates only paraphrased content without meta-commentary or instructions

### 🖥️ User Interface
- **Clean Web Interface**: Professional Gradio-based UI with modern styling
- **Side-by-Side Comparison**: View original and paraphrased text simultaneously
- **Word Count Display**: Real-time word counting for both input and output
- **Copy to Clipboard**: One-click copying of paraphrased text
- **Example Texts**: Pre-loaded examples to test different content types

### 🛠️ Additional Features
- **Error Handling**: Comprehensive error management and user feedback
- **Loading Indicators**: Visual feedback during processing
- **Responsive Design**: Works on desktop and mobile devices
- **GPU Support**: Automatic GPU detection and utilization

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended for better performance)
- Internet connection for model download
- Optional: CUDA-compatible GPU for faster processing

### Installation

1. **Clone or download the project files**
   ```bash
   # If using git
   git clone https://github.com/shreyas114/context-aware-text-paraphraser.git
   cd NLP_Project
   
   # Or simply ensure you have the project files in your directory
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the web interface**
   - Open your browser and go to `http://localhost:7860`
   - The application will automatically download the Humarin paraphrasing model on first run (~1GB)

### 🎓 College Project Setup

For academic presentations and demonstrations:

1. **Prepare your environment**
   ```bash
   # Ensure you have Python 3.8+ installed
   python --version
   
   # Create a dedicated project folder
   mkdir AI_Paraphraser_Project
   cd AI_Paraphraser_Project
   ```

2. **Quick demo setup**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Run with verbose output for presentation
   python app.py
   ```

3. **For presentations, use these example texts:**
   - **Academic**: "The research methodology employed in this study involves a comprehensive analysis of quantitative data collected through structured surveys administered to a representative sample of participants."
   - **Business**: "I am writing to inform you that the meeting scheduled for tomorrow has been postponed due to unforeseen circumstances."
   - **Technical**: "The algorithm processes input data through multiple layers of neural networks, applying activation functions and backpropagation to optimize the model parameters for improved accuracy."

## 📖 Usage Guide

### Basic Usage
1. **Enter Text**: Type or paste your text in the input area
2. **Select Style**: Choose from the dropdown menu:
   - **Formal**: Professional, business-appropriate language
   - **Casual**: Conversational, everyday language
   - **Academic**: Scholarly, research-oriented language
   - **Simple**: Easy-to-understand, accessible language
3. **Adjust Creativity**: Use the slider to control output variation:
   - **Low (0.3-0.5)**: Conservative, stays close to original
   - **Medium (0.6-0.8)**: Balanced creativity and accuracy
   - **High (0.9-1.5)**: More creative and varied output
4. **Generate**: Click "Generate Paraphrase" to process your text
5. **Copy**: Use the copy button to save the result to clipboard

### Example Use Cases

#### Business Communication
- **Input**: "I am writing to inform you that the meeting scheduled for tomorrow has been postponed due to unforeseen circumstances."
- **Formal Style**: Professional email language
- **Casual Style**: Friendly, approachable tone

#### Academic Writing
- **Input**: Complex research descriptions
- **Academic Style**: Maintains scholarly tone
- **Simple Style**: Makes content accessible to broader audiences

#### Content Creation
- **Input**: Marketing copy or blog content
- **Various Styles**: Adapt tone for different audiences

## ⚙️ Technical Details

### Model Information
- **Base Model**: Humarin/chatgpt_paraphraser_on_T5_base
- **Architecture**: Fine-tuned T5-base (Text-to-Text Transfer Transformer)
- **Parameters**: ~220M parameters
- **Specialization**: Specifically trained for high-quality text paraphrasing
- **Training Data**: Fine-tuned on paraphrasing datasets for optimal performance
- **Capabilities**: Advanced paraphrasing with natural language understanding

### Performance Optimization
- **GPU Acceleration**: Automatic CUDA detection and usage
- **Memory Management**: Efficient model loading and inference
- **Batch Processing**: Optimized for single and multiple requests
- **Caching**: Model weights cached after first load

### System Requirements
- **Minimum**: 4GB RAM, Python 3.8+
- **Recommended**: 8GB+ RAM, GPU with 4GB+ VRAM
- **Storage**: ~1GB for model weights and dependencies

## 🔧 Configuration

### Environment Variables
You can customize the application behavior using environment variables:

```bash
# Model configuration
export MODEL_NAME="t5-base"  # Default model
export MAX_LENGTH=512        # Maximum input/output length
export DEVICE="auto"         # Device selection (auto/cpu/cuda)

# Server configuration
export HOST="0.0.0.0"       # Server host
export PORT=7860            # Server port
export SHARE=false          # Enable public sharing
```

### Advanced Settings
Modify the following parameters in `app.py` for advanced customization:

```python
# Model parameters
temperature_range = (0.3, 1.5)  # Creativity range
max_input_length = 512           # Maximum input tokens
num_return_sequences = 1         # Number of outputs

# Generation parameters
top_k = 50                       # Top-k sampling
top_p = 0.95                     # Nucleus sampling
```

## 🐛 Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/transformers/

# Reinstall transformers
pip uninstall transformers
pip install transformers>=4.21.0
```

#### Memory Issues
- Reduce `max_length` parameter
- Use CPU instead of GPU for large texts
- Close other applications to free RAM

#### Slow Performance
- Ensure GPU drivers are installed
- Check CUDA compatibility
- Consider using a smaller model for faster inference

#### Connection Issues
- Check firewall settings
- Verify port 7860 is available
- Try different port: `python app.py --server-port 8080`

### Error Messages
- **"CUDA out of memory"**: Reduce batch size or use CPU
- **"Model not found"**: Check internet connection for model download
- **"Port already in use"**: Change port or stop other applications

## 🚀 Deployment Options

### Hugging Face Spaces Deployment

Deploy your paraphraser as a public web app on Hugging Face Spaces:

1. **Create a Hugging Face account** at [huggingface.co](https://huggingface.co)

2. **Create a new Space**
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space)
   - Choose "Gradio" as the SDK
   - Set visibility to "Public" for sharing

3. **Upload your files**
   ```bash
   # Clone your space
   git clone https://huggingface.co/spaces/yourusername/your-space-name
   cd your-space-name
   
   # Copy your files
   cp app.py .
   cp requirements.txt .
   cp README.md .
   ```

4. **Configure for Spaces**
   - Update `app.py` to use `app.launch(share=True)` for public sharing
   - Ensure `requirements.txt` includes all dependencies
   - Add a `README.md` with project description

5. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy AI Text Paraphraser"
   git push
   ```

### Local Network Sharing

For classroom demonstrations:

```bash
# Run with network access
python app.py --server-name 0.0.0.0 --server-port 7860

# Share the local IP address with classmates
# Example: http://192.168.1.100:7860
```

## 🎓 Academic Project Features

### What This Tool Demonstrates
- **Natural Language Processing**: Real-world application of transformer models
- **AI Integration**: Seamless integration of pre-trained models into web applications
- **User Experience Design**: Intuitive interface for complex AI operations
- **Model Fine-tuning**: Understanding of specialized vs. general-purpose models

### Technical Learning Outcomes
- **Transformer Architecture**: Understanding of T5 model capabilities
- **Model Specialization**: Benefits of fine-tuned models for specific tasks
- **Web Development**: Building AI-powered web applications
- **API Integration**: Working with HuggingFace model hub

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Bugs**: Open an issue with detailed description
2. **Suggest Features**: Propose new functionality or improvements
3. **Submit Pull Requests**: Fix bugs or add features
4. **Improve Documentation**: Help make the README clearer

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest

# Run tests
pytest tests/

# Format code
black app.py

# Check code quality
flake8 app.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **HuggingFace**: For the amazing Transformers library and model hub
- **Humarin**: For the specialized paraphrasing model
- **Gradio**: For the excellent web interface framework
- **Google Research**: For developing the T5 architecture
- **PyTorch**: For the deep learning framework

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing issues on GitHub
3. Create a new issue with detailed information
4. Join our community discussions

## 🔮 Future Enhancements

- [ ] Support for multiple languages
- [ ] Batch processing for multiple texts
- [ ] API endpoint for programmatic access
- [ ] Custom model fine-tuning options
- [ ] Text similarity scoring
- [ ] Export functionality (PDF, DOCX)
- [ ] User authentication and history
- [ ] Advanced style customization

---

**Made with ❤️ using Python, HuggingFace Transformers, and Gradio**

*Perfect for academic projects, research demonstrations, and learning about AI-powered text processing!*
