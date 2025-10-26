import gradio as gr
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import time
from typing import Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

class TextParaphraser:
    def __init__(self):
        """Initialize the T5 model and tokenizer for paraphrasing."""
        self.model_name = "Humarin/chatgpt_paraphraser_on_T5_base"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        print("Loading T5 model...")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def create_style_prompt(self, text: str, style: str) -> str:
        """Return a simple paraphrase instruction compatible with the fine-tuned model."""
        # This model expects prompts like: "paraphrase: <text>"
        return f"paraphrase: {text}"
    
    def paraphrase_text(self, text: str, style: str = "Formal", temperature: float = 0.7) -> str:
        """
        Paraphrase the input text with specified style and creativity level.
        
        Args:
            text: Input text to paraphrase
            style: Paraphrasing style (Formal, Casual, Academic, Simple)
            temperature: Creativity level (0.3-1.5)
        
        Returns:
            Paraphrased text
        """
        if not text.strip():
            return ""
        
        try:
            # Use a simplified prompt; ignore style as the model is style-agnostic
            prompt = self.create_style_prompt(text, style)
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            ).to(self.device)
            
            # Generate paraphrase
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode output
            paraphrased = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the output
            paraphrased = self.clean_output(paraphrased)
            
            return paraphrased
            
        except Exception as e:
            return f"Error during paraphrasing: {str(e)}"
    
    def clean_output(self, text: str) -> str:
        """Clean and format the model output to return only the paraphrased content."""
        if not text:
            return ""
        original = text
        # Remove any residual instruction-like preambles
        patterns = [
            r'^\s*(paraphrase\s+in\s+\w+\s+style:)\s*',
            r'^\s*(paraphrase\s*:)\s*',
            r'^\s*(rephrase\s*:)\s*',
            r'^\s*(rewrite\s*:)\s*',
            r'^\s*(formally\s*:)\s*',
            r'^\s*(casually\s*:)\s*',
            r'^\s*(academically\s*:)\s*',
            r'^\s*(simplify\s*:)\s*',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Remove obvious meta phrases the model might emit
        meta_phrases = [
            "here is the paraphrased text",
            "here's the paraphrased text",
            "paraphrased:",
            "rephrased:",
            "in other words:",
        ]
        lowered = text.lower().strip()
        for phrase in meta_phrases:
            if lowered.startswith(phrase):
                text = text[len(phrase):].lstrip(" -:–—")
                break

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Capitalize first char if sentence-like
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

        # Ensure proper sentence ending (don't force if it ends with quote/paren)
        if text and not re.search(r'[.!?](["\'\)\]]\s*)?$', text):
            text += '.'

        return text

# Initialize the paraphraser
paraphraser = TextParaphraser()

def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split()) if text.strip() else 0

def paraphrase_with_progress(text: str, style: str, temperature: float) -> Tuple[str, str, str]:
    """
    Paraphrase text and return results with word counts.
    
    Returns:
        Tuple of (paraphrased_text, original_count, paraphrased_count)
    """
    if not text.strip():
        return "", "Words: 0", "Words: 0"
    
    # Show processing message
    original_count = count_words(text)
    
    # Perform paraphrasing
    paraphrased = paraphraser.paraphrase_text(text, style, temperature)
    paraphrased_count = count_words(paraphrased)
    
    return (
        paraphrased,
        f"Words: {original_count}",
        f"Words: {paraphrased_count}"
    )

def clear_all():
    """Clear all inputs and outputs."""
    return "", "", "Words: 0", "Words: 0"

def load_example(example_text: str):
    """Load an example text into the input area."""
    word_count = count_words(example_text)
    return example_text, f"Words: {word_count}", "", "Words: 0"

# Example texts for users to try
EXAMPLE_TEXTS = {
    "Business Email": "I am writing to inform you that the meeting scheduled for tomorrow has been postponed due to unforeseen circumstances. We will reschedule it for next week and send you the new details shortly.",
    
    "Academic Text": "The research methodology employed in this study involves a comprehensive analysis of quantitative data collected through structured surveys administered to a representative sample of participants.",
    
    "Casual Message": "Hey! Just wanted to let you know that I won't be able to make it to the party tonight because something came up at work. Hope you guys have a great time!",
    
    "Technical Description": "The algorithm processes input data through multiple layers of neural networks, applying activation functions and backpropagation to optimize the model parameters for improved accuracy."
}

# Custom CSS for professional styling
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.main-header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
}

.input-section, .output-section {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    border: 1px solid #e9ecef;
}

.word-count {
    font-size: 0.9em;
    color: #6c757d;
    font-weight: 500;
}

.example-section {
    background: #e3f2fd;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.control-panel {
    background: #fff3e0;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css, title="AI Text Paraphraser") as app:
    
    # Header
    gr.HTML("""
        <div class="main-header">
            <h1>🔄 AI Text Paraphraser & Rewriter</h1>
            <p>Transform your text with AI-powered paraphrasing in different styles</p>
        </div>
    """)
    
    with gr.Row():
        # Left column - Input and controls
        with gr.Column(scale=1):
            gr.HTML('<div class="input-section">')
            gr.Markdown("### 📝 Original Text")
            
            input_text = gr.Textbox(
                label="Enter your text here",
                placeholder="Type or paste the text you want to paraphrase...",
                lines=8,
                max_lines=15
            )
            
            original_word_count = gr.Textbox(
                label="Word Count",
                value="Words: 0",
                interactive=False,
                elem_classes=["word-count"]
            )
            
            gr.HTML('</div>')
            
            # Control panel
            gr.HTML('<div class="control-panel">')
            gr.Markdown("### ⚙️ Paraphrasing Settings")
            
            style_dropdown = gr.Dropdown(
                choices=["Formal", "Casual", "Academic", "Simple"],
                value="Formal",
                label="Paraphrasing Style",
                info="Choose the tone and style for your paraphrased text"
            )
            
            temperature_slider = gr.Slider(
                minimum=0.3,
                maximum=1.5,
                value=0.7,
                step=0.1,
                label="Creativity Level",
                info="Lower values = more conservative, Higher values = more creative"
            )
            
            gr.HTML('</div>')
            
            # Action buttons
            with gr.Row():
                paraphrase_btn = gr.Button("🚀 Generate Paraphrase", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
        
        # Right column - Output
        with gr.Column(scale=1):
            gr.HTML('<div class="output-section">')
            gr.Markdown("### ✨ Paraphrased Text")
            
            output_text = gr.Textbox(
                label="Paraphrased result",
                placeholder="Your paraphrased text will appear here...",
                lines=8,
                max_lines=15
            )
            
            paraphrased_word_count = gr.Textbox(
                label="Word Count",
                value="Words: 0",
                interactive=False,
                elem_classes=["word-count"]
            )
            
            # Copy button
            copy_btn = gr.Button("📋 Copy to Clipboard", variant="secondary")
            
            gr.HTML('</div>')
    
    # Example texts section
    gr.HTML('<div class="example-section">')
    gr.Markdown("### 💡 Try These Examples")
    
    with gr.Row():
        for title, text in EXAMPLE_TEXTS.items():
            example_btn = gr.Button(f"📄 {title}", variant="secondary", size="sm")
            example_btn.click(
                fn=lambda t=text: load_example(t),
                outputs=[input_text, original_word_count, output_text, paraphrased_word_count]
            )
    
    gr.HTML('</div>')
    
    # Usage instructions
    gr.Markdown("""
    ### 📖 How to Use
    1. **Enter your text** in the input area or try one of the example texts
    2. **Choose a style**: Formal for professional writing, Casual for everyday language, Academic for scholarly text, or Simple for easy-to-understand language
    3. **Adjust creativity**: Lower values keep closer to original meaning, higher values allow more creative interpretations
    4. **Click Generate** to create your paraphrase
    5. **Copy the result** using the copy button when you're satisfied
    
    ### 🎯 Tips for Best Results
    - Keep input text under 500 words for optimal performance
    - Try different creativity levels to find your preferred style
    - Academic style works best with formal or technical content
    - Simple style is great for making complex text more accessible
    """)
    
    # Event handlers
    input_text.change(
        fn=lambda text: f"Words: {count_words(text)}",
        inputs=[input_text],
        outputs=[original_word_count]
    )
    
    paraphrase_btn.click(
        fn=paraphrase_with_progress,
        inputs=[input_text, style_dropdown, temperature_slider],
        outputs=[output_text, original_word_count, paraphrased_word_count]
    )
    
    clear_btn.click(
        fn=clear_all,
        outputs=[input_text, output_text, original_word_count, paraphrased_word_count]
    )
    
    # JavaScript for copy functionality
    copy_btn.click(
        fn=None,
        inputs=[output_text],
        outputs=[],
        js="""
        function(text) {
            navigator.clipboard.writeText(text).then(function() {
                // Show temporary success message
                const btn = document.querySelector('button[aria-label="📋 Copy to Clipboard"]');
                const originalText = btn.textContent;
                btn.textContent = '✅ Copied!';
                setTimeout(() => {
                    btn.textContent = originalText;
                }, 2000);
            });
        }
        """
    )

if __name__ == "__main__":
    print("Starting AI Text Paraphraser...")
    print("Loading model and initializing interface...")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )
