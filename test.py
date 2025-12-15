import gradio as gr
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models import MNISTCNN
from config import config
import os

class MNISTPredictor:
    """Wrapper class for MNIST prediction"""
    
    def __init__(self, model_path):
        """Initialize the model"""
        self.device = config.DEVICE
        self.model = MNISTCNN().to(self.device)
        
        # Load trained weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
            print(f"Model accuracy: {checkpoint['accuracy']:.2f}%")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Define the same transform used during training
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def preprocess_image(self, image_dict):
        """
        Preprocess the drawn image for model input
        
        Args:
            image_dict: Dictionary from Gradio Sketchpad containing 'background' and 'layers'
            
        Returns:
            Preprocessed tensor
        """
        # Extract the composite image from Gradio Sketchpad
        if image_dict is None:
            return None
        
        # Get the background or composite layer
        if 'composite' in image_dict and image_dict['composite'] is not None:
            image_array = image_dict['composite']
        elif 'background' in image_dict and image_dict['background'] is not None:
            image_array = image_dict['background']
        else:
            return None
        
        # Convert to PIL Image
        if isinstance(image_array, np.ndarray):
            # Convert to grayscale by averaging RGB channels
            if len(image_array.shape) == 3:
                image_array = np.mean(image_array, axis=2).astype(np.uint8)
            
            image = Image.fromarray(image_array)
        else:
            image = image_array
        
        # Invert colors (Gradio drawing is black on white, MNIST is white on black)
        image = image.convert('L')  # Ensure grayscale
        image_array = np.array(image)
        image_array = 255 - image_array  # Invert
        image = Image.fromarray(image_array)
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def predict(self, image_dict):
        """
        Predict the digit from the drawn image
        
        Args:
            image_dict: Input image dictionary from Gradio
            
        Returns:
            Dictionary with probabilities for each digit
        """
        if image_dict is None:
            return {str(i): 0.0 for i in range(10)}
        
        # Preprocess image
        image_tensor = self.preprocess_image(image_dict)
        
        if image_tensor is None:
            return {str(i): 0.0 for i in range(10)}
        
        image_tensor = image_tensor.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
        
        # Convert to dictionary for Gradio
        prob_dict = {str(i): float(probabilities[i]) for i in range(10)}
        
        return prob_dict


def create_interface():
    """Create and configure the Gradio interface"""
    
    # Initialize predictor
    predictor = MNISTPredictor(config.MODEL_SAVE_PATH)
    
    # Create Gradio interface
    with gr.Blocks(title="MNIST Digit Classifier") as demo:
        
        # Header
        gr.Markdown(
            """
            # 🔢 MNIST Digit Classifier
            ### Draw a digit (0-9) and watch the AI predict it in real-time!
            
            This neural network was trained on 60,000 handwritten digits and achieves 99.3% accuracy.
            The model uses Convolutional Neural Networks (CNN) optimized for Apple M1.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Drawing canvas
                canvas = gr.Sketchpad(
                    label="Draw a Digit Here",
                    type="numpy",
                    image_mode="L",
                    canvas_size=(280, 280),
                    brush=gr.Brush(
                        default_size=15,
                        colors=["#000000"],
                        color_mode="fixed"
                    )
                )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    predict_btn = gr.Button("🚀 Predict", variant="primary")
            
            with gr.Column(scale=1):
                # Prediction output
                label = gr.Label(
                    label="Prediction Probabilities",
                    num_top_classes=10
                )
                
                # Prediction confidence
                gr.Markdown("### 📊 Model Confidence")
                confidence_text = gr.Textbox(
                    label="Top Prediction",
                    interactive=False
                )
        
        # Examples section
        gr.Markdown("### 💡 Tips for Best Results")
        gr.Markdown(
            """
            - Draw the digit **large and centered**
            - Use **thick strokes** (like a marker)
            - Keep digits **clear and simple**
            - Try different writing styles!
            """
        )
        
        # Model info section
        with gr.Accordion("ℹ️ About the Model", open=False):
            gr.Markdown(
                """
                **Architecture:** Convolutional Neural Network (CNN)
                - 2 Convolutional Layers (32 and 64 filters)
                - 2 MaxPooling Layers
                - 2 Fully Connected Layers
                - Dropout for regularization
                
                **Training Details:**
                - Dataset: MNIST (60,000 training images)
                - Optimizer: Adam
                - Training Time: ~3-5 minutes on M1 Mac
                - Final Accuracy: 99.33%
                
                **Technology Stack:**
                - PyTorch 2.9.1
                - Apple M1 MPS Acceleration
                - Gradio for Web Interface
                """
            )
        
        # Define prediction callback
        def predict_and_show_confidence(image):
            """Wrapper to show prediction and confidence"""
            if image is None:
                return {str(i): 0.0 for i in range(10)}, "Draw a digit first!"
            
            probs = predictor.predict(image)
            
            # Get top prediction
            top_digit = max(probs, key=probs.get)
            top_confidence = probs[top_digit] * 100
            
            confidence_msg = f"Predicted Digit: **{top_digit}** with **{top_confidence:.1f}%** confidence"
            
            return probs, confidence_msg
        
        # Event handlers
        predict_btn.click(
            fn=predict_and_show_confidence,
            inputs=[canvas],
            outputs=[label, confidence_text]
        )
        
        # Auto-predict on draw (optional - real-time prediction)
        canvas.change(
            fn=predict_and_show_confidence,
            inputs=[canvas],
            outputs=[label, confidence_text]
        )
        
        clear_btn.click(
            fn=lambda: (None, {str(i): 0.0 for i in range(10)}, "Draw a digit to see prediction!"),
            inputs=None,
            outputs=[canvas, label, confidence_text]
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            Built with ❤️ using PyTorch and Gradio
            """
        )
    
    return demo


def main():
    """Launch the web interface"""
    
    print("="*60)
    print("Starting MNIST Digit Classifier Web Interface")
    print("="*60)
    
    # Create interface
    demo = create_interface()
    
    # Launch
    print("\nLaunching web interface...")
    print("The app will open in your browser automatically.")
    print("Press Ctrl+C to stop the server.")
    print("="*60)
    
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()