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
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def preprocess_image(self, image_dict):
        """
        Preprocess the drawn image for model input
        
        Args:
            image_dict: Dictionary from Gradio Sketchpad
            
        Returns:
            Preprocessed tensor
        """
        # Handle None or empty input
        if image_dict is None:
            return None
        
        # Extract image from dictionary
        if isinstance(image_dict, dict):
            # Try different keys that Gradio might use
            if 'composite' in image_dict and image_dict['composite'] is not None:
                image_array = image_dict['composite']
            elif 'background' in image_dict and image_dict['background'] is not None:
                image_array = image_dict['background']
            elif 'layers' in image_dict and len(image_dict['layers']) > 0:
                image_array = image_dict['layers'][0]
            else:
                return None
        else:
            image_array = image_dict
        
        # Check if we got valid data
        if image_array is None or (isinstance(image_array, np.ndarray) and image_array.size == 0):
            return None
        
        # Convert numpy array to PIL Image
        if isinstance(image_array, np.ndarray):
            # If it's already grayscale (2D)
            if len(image_array.shape) == 2:
                image = Image.fromarray(image_array.astype(np.uint8), mode='L')
            # If it's RGB (3D)
            elif len(image_array.shape) == 3:
                # Convert to grayscale by averaging channels
                gray = np.mean(image_array, axis=2).astype(np.uint8)
                image = Image.fromarray(gray, mode='L')
            else:
                return None
        else:
            image = image_array
        
        # Invert colors (black on white -> white on black)
        image_np = np.array(image)
        image_np = 255 - image_np
        image = Image.fromarray(image_np)
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def predict(self, image_dict):
        """
        Predict the digit from the drawn image
        
        Args:
            image_dict: Input from Gradio
            
        Returns:
            Dictionary with probabilities for each digit
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_dict)
        
        if image_tensor is None:
            return {str(i): 0.0 for i in range(10)}
        
        image_tensor = image_tensor.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
        
        # Convert to dictionary
        prob_dict = {str(i): float(probabilities[i]) for i in range(10)}
        
        return prob_dict


def create_interface():
    """Create Gradio interface"""
    
    predictor = MNISTPredictor(config.MODEL_SAVE_PATH)
    
    with gr.Blocks(title="MNIST Digit Classifier") as demo:
        
        gr.Markdown(
            """
            # 🔢 MNIST Digit Classifier
            ### Draw a digit (0-9) and the AI will predict it!
            
            This CNN achieves 99.3% accuracy on MNIST dataset.
            """
        )
        
        with gr.Row():
            with gr.Column():
                canvas = gr.Sketchpad(
                    label="Draw Here",
                    type="numpy",
                    image_mode="L",
                    canvas_size=(280, 280),
                    brush=gr.Brush(default_size=15, colors=["#000000"], color_mode="fixed")
                )
                
                with gr.Row():
                    clear_btn = gr.Button("Clear")
                    predict_btn = gr.Button("Predict", variant="primary")
            
            with gr.Column():
                label = gr.Label(label="Predictions", num_top_classes=10)
                confidence_text = gr.Textbox(label="Result", interactive=False)
        
        gr.Markdown(
            """
            ### 💡 Tips
            - Draw large and centered
            - Use thick strokes
            - Keep it simple
            """
        )
        
        def predict_and_show(image):
            if image is None:
                return {str(i): 0.0 for i in range(10)}, "Draw a digit first!"
            
            probs = predictor.predict(image)
            top_digit = max(probs, key=probs.get)
            top_conf = probs[top_digit] * 100
            
            return probs, f"Predicted: {top_digit} ({top_conf:.1f}% confidence)"
        
        predict_btn.click(
            fn=predict_and_show,
            inputs=[canvas],
            outputs=[label, confidence_text]
        )
        
        canvas.change(
            fn=predict_and_show,
            inputs=[canvas],
            outputs=[label, confidence_text]
        )
        
        clear_btn.click(
            fn=lambda: (None, {str(i): 0.0 for i in range(10)}, ""),
            outputs=[canvas, label, confidence_text]
        )
    
    return demo


def main():
    print("="*60)
    print("Starting MNIST Digit Classifier")
    print("="*60)
    
    demo = create_interface()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()