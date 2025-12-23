import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from torchvision import transforms
from models import MNISTCNN
from config import config
import os
from scipy import ndimage

class MNISTPredictor:
    def __init__(self, model_path):
        self.device = config.DEVICE
        self.model = MNISTCNN().to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded: {checkpoint['accuracy']:.2f}% accuracy")
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def preprocess_image(self, image_input):
        """Enhanced preprocessing to better match MNIST format"""
        if image_input is None:
            return None
        
        # Extract image from Sketchpad dictionary
        if isinstance(image_input, dict):
            if 'composite' in image_input:
                image_array = image_input['composite']
            elif 'background' in image_input:
                image_array = image_input['background']
            else:
                return None
            
            if isinstance(image_array, np.ndarray):
                if len(image_array.shape) == 3:
                    image = Image.fromarray(image_array.astype('uint8')).convert('L')
                else:
                    image = Image.fromarray(image_array.astype('uint8'), mode='L')
            else:
                return None
        else:
            return None
        
        # Convert to numpy for processing
        img_array = np.array(image)
        
        # Invert if background is white
        if img_array.mean() > 127:
            img_array = 255 - img_array
        
        # Find bounding box of the digit
        rows = np.any(img_array > 30, axis=1)
        cols = np.any(img_array > 30, axis=0)
        
        if not rows.any() or not cols.any():
            return None
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Add padding
        padding = 20
        rmin = max(0, rmin - padding)
        rmax = min(img_array.shape[0], rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(img_array.shape[1], cmax + padding)
        
        # Crop to bounding box
        img_array = img_array[rmin:rmax, cmin:cmax]
        
        # Resize to 20x20 (MNIST uses 20x20 for the digit, then pads to 28x28)
        image = Image.fromarray(img_array)
        
        # Calculate aspect ratio to maintain proportions
        width, height = image.size
        if width > height:
            new_width = 20
            new_height = int(20 * height / width)
        else:
            new_height = 20
            new_width = int(20 * width / height)
        
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create 28x28 black canvas and paste centered
        final_img = Image.new('L', (28, 28), 0)
        paste_x = (28 - new_width) // 2
        paste_y = (28 - new_height) // 2
        final_img.paste(image, (paste_x, paste_y))
        
        # Apply transforms
        image_tensor = self.transform(final_img).unsqueeze(0)
        return image_tensor, final_img
    
    def predict(self, image_input):
        result = self.preprocess_image(image_input)
        
        if result is None:
            return {str(i): 0.0 for i in range(10)}, None
        
        image_tensor, processed_img = result
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
        
        probs_dict = {str(i): float(probabilities[i]) for i in range(10)}
        return probs_dict, processed_img


def create_interface():
    predictor = MNISTPredictor(config.MODEL_SAVE_PATH)
    
    with gr.Blocks(title="MNIST Digit Classifier") as demo:
        gr.Markdown(
            """
            # 🔢 MNIST Digit Classifier
            Draw a digit (0-9) and see the AI predict it! **99.3% accuracy** on test set.
            """
        )
        
        with gr.Row():
            with gr.Column():
                canvas = gr.Sketchpad(
                    type="numpy",
                    label="✏️ Draw Here (Draw BIG!)",
                    canvas_size=(280, 280)
                )
                
                gr.Markdown(
                    """
                    ### 💡 Tips for Better Accuracy:
                    - **Draw LARGE** (fill most of the canvas)
                    - **Center your digit**
                    - **Use thick strokes**
                    - **Draw clearly** (like writing on paper)
                    - For **6**: close the loop at top
                    - For **9**: close the loop at bottom
                    - For **3**: make distinct curves
                    """
                )
                
                clear_btn = gr.Button("🗑️ Clear Canvas", size="sm")
            
            with gr.Column():
                label = gr.Label(num_top_classes=10, label="📊 Predictions")
                processed = gr.Image(label="🔍 What Model Sees (28×28)", height=140)
                
                gr.Markdown(
                    """
                    **How it works:**
                    1. Your drawing is cropped to the digit
                    2. Resized to 20×20 pixels
                    3. Centered in 28×28 canvas (like MNIST)
                    4. Fed to the CNN
                    """
                )
        
        # Predict on change
        canvas.change(
            fn=predictor.predict,
            inputs=canvas,
            outputs=[label, processed]
        )
        
        # Clear button
        clear_btn.click(
            fn=lambda: (None, None, None),
            outputs=[canvas, label, processed]
        )
    
    return demo


def main():
    print("="*60)
    print("Starting MNIST Digit Classifier")
    print("="*60)
    
    demo = create_interface()
    demo.launch(share=False)


if __name__ == "__main__":
    main()