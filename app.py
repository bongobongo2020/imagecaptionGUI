import os
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import warnings
warnings.filterwarnings("ignore")

class ImageProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LoRA Image Processor")
        self.root.geometry("800x600")
        
        # Initialize model variables
        self.model = None
        self.processor = None
        
        self.setup_gui()
        
    def setup_gui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input Directory Selection
        ttk.Label(main_frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.input_dir_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.input_dir_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_input).grid(row=0, column=2)
        
        # Output Directory Selection
        ttk.Label(main_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.output_dir_var, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output).grid(row=1, column=2)
        
        # Image Size Settings
        size_frame = ttk.LabelFrame(main_frame, text="Image Size Settings", padding="5")
        size_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(size_frame, text="Width:").grid(row=0, column=0, padx=5)
        self.width_var = tk.StringVar(value="512")
        ttk.Entry(size_frame, textvariable=self.width_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(size_frame, text="Height:").grid(row=0, column=2, padx=5)
        self.height_var = tk.StringVar(value="512")
        ttk.Entry(size_frame, textvariable=self.height_var, width=10).grid(row=0, column=3, padx=5)
        
        # Progress Frame
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="5")
        progress_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Log Area
        self.log_area = scrolledtext.ScrolledText(main_frame, height=15, width=70)
        self.log_area.grid(row=4, column=0, columnspan=3, pady=10)
        
        # Control Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start Processing", command=self.start_processing)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_button.grid(row=0, column=1, padx=5)
        
        # Processing flag
        self.processing = False
        
    def browse_input(self):
        directory = filedialog.askdirectory()
        if directory:
            self.input_dir_var.set(directory)
            
    def browse_output(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir_var.set(directory)
            
    def log_message(self, message):
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        
    def setup_captioning_model(self):
        """Initialize the image captioning model."""
        self.log_message("Loading captioning model...")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.log_message("Model loaded successfully!")
        
    def generate_caption(self, image):
        """Generate a caption for the given image using BLIP model."""
        inputs = self.processor(images=image, return_tensors="pt")
        output = self.model.generate(**inputs, max_new_tokens=50)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption
        
    def process_image(self, image_path, output_size):
        """Process a single image: resize it and maintain aspect ratio."""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                width, height = img.size
                aspect_ratio = width / height
                
                if aspect_ratio > 1:
                    new_width = output_size[0]
                    new_height = int(output_size[1] / aspect_ratio)
                else:
                    new_height = output_size[1]
                    new_width = int(output_size[0] * aspect_ratio)
                
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                new_img = Image.new('RGB', output_size, (0, 0, 0))
                paste_x = (output_size[0] - new_width) // 2
                paste_y = (output_size[1] - new_height) // 2
                new_img.paste(resized_img, (paste_x, paste_y))
                
                return new_img, "Successfully processed"
                
        except Exception as e:
            return None, f"Error processing image: {str(e)}"
            
    def process_directory(self):
        """Main processing function."""
        try:
            input_dir = self.input_dir_var.get()
            output_dir = self.output_dir_var.get()
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            
            if not all([input_dir, output_dir, width, height]):
                self.log_message("Please fill in all fields!")
                return
                
            os.makedirs(output_dir, exist_ok=True)
            
            # Load model if not already loaded
            if self.model is None:
                self.setup_captioning_model()
                
            # Get list of image files
            image_files = [f for f in os.listdir(input_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            total_files = len(image_files)
            
            if total_files == 0:
                self.log_message("No image files found in input directory!")
                return
                
            self.log_message(f"Found {total_files} images to process")
            
            # Process each image
            for i, filename in enumerate(image_files):
                if not self.processing:
                    break
                    
                base_name = os.path.splitext(filename)[0]
                input_path = os.path.join(input_dir, filename)
                output_image_path = os.path.join(output_dir, filename)
                output_text_path = os.path.join(output_dir, f"{base_name}.txt")
                
                self.log_message(f"Processing {filename}...")
                
                processed_img, status = self.process_image(input_path, (width, height))
                
                if processed_img:
                    processed_img.save(output_image_path, quality=95)
                    caption = self.generate_caption(processed_img)
                    
                    with open(output_text_path, 'w', encoding='utf-8') as f:
                        f.write(caption)
                    
                    self.log_message(f"Generated caption: {caption}")
                else:
                    self.log_message(f"Failed: {status}")
                
                # Update progress
                progress = ((i + 1) / total_files) * 100
                self.progress_var.set(progress)
                
            self.log_message("Processing completed!")
            
        except Exception as e:
            self.log_message(f"Error during processing: {str(e)}")
        
        finally:
            self.processing = False
            self.start_button.config(state=tk.NORMAL)
            self.cancel_button.config(state=tk.DISABLED)
            self.progress_var.set(0)
            
    def start_processing(self):
        """Start the processing thread."""
        self.processing = True
        self.start_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.log_area.delete(1.0, tk.END)
        
        # Start processing in a separate thread
        threading.Thread(target=self.process_directory, daemon=True).start()
        
    def cancel_processing(self):
        """Cancel the processing."""
        self.processing = False
        self.log_message("Cancelling processing...")
        
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorGUI(root)
    root.mainloop()