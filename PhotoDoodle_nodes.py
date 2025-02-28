import torch
from PIL import Image
import numpy as np
from PhotoDoodle.pipeline_pe_clone import FluxPipeline

class PhotoDoodle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "black-forest-labs/FLUX.1-dev"}),
                "image_path": ("STRING", {"default": "assets/1.png"}),
                "lora_name": ("STRING", {"default": "sksmagiceffects"}),
                "prompt": ("STRING", {"default": "add a halo and wings for the cat by sksmagiceffects"}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                "num_steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "height": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 64}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "PhotoDoodle"
    
    def __init__(self):
        self.pipeline = None
    
    def load_pipeline(self, model_path):
        if self.pipeline is None:
            self.pipeline = FluxPipeline.from_pretrained(
                model_path, torch_dtype=torch.bfloat16
            ).to('cuda')
            self.pipeline.load_lora_weights("nicolaus-huang/PhotoDoodle", weight_name="pretrain.safetensors")
            self.pipeline.fuse_lora()
            self.pipeline.unload_lora_weights()
    
    def generate_image(self, model_path, image_path, lora_name, prompt, guidance_scale, num_steps, height, width):
        self.load_pipeline(model_path)
        
        if lora_name != "pretrained":
            self.pipeline.load_lora_weights("nicolaus-huang/PhotoDoodle", weight_name=f"{lora_name}.safetensors")
        
        condition_image = Image.open(image_path).resize((width, height)).convert("RGB")
        result = self.pipeline(
            prompt=prompt,
            condition_image=condition_image,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            max_sequence_length=512
        ).images[0]
        
        output_image = np.array(result, dtype=np.uint8)
        return (output_image,)
