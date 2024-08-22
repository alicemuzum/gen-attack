import torch
from diffusers import DiffusionPipeline


class SDIFF():
    def __init__(self) -> None:
        self.model = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                       torch_dtype=torch.float16,
                                                       use_safetensors=True,).to("cuda")
        self.model.enable_attention_slicing()

    def __call__(self, prompt):
        
        image = self.model(prompt).images[0]
        image.save("examp.png")

a = SDIFF()
a("a corner of a realistic bench")