import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import cv2
import numpy as np
import matplotlib.pyplot as plt

pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

init_image_path = "/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/images/26187.png"
init_image = load_image(init_image_path)

init_image = np.array(init_image)

gray_image = cv2.cvtColor(init_image, cv2.COLOR_RGB2GRAY)

_, binary_mask = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)

inverted_mask = cv2.bitwise_not(binary_mask)

mask_image = cv2.cvtColor(inverted_mask, cv2.COLOR_GRAY2RGB)
mask_image = np.where(mask_image == [255, 255, 255], [0, 0, 0], [255, 255, 255])

# Display the mask image
plt.imshow(mask_image)
plt.title('Mask Image (Inverted)')
plt.axis('off')
plt.show()


prompt = "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k"
negative_prompt = "bad anatomy, deformed, ugly, disfigured"
image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)

