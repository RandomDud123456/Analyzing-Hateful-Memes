# from PIL import Image, ImageDraw
# from ultralytics import YOLO

# # Initialize YOLO model
# model = YOLO("yolov5s.pt")

# og_images=["/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Ancient_Aliens.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Black_guy.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Cartoon.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Charlie.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Dave_Chapelle.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Disaster_girl.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Distracted_Boyfriend.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/OneDoesNotSimply.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Pablo.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/WWE.jpg"]
# meme_images=["/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Ancient_Aliens_meme.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Black_Guy_meme.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Cartoon_meme.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Charlie_meme.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Dave_Chapelle_meme.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Disaster_girl_meme.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Distracted_Boyfriend_meme.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/OneDoesNotSimply_meme.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Pablo_meme.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/WWE_meme.jpg"]

# # Directory containing images for frequency/probability calculation
# img1 = "/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Ancient_Aliens.jpg"
# img2 = "/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Ancient_Aliens_meme.jpg"

# # Process first image
# results = model.predict(img1)
# result = results[0]

# # Get bounding box coordinates, classes, and confidences
# boxes = results[0].boxes.xyxy.tolist()
# classes = results[0].boxes.cls.tolist()
# names = results[0].names
# confidences = results[0].boxes.conf.tolist()

# # Open the image
# image = Image.open(img1)
# draw = ImageDraw.Draw(image)

# # Draw bounding boxes on the image and print detected objects
# for box, cls, conf in zip(boxes, classes, confidences):
#     x1, y1, x2, y2 = box
#     name = names[int(cls)]
#     draw.rectangle([x1, y1, x2, y2], outline="black", width=2)
#     draw.text((x1, y1), f"{name} {conf:.2f}", fill="black")
#     print(f"Detected object: {name}")

# # Save the modified image
# image.save("detected_img1.jpg")

# # Process second image
# results = model.predict(img2)
# result = results[0]

# # Get bounding box coordinates, classes, and confidences
# boxes = results[0].boxes.xyxy.tolist()
# classes = results[0].boxes.cls.tolist()
# names = results[0].names
# confidences = results[0].boxes.conf.tolist()

# # Open the image
# image = Image.open(img2)
# draw = ImageDraw.Draw(image)

# # Draw bounding boxes on the image and print detected objects
# for box, cls, conf in zip(boxes, classes, confidences):
#     x1, y1, x2, y2 = box
#     name = names[int(cls)]
#     draw.rectangle([x1, y1, x2, y2], outline="black", width=2)
#     draw.text((x1, y1), f"{name} {conf:.2f}", fill="black")
#     print(f"Detected object: {name}")

# # Save the modified image
# image.save("detected_img2.jpg")


import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from ultralytics import YOLO
import numpy as np

ratios = []
model = YOLO("yolov5s.pt")

og_images=["/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Ancient_Aliens.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Black_guy.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Cartoon.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Charlie.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Dave_Chapelle.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Disaster_girl.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Distracted_Boyfriend.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/OneDoesNotSimply.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Pablo.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/WWE.jpg"]
meme_images=["/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Ancient_Aliens_meme.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Black_Guy_meme.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Cartoon_meme.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Charlie_meme.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Dave_Chapelle_meme.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Disaster_girl_meme.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Distracted_Boyfriend_meme.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/OneDoesNotSimply_meme.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/Pablo_meme.jpg","/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/caption_image_analysis/WWE_meme.jpg"]

for i in range(10):

    img1=og_images[i]
    img2=meme_images[i]

    # Process first image
    results = model.predict(img1)
    result = results[0]

    # Get bounding box coordinates, classes, and confidences
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    img1_names=[]

    # Draw bounding boxes on the image and print detected objects
    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = box
        name = names[int(cls)]
        img1_names.append(name)

    results = model.predict(img2)
    result = results[0]

    # Get bounding box coordinates, classes, and confidences
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    img2_names=[]

    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = box
        name = names[int(cls)]
        img2_names.append(name)

    x=len(img1_names)

    count=0

    for val in img1_names:
        if val in img2_names:
            img2_names.remove(val)
            count=count+1

    ratio=1-(count/x)

    ratios.append(ratio)

# Plotting the graph
plt.plot(ratios, marker='o')
plt.xlabel('Image Index')
plt.ylabel('Ratio of Objects Unique to Meme Image')
plt.title('Comparison of Objects Detected in Meme and Original Images')
plt.xticks(range(len(og_images)), range(1, len(og_images)+1))
plt.grid(True)
plt.show()

print(np.mean(ratios))