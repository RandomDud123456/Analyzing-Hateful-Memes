import os
import shutil
import json

# Define paths

json_file = "/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/hateful_memes/test_seen.jsonl"
image_source_dir = "/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/hateful_memes/"
image_dest_dir = "/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/dataset_hate/test/images/"
text_dest_dir = "/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/dataset_hate/test/texts/"

# Count images in each label directory
def count_images(label):
    return len([name for name in os.listdir(os.path.join(image_dest_dir, str(label))) if os.path.isfile(os.path.join(image_dest_dir, str(label), name))])

# Process JSON file
with open(json_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        img_path = os.path.join(image_source_dir, data['img'])
        label = data['label']
        text = data['text']

        # Check label count
        if label == 0:
            label_dir = "0"
        elif label == 1:
            label_dir = "1"
        else:
            continue

        # Check if the source image file exists
        if not os.path.exists(img_path):
            print(f"Error: Image file '{img_path}' does not exist.")
            continue

        # Move image
        dest_img_path = os.path.join(image_dest_dir, label_dir, os.path.basename(img_path))
        shutil.copyfile(img_path, dest_img_path)

        # Move text
        text_filename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        dest_text_path = os.path.join(text_dest_dir, label_dir, text_filename)
        with open(dest_text_path, 'w') as text_file:
            text_file.write(text)

print("Processing completed.")
