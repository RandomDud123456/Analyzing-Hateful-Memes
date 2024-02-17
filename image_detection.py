from ultralytics import YOLO
import os
import itertools
import numpy as np
from collections import Counter
import pandas as pd

# Define the dictionary containing the classes
classes_dict = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
    8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
    29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv',
    63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# Create an empty DataFrame
df = pd.DataFrame(columns=classes_dict.values())

# Add Output column
df['Output'] = None

# Initialize YOLO model
model = YOLO("yolov8m.pt")

# Directory containing images for frequency/probability calculation
frequency_image_dir_1 = "/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/dataset_hate/train/images/1"

frequency_image_dir_0 = "/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/dataset_hate/train/images/0"


# Directory containing images for total score calculation
total_score_image_dir_0 = "/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/dataset_hate/test/images/0"
total_score_image_dir_1 = "/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/dataset_hate/test/images/1"

# Initialize dictionary to store object frequencies
object_frequencies = {}
pair_frequencies = {}

# Iterate over images in the directory for frequency/probability calculation
# for image_file in os.listdir(frequency_image_dir_1):
#     if image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg"):
#         # Predict objects in the current image
#         results = model.predict(os.path.join(frequency_image_dir_1, image_file))
#         result = results[0]  # Assuming there's only one result
#         # Iterate over detected boxes in the current image
#         detected_objects = [result.names[box.cls[0].item()] for box in result.boxes]
#         print(detected_objects)
#         # Update object frequency dictionary
#         for obj in detected_objects:
#             if obj not in object_frequencies:
#                 object_frequencies[obj] = 1
#             else:
#                 object_frequencies[obj] += 1
#         # Update pair frequency dictionary
#         for pair in itertools.combinations(detected_objects, 2):
#             pair_key = tuple(sorted(pair))
#             if pair_key not in pair_frequencies:
#                 pair_frequencies[pair_key] = 1
#             else:
#                 pair_frequencies[pair_key] += 1

new_data = []

for image_file in os.listdir(total_score_image_dir_1):
    if image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg"):
        # Predict objects in the current image
        results = model.predict(os.path.join(total_score_image_dir_1, image_file))
        result = results[0]  # Assuming there's only one result
        # Iterate over detected boxes in the current image
        detected_objects = [result.names[box.cls[0].item()] for box in result.boxes]
        
        # Count the occurrences of each object
        object_counts = Counter(detected_objects)
        
        # Create dictionary to store the frequency of each object
        object_freq = {obj: object_counts.get(obj, 0) for obj in df.columns[:-1]}  # Last column is 'Output', so exclude it
        
        # Set output value
        object_freq['Output'] = 1
        
        # Append the dictionary to the list
        new_data.append(object_freq)

# Concatenate the original DataFrame with the new data
df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True) 


# for image_file in os.listdir(frequency_image_dir_0):
#     if image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg"):
#         # Predict objects in the current image
#         results = model.predict(os.path.join(frequency_image_dir_0, image_file))
#         result = results[0]  # Assuming there's only one result
#         # Iterate over detected boxes in the current image
#         detected_objects = [result.names[box.cls[0].item()] for box in result.boxes]
#         # Update object frequency dictionary
#         for obj in detected_objects:
#             if obj in object_frequencies:
#                 object_frequencies[obj] -= 1

#         # Update pair frequency dictionary
#         for pair in itertools.combinations(detected_objects, 2):
#             pair_key = tuple(sorted(pair))
#             if pair_key in pair_frequencies:
#                 pair_frequencies[pair_key] -= 1

new_data = []

for image_file in os.listdir(total_score_image_dir_0):
    if image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg"):
        # Predict objects in the current image
        results = model.predict(os.path.join(total_score_image_dir_0, image_file))
        result = results[0]  # Assuming there's only one result
        # Iterate over detected boxes in the current image
        detected_objects = [result.names[box.cls[0].item()] for box in result.boxes]
        
        # Count the occurrences of each object
        object_counts = Counter(detected_objects)
        
        # Create dictionary to store the frequency of each object
        object_freq = {obj: object_counts.get(obj, 0) for obj in df.columns[:-1]}  # Last column is 'Output', so exclude it
        
        # Set output value
        object_freq['Output'] = 0
        
        # Append the dictionary to the list
        new_data.append(object_freq)

# Concatenate the original DataFrame with the new data
df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True) 


# Calculate probabilities for objects and pairs
# total_pairs = sum(pair_frequencies.values())
# total_objects = sum(object_frequencies.values())

# object_probabilities = {obj: freq / total_objects for obj, freq in object_frequencies.items()}
# pair_probabilities = {pair: freq / total_pairs for pair, freq in pair_frequencies.items()}

# print("\n")

# list_0=[]

# for image_file in os.listdir(total_score_image_dir_0):
#     if image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg"):
#         # Predict objects in the current image
#         results = model.predict(os.path.join(total_score_image_dir_0, image_file))
#         result = results[0]  # Assuming there's only one result
#         # Iterate over detected boxes in the current image
#         detected_objects = [result.names[box.cls[0].item()] for box in result.boxes]
#         list1=[]
#         list2=[]
#         # Update object frequency dictionary
#         for obj in detected_objects:
#             if obj not in object_probabilities:
#                 list1.append(0)
#             else:
#                 list1.append(object_probabilities[obj])
#         # Update pair frequency dictionary
#         for pair in itertools.combinations(detected_objects, 2):
#             pair_key = tuple(sorted(pair))
#             if pair_key not in pair_frequencies:
#                 list2.append(0)
#             else:
#                 list2.append(pair_probabilities[pair_key])


#         first_avg=np.mean(list1)
#         second_avg=np.mean(list2)

#         final_ans=(2*second_avg+first_avg)/3

#         list_0.append(final_ans)

# list_1=[]

# for image_file in os.listdir(total_score_image_dir_1):
#     if image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg"):
#         # Predict objects in the current image
#         results = model.predict(os.path.join(total_score_image_dir_1, image_file))
#         result = results[0]  # Assuming there's only one result
#         # Iterate over detected boxes in the current image
#         detected_objects = [result.names[box.cls[0].item()] for box in result.boxes]
#         list1=[]
#         list2=[]
#         # Update object frequency dictionary
#         for obj in detected_objects:
#             if obj not in object_probabilities:
#                 list1.append(0)
#             else:
#                 list1.append(object_probabilities[obj])
#         # Update pair frequency dictionary
#         for pair in itertools.combinations(detected_objects, 2):
#             pair_key = tuple(sorted(pair))
#             if pair_key not in pair_frequencies:
#                 list2.append(0)
#             else:
#                 list2.append(pair_probabilities[pair_key])


#         first_avg=np.mean(list1)
#         second_avg=np.mean(list2)

#         final_ans=(2*second_avg+first_avg)/3

#         list_1.append(final_ans)


# print(np.mean(list_0))
# print(np.mean(list_1))

csv_file_path = "objects_test.csv"

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)

print("DataFrame saved to", csv_file_path)