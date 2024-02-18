# File Structure:

1) add_file.py: This python script divides the images in the dataset into 2 folders based on the json file for train and test, in the new directory dataset_hate:

a) 1 folder contains the hateful memes
b) 1 folder contains the non-hateful memes

It does the same thing for text in the json files.

2) caption_impact.py: This code analyses the impact of the captions of memes.

3) image_detection.py: This code creates csv files based on the objects detected in the memes in train and test set.

4) image_detection_main.ipynb: This code analyses the impact of objects in memes(hateful and non-hateful) in train and test set.

5) meme_classifier.ipynb: This code classifies whether an image is a meme or not.

6) model_1.h5, model_2.h5, model_3.h5, model_4.h5, model_5.h5: These models are the models trained of 5 k-folds to predict if a meme is hateful or not.

7) objects_test.csv: This is the csv file for the test set.

8) objects_train_kfold.csv: This is the csv file for the train set

9) output.png: This is the graph analysing the toxicity scores of hateful and non-hateful memes in the test set.

10) text_model.ipynb: This code predicts whether a meme is hateful solely based on the analysis of its caption.

11) total_model.ipynb: This code predicts whether a meme is hateful or not based on the objects detected in it and also implements a multimodal predication using both text and image for analysis.

# Assumptions:

1) I am not submitting the images/memes directory which I used for training/testing.
2) For minimizing the impact of caption analysis, I have tried to use inpainting to fill out the caption with the surrounding parts of the image. But I couldn't run the model because it required CUDA drivers and GPU and my laptop doesn't have those, and its to big to run on google collab. Here is the link of the github repo for the model:


