# File Structure:

## 1) add_file.py: This python script divides the images in the dataset into 2 folders based on the json file for train and test, in the new directory dataset_hate:

1) 1 folder contains the hateful memes
2) 0 folder contains the non-hateful memes

## It does the same thing for text in the json files.

## 2) caption_impact.py: This code analyses the impact of the captions of memes.

## 3) image_detection.py: This code creates csv files based on the objects detected in the memes in train and test set.

## 4) image_detection_main.ipynb: This code analyses the impact of objects in memes(hateful and non-hateful) in train and test set.

## 5) meme_classifier.ipynb: This code classifies whether an image is a meme or not.

## 6) model_1.h5, model_2.h5, model_3.h5, model_4.h5, model_5.h5: These models are the models trained of 5 k-folds to predict if a meme is hateful or not.

## 7) objects_test.csv: This is the csv file for the test set.

## 8) objects_train_kfold.csv: This is the csv file for the train set

## 9) output.png: This is the graph analysing the toxicity scores of hateful and non-hateful memes in the test set.

## 10) text_model.ipynb: This code predicts whether a meme is hateful solely based on the analysis of its caption.

## 11) total_model.ipynb: This code predicts whether a meme is hateful or not based on the objects detected in it and also implements a multimodal predication using both text and image for analysis.

## 12) test_impainting.py: This code impaints an image.

## 13) Paper_Report.pdf: This file is the report of the paper reading task.

## 14) Analyzing_hateful_memes__Report.pdf: This file explains the way I approached and implemented the tasks.

# Assumptions:

## 1) I am not submitting the images/memes directory which I used for training/testing.

# Resources referred to:

## https://medium.datadriveninvestor.com/memes-detection-android-app-using-deep-learning-d2c65347e6f3
## https://medium.com/codex/hateful-meme-detection-3c5a47097a08
## https://docs.ultralytics.com/
## https://www.analyticsvidhya.com/blog/2021/12/fine-tune-bert-model-for-sentiment-analysis-in-google-colab/


