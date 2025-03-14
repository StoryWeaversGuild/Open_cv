📌 Sentiment Analysis from Video using OpenCV and CNN
🚀 Project Overview
This project performs Sentiment Analysis by detecting and classifying human emotions from video clips using OpenCV and a Convolutional Neural Network (CNN). The system identifies emotions like Happy, Sad, Angry, Fear, Surprise, Disgust, and Neutral from short video files.
download the FER 2013 dataset from kaggle for this project.

dataset/
├── train/
│   ├── Angry/
│   ├── Disgust/
│   ├── Fear/
│   ├── Happy/
│   ├── Sad/
│   ├── Surprise/
│   └── Neutral/
└── validation/
    ├── Angry/
    ├── Disgust/
    ├── Fear/
    ├── Happy/
    ├── Sad/
    ├── Surprise/
    └── Neutral/
After downloading, extract the dataset and place it inside a folder named dataset in the project directory.
The folder structure should look like this:
open_cv/
├── dataset/
│   ├── train/
│   └── test/
├── emotion_model.h5
├── sample.mp4
└── video_processing-checkpoint.ipynb
🎉 Results
The program will display real-time emotion detection on video frames and print the sentiment summary at the end.
💡 Technologies Used
Python
OpenCV
TensorFlow/Keras
NumPy
📧 Contributi
