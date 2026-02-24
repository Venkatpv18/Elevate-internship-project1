Music Genre Classification using Deep Learning
Introduction

Music genre classification is the process of automatically categorizing music into genres such as rock, jazz, classical, pop, and more.

With the rapid growth of digital music platforms, automatic genre classification plays an important role in:

🎧 Music recommendation systems

📂 Playlist creation

🎶 Music organization

This project builds a Deep Learning-based Music Genre Classification system that predicts the genre of an audio file using feature extraction and a neural network classifier.

📖 Abstract

The objective of this project is to design a Music Genre Classification system using Deep Learning techniques.

Dataset Used: GTZAN Music Genre Dataset

Total Genres: 10

Feature Extraction: MFCC (Mel-Frequency Cepstral Coefficients)

Model: Artificial Neural Network (ANN)

Framework: TensorFlow & Keras

Train-Test Split: 80-20

The system successfully classifies music into 10 genres and performs well on unseen test data.

📂 Dataset
🎼 GTZAN Music Genre Dataset
4

The project uses the GTZAN Music Genre Dataset, which contains:

1000 audio files

10 genres:

Blues

Classical

Country

Disco

Hiphop

Jazz

Metal

Pop

Reggae

Rock

Each audio file is 30 seconds long.

🛠️ Tools & Technologies Used

Python

Google Colab

Librosa (Audio Processing)

NumPy

Scikit-learn

TensorFlow / Keras

Matplotlib

⚙️ Project Workflow
Step 1: Dataset Collection

The GTZAN dataset was downloaded from Kaggle.

Step 2: Data Preprocessing

Audio files loaded using Librosa

Sampling rate: 22050 Hz

Step 3: Feature Extraction

Extracted 40 MFCC coefficients

Computed mean of MFCC values

Created fixed-length feature vectors

Step 4: Label Encoding

Converted genre labels to numeric values using LabelEncoder

Applied One-Hot Encoding

Step 5: Train-Test Split

80% Training Data

20% Testing Data

Step 6: Model Building

Neural Network Architecture:

Dense Layer (256 units, ReLU)

Dropout Layer

Dense Layer (128 units, ReLU)

Dropout Layer

Dense Layer (64 units, ReLU)

Output Layer (10 units, Softmax)

Step 7: Model Training

Epochs: 50

Optimizer: Adam

Loss Function: Categorical Crossentropy

Step 8: Model Evaluation

Performance evaluated using:

Accuracy Score

Confusion Matrix

Classification Report

The trained model was saved in .h5 format for future use.

📊 Model Performance

The model achieved good accuracy on unseen test data and demonstrated that deep learning can effectively classify audio signals using MFCC features.

🚀 Future Improvements

✅ Implement CNN for better performance

✅ Use spectrogram images as input

✅ Increase dataset size

✅ Perform hyperparameter tuning

✅ Apply data augmentation

▶️ How to Run the Project
# Clone the repository
git clone <repository-link>

# Install required libraries
pip install -r requirements.txt

# Run the notebook
Open internship_project.ipynb in Google Colab or Jupyter Notebook
📌 Conclusion

This project demonstrates the effectiveness of Deep Learning in Audio Signal Processing.

Using MFCC feature extraction and an Artificial Neural Network, the system successfully classifies music into 10 different genres with good accuracy.
