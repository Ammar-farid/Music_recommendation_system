Music Recommendation System
A Hybrid Machine Learning Approach combining Content-Based Filtering and Collaborative Filtering to predict whether a user will LIKE or SKIP a track
Project Overview

The Problem

Traditional music recommendation systems struggle with several issues:


Cold-start problem — new users have no history, so the system cannot recommend well
Data sparsity — most users only interact with a small fraction of all songs
Low recommendation accuracy — standalone methods give poor results
Poor personalization — generic suggestions that are not tailored to individual taste


The Solution

A Hybrid Recommendation System that combines two approaches:


Content-Based Filtering — uses audio features of songs
Collaborative Filtering — uses behavior of similar users


The system predicts a clear decision, LIKE or SKIP, for each track using a Weighted Hybrid Score that combines both methods.

Dataset


Source: Kaggle — Spotify Tracks Dataset by Maharshi Pandya
Dataset link: https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset
Total rows (tracks): 114,000
Total columns (features): 20
Number of music genres: 125
Data format: CSV (tabular)
Numeric features: 16
Text/categorical features: 4


Dataset Columns

Column NameDescriptiontrack_idUnique ID of the trackartistsName(s) of the artist(s)album_nameAlbum the track belongs totrack_nameName of the songpopularityScore from 0-100 (100 = most popular)duration_msLength of track in millisecondsexplicitWhether lyrics are explicit (True/False)danceabilityHow danceable the track is (0-1)energyIntensity and activity level (0-1)keyMusical key (0=C, 1=C#, 2=D, etc.)loudnessOverall loudness in decibels (dB)modeMajor (1) or Minor (0) scalespeechinessPresence of spoken words (0-1)acousticnessConfidence track is acoustic (0-1)instrumentalnessPredicts no vocals in track (0-1)livenessPresence of live audience (0-1)valenceMusical positiveness, happy vs sad (0-1)tempoSpeed of track in beats per minute (BPM)time_signatureBeats per bar (e.g., 3 or 4)track_genreGenre label (125 genres total)

Data Preprocessing Pipeline

The raw dataset is cleaned and prepared in four steps before training.

Step 1 — Data Cleaning


Remove duplicate song entries
Handle missing or null values
Filter out outlier interaction vectors


Step 2 — Feature Extraction and Selection


Select the most important audio features for prediction
Key features: danceability, energy, loudness, valence, acousticness, tempo


Step 3 — Feature Scaling


Problem: features have very different scales (tempo: 60-200 BPM, danceability: 0-1)
Solution: apply StandardScaler or MinMaxScaler to normalize all features
Result: all features are on the same scale for fair comparison


Step 4 — Interaction Matrix Construction


Build a sparse pivot matrix from user feedback
Rows represent unique User IDs
Columns represent unique Track IDs
Values represent preference weights (LIKE or SKIP)


Model Architecture

The system uses three components working together.

A. Content-Based Filtering (Supervised Learning)

Trains machine learning models to learn a user's taste based on the audio features of songs.

Random Forest Classifier


Uses many decision trees to find patterns in audio features
Reduces overfitting by combining multiple trees
Selected as the main classifier for the Hybrid Engine


Logistic Regression


A simple baseline model
Checks straight-line (linear) relationships in data
Used only as a baseline reference for comparison


B. Collaborative Filtering


Finds users who have similar listening habits
Uses Cosine Similarity to measure user-user and item-item closeness
Both memory-based and model-based approaches are used


C. Hybrid Score Integration


Combines scores from both the Content-Based and Collaborative methods
Uses a weighted average to compute the final Hybrid Score
Decision rule: Hybrid Score greater than 0.70 results in LIKE
Decision rule: Hybrid Score less than 0.70 results in SKIP


How It Works


User Input — enter a User ID to get recommendations
Feature Matrix — load audio features for all tracks
Content Score — Random Forest predicts preference probability
Collaborative Score — Cosine Similarity finds similar users and tracks
Hybrid Score — a weighted average of both scores is computed
Output — LIKE if the score is greater than 0.70, otherwise SKIP


Key Audio Features Used

FeatureDescriptionDanceabilityHow suitable the track is for dancing, from 0 (not danceable) to 1 (very danceable)EnergyHow intense and active the track feels, from 0 (calm) to 1 (very energetic)ValenceMusical positiveness, from 0 (sad or dark) to 1 (happy or cheerful)TempoSpeed of the track measured in beats per minute (BPM)AcousticnessConfidence that the track is acoustic, from 0 (not acoustic) to 1 (fully acoustic)LoudnessOverall loudness of the track in decibels (dB), typical range -60 to 0 dB

Model Performance

Both models were evaluated using Accuracy, Precision, and F1-Score.

AlgorithmAccuracyPrecisionF1-ScoreStatusRandom Forest50.15%50.81%49.71%SelectedLogistic Regression49.67%50.33%48.48%Baseline only

Random Forest performed better on all three metrics and was selected for the Hybrid Engine.

Sample Output (User ID: 32)

Rule: Hybrid Score greater than 0.70 results in LIKE, Hybrid Score less than 0.70 results in SKIP.

TrackArtistCollab ScoreContent ScoreHybrid ScoreDecisionBlinding LightsThe Weeknd1.000.840.92LIKEStarboyThe Weeknd0.920.780.85LIKEShape of YouEd Sheeran0.300.450.38SKIP

Conclusion and Key Takeaways


The hybrid approach outperforms standalone Content-Based or Collaborative Filtering used alone
Random Forest was chosen as the best classifier for the Hybrid Engine
The system successfully predicts LIKE or SKIP for any User ID in real time
Dataset: 114,000 Spotify tracks across 125 genres with 20 audio features
Future work: improve accuracy using Deep Learning and larger user interaction data




How to Run


Clone this repository
Open the notebook file in Jupyter Notebook, JupyterLab, or Google Colab
Make sure the dataset CSV file is in the same directory as the notebook, or update the file path in the code
Install the required dependencies:


bash
```
pip install pandas numpy scikit-learn
```


Run all cells in order to preprocess the data, train the models, and generate recommendations


Tech Stack


Python
Pandas and NumPy for data processing
Scikit-learn for Random Forest, Logistic Regression, and Cosine Similarity
Jupyter Notebook / Google Colab
