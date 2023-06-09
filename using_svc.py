import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

features_dir = 'extracted_features'
os.makedirs(features_dir, exist_ok=True)

def preprocess_audio(file_path):
    features_file = os.path.join(features_dir, os.path.splitext(os.path.basename(file_path))[0] + '.npy')
    if os.path.exists(features_file):
        print(f"Loading features from {features_file}")
        return np.load(features_file)
    
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', mono=True)
    
    pitch = librosa.yin(audio, fmin=100, fmax=1000)
    energy = np.mean(audio ** 2)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate)[0]
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    max_len = 1000
    pitch = np.pad(pitch, (0, max_len - len(pitch)))
    mfccs = np.pad(mfccs, ((0, 0), (0, max_len - mfccs.shape[1])))
    delta_mfccs = np.pad(delta_mfccs, ((0, 0), (0, max_len - delta_mfccs.shape[1])))
    delta2_mfccs = np.pad(delta2_mfccs, ((0, 0), (0, max_len - delta2_mfccs.shape[1])))
    features = np.concatenate((pitch, [energy], [tempo], mfccs.flatten(), delta_mfccs.flatten(), delta2_mfccs.flatten()))
    
    np.save(features_file, features)
    
    print(f"Extracted features saved to {features_file}")
    return features

train_df['audio_features'] = train_df['path'].apply(preprocess_audio)
test_df['audio_features'] = test_df['path'].apply(preprocess_audio)

X = np.stack(train_df['audio_features'].values)
y = train_df['label'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

print("Training the model...")
model = SVC()
model.fit(X_train, y_train_encoded)

val_predictions = model.predict(X_val)
val_accuracy = accuracy_score(y_val_encoded, val_predictions)
print("Validation accuracy:", val_accuracy)


X_test = np.stack(test_df['audio_features'].values)
test_predictions = model.predict(X_test)
test_df['predicted_label'] = label_encoder.inverse_transform(test_predictions)


test_df.to_csv('test_predictions.csv', index=False)



submission_df = pd.DataFrame({
    'id': test_df['id'],
    'label': test_df['predicted_label']
})
submission_df.to_csv('submission.csv', index=False)
