import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATA_DIR = 'landmarks_dataset2'

X, y = [], []

for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)
    for npy_file in os.listdir(label_path):
        file_path = os.path.join(label_path, npy_file)
        landmarks = np.load(file_path)
        if landmarks.shape == (63,): 
            X.append(landmarks)
            y.append(label)

X = np.array(X)
y = np.array(y)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

print('training...')
model = RandomForestClassifier(bootstrap=False, max_depth=30, min_samples_leaf=2,n_estimators=300)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))


joblib.dump(model, 'RFC_model3.pkl')
joblib.dump(le, 'label_encoder.pkl')



