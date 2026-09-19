import os
import zipfile
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def main():
    base_dir = r"C:\Users\venka\Downloads\ElevateLabs"
    gtzan_dir = os.path.join(base_dir, "GTZAN")
    zip_path = os.path.join(gtzan_dir, "gtzan-dataset-music-genre-classification.zip")

    # Unzip if zipped
    if os.path.exists(zip_path) and not os.path.exists(os.path.join(gtzan_dir, "Data")):
        print("Extracting GTZAN dataset archive...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(gtzan_dir)
        print("Extraction complete.")

    # Find features_3_sec.csv
    csv_path = None
    for root, dirs, files in os.walk(gtzan_dir):
        if "features_3_sec.csv" in files:
            csv_path = os.path.join(root, "features_3_sec.csv")
            break

    if not csv_path:
        print("CSV dataset not found.")
        return

    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")

    # Drop non-feature columns
    drop_cols = ['filename', 'length', 'label']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].values
    y_raw = df['label'].values

    # Encoding
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    classes = encoder.classes_
    print("Genre classes:", list(classes))

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Train Random Forest Classifier (Fast & High Accuracy)
    print("Training Random Forest Audio Classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n==========================================")
    print(f"   Model Test Accuracy: {acc*100:.2f}%")
    print(f"==========================================\n")

    print(classification_report(y_test, y_pred, target_names=classes))

    # Save artifacts
    model_path = os.path.join(base_dir, "music_genre_model.pkl")
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    encoder_path = os.path.join(base_dir, "label_encoder.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)

    print(f"Saved artifacts to {base_dir}")

    # Save Confusion Matrix plot
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=classes, yticklabels=classes)
    plt.title(f"GTZAN Music Genre Classification - Confusion Matrix (Accuracy: {acc*100:.2f}%)")
    plt.xlabel("Predicted Genre")
    plt.ylabel("True Genre")
    plt.tight_layout()
    cm_img_path = os.path.join(base_dir, "confusion_matrix.png")
    plt.savefig(cm_img_path, dpi=300)
    plt.close()

    # Save Feature Importance plot
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]
    plt.barh(range(len(indices)), importances[indices], color='#1DB954')
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
    plt.title("Top 15 Audio Feature Importances")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    loss_img_path = os.path.join(base_dir, "training_loss.png")
    plt.savefig(loss_img_path, dpi=300)
    plt.close()

    # Save metrics JSON for documentation generator
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    metrics_data = {
        "accuracy": float(acc),
        "classes": list(classes),
        "per_class": {
            cls: {
                "precision": float(p),
                "recall": float(r),
                "f1_score": float(f)
            }
            for cls, p, r, f in zip(classes, precision, recall, f1)
        }
    }
    with open(os.path.join(base_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=2)

    print("Model training, evaluation, and plot generation completed successfully!")

if __name__ == "__main__":
    main()
