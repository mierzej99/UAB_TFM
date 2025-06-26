#Raster map
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
import matplotlib.pyplot as plt

states = {0:'aw', 1:'qw', 2:'nrem', 3:'rem'}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mouse_id", type=str, required=True,
                        help="mouse id like ESPM113")
        
    return parser.parse_args()

def train_classifier(data, mouse_id):
    # Split features and labels
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Split data: 80% train+val, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # # Split train+val: 80% train, 20% val (from original dataset)
    # X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    # Train Random Forest with default parameters
    rf = RandomForestClassifier(random_state=42, verbose=4, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Save model
    filename_model = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/ml/random_forest_model.pck"
    with open(filename_model, 'wb') as f:
        pickle.dump(rf, f)
    
    return rf, X_test, y_test

def evaluate_model(model, X_test, y_test, plotname):
    # Predictions
    y_pred = model.predict(X_test)
    
    # Statistics - tylko accuracy i F1
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    # Prosty wykres
    plt.figure(figsize=(8, 4))
    
    # Wykres słupkowy z metrykami
    plt.subplot(1, 2, 1)
    metrics = ['Accuracy', 'F1-Score']
    values = [accuracy, f1]
    bars = plt.bar(metrics, values, color=['blue', 'green'])
    plt.ylim(0, 1)
    # plt.title('Model Performance')
    plt.ylabel('Score')
    
    # Dodaj wartości na słupkach
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # Confusion matrix
    plt.subplot(1, 2, 2)
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Etykiety osi z mapowaniem states
    labels = [states[i] for i in range(len(states))]
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Dodaj wartości do confusion matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(plotname, dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def pipeline(mouse_id, stimuli=False):
    filename_umap = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/umap/umap_embeddings.pck"
    with open(filename_umap, "rb") as pk:
        embedding = pickle.load(pk)
    embedding = pd.DataFrame(embedding)
    
    rf, X_test, y_test = train_classifier(embedding, mouse_id)
    plotname = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/ml/model_evaluation.png"
    evaluate_model(rf, X_test, y_test, plotname)


def main(args):
    print("Starting raw")
    pipeline(args.mouse_id)


    # print("Starting smooth")
    # pca_calculation(args.mouse_id, smooth=True)
        

if __name__ == "__main__":
    args = parse_args()
    main(args)