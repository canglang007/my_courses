import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_score, confusion_matrix
import numpy as np
import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt

from model import RNN,LSTM,GRU
from dataloader import get_dataloader

def evaluate(model, test_loader, device='cuda'):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating', unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

    return all_labels, all_predictions


data_dir = './aclImdb'
batch_size = 64
num_steps = 500
val_split = 0.2
model_name = 'RNN'

_, _, test_loader, vocab_size ,_= get_dataloader(data_dir, batch_size, num_steps, val_split)

model_path = f'./checkpoints/{model_name}_best_model.pth'
if model_name == 'RNN':
    model = RNN(vocab_size=vocab_size, embedding_dim=512, hidden_size=128)
elif model_name == 'LSTM':
    model = LSTM(vocab_size=vocab_size, embedding_dim=512, hidden_size=128)
elif model_name == 'GRU':
    model = GRU(vocab_size=vocab_size, embedding_dim=512, hidden_size=128)    

model.load_state_dict(torch.load(model_path))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

all_labels, all_predictions = evaluate(model, test_loader, device)
print(len(all_labels), len(all_predictions))
accuracy = accuracy_score(all_labels, [1 if pred[1] >= 0.5 else 0 for pred in all_predictions])
recall = recall_score(all_labels, [1 if pred[1] >= 0.5 else 0 for pred in all_predictions])
f1 = f1_score(all_labels, [1 if pred[1] >= 0.5 else 0 for pred in all_predictions])
auroc = roc_auc_score(all_labels, [1 if pred[1] >= 0.5 else 0 for pred in all_predictions])
precision = precision_score(all_labels, [1 if pred[1] >= 0.5 else 0 for pred in all_predictions])
conf_matrix = confusion_matrix(all_labels, [1 if pred[1] >= 0.5 else 0 for pred in all_predictions])



print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUROC: {auroc:.4f}")
print(f"Precision: {precision:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

results_dir = 'test_results'
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, 'evaluation_results.csv')

data = []
for inputs, labels in tqdm(test_loader, desc='Collecting data', unit='batch'):
    inputs = inputs.cpu().numpy()
    labels = labels.cpu().numpy()
    outputs = model(torch.tensor(inputs, device=device))
    
    for idx in range(len(inputs)):
        if outputs[idx][0] > 0.5:
            i = 0
        else:
            i = 1 
        row = {
            'Input Data': inputs[idx],
            'Label': labels[idx],
            'Prediction': i
        }
        data.append(row)


data_df = pd.DataFrame(data)
data_file = os.path.join(results_dir, 'input_label_prediction.csv')
data_df.to_csv(data_file, index=False)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
plt.show()

results_data = {
    'Accuracy': [accuracy],
    'Recall': [recall],
    'F1-score': [f1],
    'AUROC': [auroc],
    'Precision': [precision],
}

results_df = pd.DataFrame(results_data)
results_df.to_csv(results_file, index=False)
