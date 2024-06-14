#Load source dataset (EdmondsDanceDataset) and create train and evaluation sets
import pandas as pd

source_data = pd.read_csv('clean_EdmondsDance.csv', index_col=False)
target_features = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation']
labels = source_data[target_features]
X = source_data['Lyrics']
y = labels

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

data_train = pd.concat([X_train, y_train], axis=1) # 'text' and 'labels' are the standard names used in PyTorch implementation of BERT
data_eval = pd.concat([X_val, y_val], axis=1)

data_train.head()

#! pip install datasets
from datasets import Dataset, DatasetDict

train_dataset = Dataset.from_pandas(data_train)
eval_dataset = Dataset.from_pandas(data_eval)
dataset = DatasetDict({"train":train_dataset,"eval":eval_dataset})

# labels mapping
id2label = {idx: label for idx, label in enumerate(target_features)}
label2id = {label: idx for idx, label in enumerate(target_features)}
id2label



import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import accelerate

tokenizer = AutoTokenizer.from_pretrained("ayoubkirouane/BERT-Emotions-Classifier")

# data preparation

MAX_LENGTH = 512

def prepare_data(data):
  # 1. get a batch and encode it
  text = data['Lyrics']
  encoding = tokenizer(text, truncation=True, padding='max_length', max_length=MAX_LENGTH)
  # 2. matching labels & array fill
  batch_labels = {key: data[key] for key in data.keys() if key in target_features}
  matrix_labels = np.zeros((len(text), len(target_features)))
  for idx, label in enumerate(target_features):
    matrix_labels[:, idx] = batch_labels[label]
  encoding['labels'] = matrix_labels.tolist()

  return encoding

encoded_data = dataset.map(prepare_data, batched=True, remove_columns=dataset['train'].column_names)
encoded_data['train'][0].keys()

encoded_data.set_format('torch')

# Load the model
from BertModelForMultiLabelSequenceClassification import BertModelForMultiLabelSequenceClassification
from transformers import PretrainedConfig

conf = PretrainedConfig.from_pretrained('ayoubkirouane/BERT-Emotions-Classifier')
conf.num_labels = len(target_features)
conf.id2label = id2label
conf.label2id = label2id
conf.problem_type = 'multi_label_classification'
model = BertModelForMultiLabelSequenceClassification(conf)

import wandb
from lightning.pytorch.loggers import WandbLogger

wandb.init(project="RefactoredTL", name='final_100ep', config=conf)
wandb_logger = WandbLogger(project='RefactoredTL')




from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, multilabel_confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score
from transformers import EvalPrediction
import torch
import matplotlib.pyplot as plt


def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='micro')
    precision = precision_score(y_true, y_pred, average='micro')
    wandb.log({"f1": f1_micro_average, "roc_auc": roc_auc, "accuracy": accuracy, "recall": recall, "precision": precision})
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy,
               'precision': precision,
               'recall': recall}
    
    fig, axes = plt.subplots(2, 4)
    for i in range(0, len(target_features)):
        disp = ConfusionMatrixDisplay.from_predictions(y_true=y_true[:, i], y_pred=y_pred[:, i])
        row = i // 4
        col = i % 4
        disp.plot(ax=axes[row, col], values_format='.4g')
        disp.ax_.set_title(f'class {id2label[i]}')
        if row==0:
            disp.ax_.set_xlabel('')
        if col!=0:
            disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()
    plt.subplots_adjust(wspace=1, hspace=0.5)
    fig.colorbar(disp.im_, ax=axes)
    wandb.log({'chart': wandb.Image(fig)})
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    
    return result

batch_size = 8


training_args = TrainingArguments(
              output_dir="checkpoints/trainer_refactored",
              eval_strategy = "epoch",
              save_strategy = "epoch",
              learning_rate=0.00001,
              per_device_train_batch_size=batch_size,
              per_device_eval_batch_size=batch_size,
              num_train_epochs= 100,
              weight_decay=0.01,
              load_best_model_at_end=True,
              metric_for_best_model='f1',
              save_total_limit = 2, 
              report_to="wandb", 
              logging_strategy="epoch",
              #resume_from_checkpoint='checkpoints/trainer_refactored/checkpoint-3551'
          )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_data["train"],
    eval_dataset=encoded_data["eval"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

wandb.watch(model, log="all", log_freq=10)

trainer.train() #resume_from_checkpoint='checkpoints/trainer_refactored/checkpoint-3551')


wandb.finish()




