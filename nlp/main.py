import re, gc, os, sys
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import logging
import datasets
from datasets import load_dataset

from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, SequentialSampler

from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

from torch_bert_classification import set_global_logging_level

set_global_logging_level(logging.ERROR, ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"])

# TODO: parameterize file path/ model name
# TODO: basically prove that we can scale this to two or more models

def index_labels(input_df):
  """
  """
  label_dict = {}
  for i in range(0, len(set(input_df.label))):
    label_dict[le.classes_[i]] = i
  return label_dict


def evaluate(dataloader_val, model, device):
  """
  """
  model.eval()
  loss_val_total = 0
  predictions, true_vals = [], []
  for batch in dataloader_val:
      batch = tuple(b.to(device) for b in batch)
      inputs = {'input_ids':      batch[0],
                'attention_mask': batch[1],
                'labels':         batch[2],
                }
      with torch.no_grad():        
          outputs = model(**inputs)
      loss = outputs[0]
      logits = outputs[1]
      loss_val_total += loss.item()
      logits = logits.detach().cpu().numpy()
      label_ids = inputs['labels'].cpu().numpy()
      predictions.append(logits)
      true_vals.append(label_ids)
  loss_val_avg = loss_val_total/len(dataloader_val) 
  predictions = np.concatenate(predictions, axis=0)
  true_vals = np.concatenate(true_vals, axis=0)
  return loss_val_avg, predictions, true_vals


def load_data(model_name, model, device, data_df):
    """
    """
    model.to(device)
    model.load_state_dict(torch.load(f'./models/{model_name}.model', map_location=torch.device(device)))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    print('Vectorizing evaluation data')
    encoded_data_test = tokenizer.batch_encode_plus(
        data_df['text'].values, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )
    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(data_df.label.values)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    dataloader_test = DataLoader(dataset_test, 
                                sampler=SequentialSampler(dataset_test), 
                                batch_size=batch_size)
    return dataloader_test


if __name__ == "__main__":
    """
    """
    batch_size = 3
    file_path = './data/dnb_care_sales-channel_data.csv'
    model_name = re.sub('./data/', '', file_path)
    model_name = re.sub('.csv', '', model_name)

    # read the training data to get a label index
    print(f'Reading training data from {file_path}')
    le = preprocessing.LabelEncoder()
    json_df = pd.read_csv(file_path)
    json_df = json_df[['text', 'Rating']]
    json_df.columns = ['text', 'label']
    json_df['label'] = le.fit_transform(json_df['label'])
    json_df['text'] = json_df['text'].astype(str)
    json_df['text'] = json_df['text'].str.encode('utf-8')
    data_df = json_df

    sample_df = data_df[data_df.label == 0][0:1]
    sample_df.to_csv('./data/sample_df.csv', index=False)
    sample_df = pd.read_csv('./data/sample_df.csv')
    print(f'Test data:\n{sample_df.head()}\n')

    label_dict = index_labels(data_df)
    print(f'Labels:\n{label_dict}\n')

    print(f'Loading the transformer model')
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                        num_labels=len(label_dict),
                                                        output_attentions=False,
                                                        output_hidden_states=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading {model_name}')
    dataloader_test = load_data(model_name, model, device, sample_df)

    _, predictions, true_vals = evaluate(dataloader_test, model, device)
    predictions

    preds_flat = np.argmax(predictions, axis=1).flatten()
    response = le.inverse_transform(preds_flat)[0]
    print(f'Predicted label: {response}')