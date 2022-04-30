# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import re, gc, os, sys
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

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

# TODO: paramterize input file, shrink, epochs, batch size
# TODO: change print statements into logging statements
# TODO: upgrade AdamW to use the newer package
# TODO: make a main.py that loads and interprets models


LANGUAGE = "english"
SENTENCES_COUNT = 10


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def shrink_data(input_df, shrink=250):
  """
  """
  output_df = pd.DataFrame()
  for l in list(set(input_df.label)):
    output_df = pd.concat([output_df, input_df[input_df.label == l][:shrink]], 
                        ignore_index=True)
  return output_df


def normalize_text(text):
    """
    """
    norm = re.sub('\([0-9]{1,}\.[0-9]{1,}\) Speaker [0-9]{1,}:', '', text)
    norm = re.sub('<hesitation-[0-9]{1,}\.[0-9]{1,}>', '', norm)
    norm = re.sub('\n', '', norm)
    return norm


def summarize(text):
  """
  """
  parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
  stemmer = Stemmer(LANGUAGE)
  summarizer = Summarizer(stemmer)
  summarizer.stop_words = get_stop_words(LANGUAGE)
  summary = [sentence for sentence in summarizer(parser.document, SENTENCES_COUNT)]
  str_sum = ''.join([re.sub('Sentence:', '', str(s)) for s in summary])
  return str_sum


def index_labels(input_df):
  """
  """
  label_dict = {}
  for i in range(0, len(set(input_df.label))):
    label_dict[le.classes_[i]] = i
  return label_dict


def imdb_data():
  """
  """
  train_data, test_data = datasets.load_dataset('imdb', 
                                                split=['train','test'], 
                                                cache_dir='./cache')
  train_df = pd.DataFrame(train_data)
  test_df = pd.DataFrame(test_data)
  label_dict = [0,1]
  return train_data, test_data


def f1_score_func(preds, labels):
  """
  """
  preds_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  return f1_score(labels_flat, preds_flat, average='weighted')


def accuracy_per_class(preds, labels):
  """
  """
  label_dict_inverse = {v: k for k, v in label_dict.items()}
  preds_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  for label in np.unique(labels_flat):
      y_preds = preds_flat[labels_flat==label]
      y_true = labels_flat[labels_flat==label]
      print(f'Class: {label_dict_inverse[label]}')
      print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')


def evaluate(dataloader_val):
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


def train_model(model_name):
  """
  """
  for epoch in tqdm(range(1, epochs+1)):
    model.train()  
    loss_train_total = 0
    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:
        model.zero_grad()
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                  }       
        outputs = model(**inputs)
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
    torch.save(model.state_dict(), f'./models/{model_name}_{epoch}.model')
    tqdm.write(f'\nEpoch {epoch}')
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')
    gc.collect()


if __name__ == "__main__":
  """
  """
  batch_size = 3
  epochs = 5
  shrink = 300
  seed_val = 17

  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)

  set_global_logging_level(logging.CRITICAL, ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"])

  file_path = './data/dnb_care_sales-channel_data.csv'
  model_name = re.sub('./data/', '', file_path)
  model_name = re.sub('.csv', '', model_name)

  # read the training data
  print(f'Reading training data from {file_path}')
  le = preprocessing.LabelEncoder()
  json_df = pd.read_csv(file_path)
  json_df = json_df[['text', 'Rating']]
  json_df.columns = ['text', 'label']
  json_df['label'] = le.fit_transform(json_df['label'])
  json_df['text'] = json_df['text'].astype(str)

  # reduce the size
  data_df = shrink_data(json_df, shrink=shrink)
  label_dict = index_labels(data_df)
  # preprocess the text data
  data_df['text'] = data_df['text'].apply(lambda x: normalize_text(x))
  # print(f'Summarizing text inputs') # experimental
  # data_df['text'] = data_df['text'].apply(lambda x: summarize(x))
  data_df['text'] = data_df['text'].str.encode('utf-8')

  # split it into 90% training data and 10% test data
  X_train, X_test, Y_train, Y_test = train_test_split(data_df['text'], data_df['label'], test_size=0.1, random_state=1)
  # deep magic
  pd.DataFrame({'text': X_train, 'label': Y_train}).to_csv('./data/train.csv', index=False)
  pd.DataFrame({'text': X_test, 'label': Y_test}).to_csv('./data/test.csv', index=False)
  train_df = pd.read_csv('./data/train.csv')
  test_df = pd.read_csv('./data/test.csv')

  # load the tokenization model
  print('Loading tokenizer')
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

  print('Vectorizing training data')
  # vectorize the training split
  encoded_data_train = tokenizer.batch_encode_plus(
      train_df['text'].values, 
      add_special_tokens=True, 
      return_attention_mask=True, 
      truncation=True,
      padding='max_length',
      max_length=512, 
      return_tensors='pt'
  )

  print('Vectorizing test data')
  # vectorize the test split
  encoded_data_val = tokenizer.batch_encode_plus(
      test_df['text'].values, 
      add_special_tokens=True, 
      return_attention_mask=True, 
      truncation=True,
      padding='max_length', 
      max_length=512, 
      return_tensors='pt'
  )

  # vectorize the training labels
  input_ids_train = encoded_data_train['input_ids']
  attention_masks_train = encoded_data_train['attention_mask']
  labels_train = torch.tensor(train_df.label.values)

  # vectorize the test labels
  input_ids_val = encoded_data_val['input_ids']
  attention_masks_val = encoded_data_val['attention_mask']
  labels_val = torch.tensor(test_df.label.values)

  # combine the vectorized data and labels into tensor objects
  dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
  dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

  # load the bert model
  print('Loading the transformer model')
  model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                        num_labels=len(label_dict),
                                                        output_attentions=False,
                                                        output_hidden_states=False)

  # deep magic
  dataloader_train = DataLoader(dataset_train, 
                                sampler=RandomSampler(dataset_train), 
                                batch_size=batch_size)

  dataloader_validation = DataLoader(dataset_val, 
                                    sampler=SequentialSampler(dataset_val), 
                                    batch_size=batch_size)

  optimizer = AdamW(model.parameters(),
                    lr=1e-5, 
                    eps=1e-8,
                    no_deprecation_warning=True)

  scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps=0,
                                              num_training_steps=len(dataloader_train) * epochs)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model.to(device)

  print(f'Training {model_name}')
  train_model(model_name=model_name)

  # evaluate the model against the test data
  print('\nCalculating per-class accuracy\n')
  model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                        num_labels=len(label_dict),
                                                        output_attentions=False,
                                                        output_hidden_states=False)

  model.to(device)
  model.load_state_dict(torch.load(f'./models/{model_name}_1.model', map_location=torch.device(device)))

  _, predictions, true_vals = evaluate(dataloader_validation)
  accuracy_per_class(predictions, true_vals)
  gc.collect()

  os.rename(f'./models/{model_name}_1.model', f'./models/{model_name}.model')
  os.remove(f'./models/{model_name}_2.model')
  os.remove(f'./models/{model_name}_3.model')
  os.remove(f'./models/{model_name}_4.model')
  os.remove(f'./models/{model_name}_5.model')