# -*- coding: utf-8 -*-


# Hướng dẫn về bộ dữ liệu này được thực hiện trong Lab Tokenizer 
!wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json -O /tmp/sarcasm.json
  
import json

with open("/tmp/sarcasm.json", 'r') as f:
    datastore = json.load(f)

datastore[0]

"""## 2. Tokenizer"""

dataset = []
label_dataset = []

for item in datastore:
    dataset.append(item["headline"])
    label_dataset.append(item["is_sarcastic"])

dataset[:10], label_dataset[:10]

import numpy as np

dataset = np.array(dataset)
label_dataset = np.array(label_dataset)

train_size = 0.8
size = int(len(dataset) * train_size)

train_sentence = dataset[:size]
test_sentence = dataset[size:]

train_label = label_dataset[:size]
test_label = label_dataset[size:]

len(train_sentence), len(test_sentence)

"""## 2. Load thư viện sử dụng BERT"""

pip install -q tf-models-official==2.4.0

import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks

"""## 3. Tải pre-trained model"""

gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
tf.io.gfile.listdir(gs_folder_bert)

hub_url_bert = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"

# Set up tokenizer to generate Tensorflow dataset
tokenizer = bert.tokenization.FullTokenizer(
    vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
     do_lower_case=True)

tokens = tokenizer.tokenize(dataset[2])
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])

train_size = 0.8
size = int(len(dataset) * train_size)

train_sentence = dataset[:size]
test_sentence = dataset[size:]

train_label = label_dataset[:size]
test_label = label_dataset[size:]

"""## 4. Chuẩn vị đầu vào cho Bert

Hàm encode_sentence có nhiệm vụ tokenize một câu và chuyển thành ids tương tự giống các tokenizer thông thường
"""

def encode_sentence(s, tokenizer):
   tokens = list(tokenizer.tokenize(s))
   tokens.append('[SEP]')
   return tokenizer.convert_tokens_to_ids(tokens)

"""Tuy nhiên đầu vào của Bert phức tạp hơn model thông thường khi là một mô hình được huấn luyện nhiều task khác nhau:

- Task 1: **Masked Language Model** - Che một số từ trong câu và cho Bert học để nhận ra từ bị thiếu
- Task 2:  **Next Sentence Prediction (NSP)** - Cho 2 câu, dự đoán xem có phải 2 câu này hay đi liền nhau hay không?


Pretrained đã được sử dụng để training 2 task này có thể sử dụng để fine tune cho bài toán nhỏ hơn của chúng ta chính là bài toán phân loại

Cấu trúc đầu vào:

- input_word_ids: ids của từng từ (tương tự như các bài phân loại thông thường)
- input_mask: masking của từ (che đi) để phục vụ **Task 1**
- input_type_ids: Câu trước tất cả giá trị là 0 câu sau tất cả là 1 phục vụ **Task 2**. Với bài phân loại thì chỉ có duy nhất câu phía trước là câu để phân loại và không có câu thứ 2

Chú ý: CLS Token là một vector đặt đầu câu sẽ học được attention của các từ trong câu với nó.

Với bài toán phân loại thì vector này sẽ được sử dụng để đưa vào mô hình phân loại
"""

def bert_encode(sentences, tokenizer):
  tokenized_sentences = tf.ragged.constant([
      encode_sentence(s, tokenizer)
      for s in sentences])

  # CLS TOken đứng đầu câu
  cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*tokenized_sentences.shape[0]
  
  input_word_ids = tf.concat([cls, tokenized_sentences], axis=-1)

  input_mask = tf.ones_like(input_word_ids).to_tensor()

  type_cls = tf.zeros_like(cls)
  
  type_s1 = tf.zeros_like(tokenized_sentences)
  
  input_type_ids = tf.concat(
      [type_cls, type_s1], axis=-1).to_tensor()

  inputs = {
      'input_word_ids': input_word_ids.to_tensor(),
      'input_mask': input_mask,
      'input_type_ids': input_type_ids}

  return inputs

"""Tokenize features tập train"""

train_inputs = bert_encode(train_sentence, tokenizer)
train_label_tensors = tf.constant(train_label)

"""Tokenize features tập test"""

test_inputs = bert_encode(test_sentence, tokenizer)
test_label_tensors = tf.constant(test_label)

"""## 5. Khởi động Bert

Tải cấu trúc của Bert.

Mô hình cơ bản Bert Base sẽ bao gồm 12 lớp Transformer Encoder

Mô hình Bert này đã được gắn thêm lớp phân loại cho nên ta không cần phải lập trình thêm
"""

import json

bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())

bert_config = bert.configs.BertConfig.from_dict(config_dict)

config_dict

"""Khởi tạo Bert"""

bert_classifier, bert_encoder = bert.bert_models.classifier_model(
    bert_config, num_labels=2)

"""Hiển thị mô hình"""

tf.keras.utils.plot_model(bert_classifier, show_shapes=True, dpi=48)

"""Feedforward thử dữ liệu"""

test_batch = {key: val[:10] for key, val in train_inputs.items()}

bert_classifier(
    test_batch, training=True
).numpy()

"""Khởi tạo checkpoint"""

checkpoint = tf.train.Checkpoint(encoder=bert_encoder)
checkpoint.read(
    os.path.join(gs_folder_bert, 'bert_model.ckpt')).assert_consumed()

"""## 6. Tiến hành training"""

# Set up epochs and steps
epochs = 3
batch_size = 32
eval_batch_size = 32

train_data_size = len(train_label)
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

# creates an optimizer with learning rate schedule
optimizer = nlp.optimization.create_optimizer(
    2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bert_classifier.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics)

bert_classifier.fit(
      train_inputs, train_label_tensors,
      batch_size=32,
      validation_data=(test_inputs, test_label_tensors),
      epochs=epochs)

"""Vậy là bạn đã fine-tune Bert trên bộ dữ liệu mỉa mai để đạt kết quả hơn 90% nhé ;)"""

