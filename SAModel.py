import pandas as pd
import numpy as np
import underthesea
import regex as re
from underthesea import word_tokenize
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import nltk.corpus
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
import warnings
warnings.filterwarnings("ignore")
import os
import pickle
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import time


from datasets import load_dataset, concatenate_datasets

# Load dataset
dataset = load_dataset("uitnlp/vietnamese_students_feedback")

# Truy cập vào phần dữ liệu train và test
train_data = dataset['train']
test_data = dataset['test']

# Concatenate the train and test datasets
combined_data = concatenate_datasets([train_data, test_data])
data = combined_data

# Chuyển dòng dữ liệu thành dòng chỉ chứa giá trị
value_rows = []
for item in data:
    value_row = list(item.values())
    value_rows.append(value_row)

neg, pos, neu = 0, 0, 0

# Đếm số lượng dòng có giá trị row[1] là 1, 0 hoặc 2
for row in value_rows:
    if row[1] == 0:
        neg += 1
    elif row[1] == 1:
        neu += 1
    elif row[1] == 2:
        pos += 1

# Thống kê các word xuất hiện ở tất cả các nhãn
total_label = 3
vocab = {}
label_vocab = {}
for row in value_rows:
    line = row[0]
    words = line.split()
    if row[1]== 0:
        label = 'neg'
    elif row[1] == 1:
        label = 'neu'
    elif row[1] == 2:
        label = 'pos'
    if label not in label_vocab:
        label_vocab[label] = {}
    for word in words:
        label_vocab[label][word] = label_vocab[label].get(word, 0) + 1
        if word not in vocab:
            vocab[word] = set()
        vocab[word].add(label)

count = {}
for word in vocab:
    if len(vocab[word]) == total_label:
        count[word] = min([label_vocab[x][word] for x in label_vocab])

sorted_count = sorted(count, key=count.get, reverse=True)

# Chia tập train/test
test_percent = 0.2

text = []
label = []
for row in value_rows:
    text.append(row[0])
    if row[1] == 0:
        label.append('neg')
    elif row[1] == 1:
        label.append('neu')
    elif row[1] == 2:
        label.append('pos')


# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=test_percent, random_state=42)


# Giữ nguyên train/test để về sau so sánh các mô hình cho công bằng

with open('train.txt', 'w', encoding="utf-8") as fp:
    for x, y in zip(X_train, y_train):
        fp.write('{} {}\n'.format(y, x))

with open('test.txt', 'w', encoding="utf-8") as fp:
    for x, y in zip(X_test, y_test):
        fp.write('{} {}\n'.format(y, x))

# encode label
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
print(list(label_encoder.classes_), '\n')
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)


MODEL_PATH = "models"
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Lưu label_encoder thành file .pkl
pickle.dump(label_encoder, open(os.path.join(MODEL_PATH, "label_encoder.pkl"), 'wb'))


# SVM
start_time = time.time()
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1),
                                             max_df=1.0, min_df=1,
                                             max_features=None)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SVC(gamma='scale'))
                    ])
text_clf = text_clf.fit(X_train, y_train)

train_time = time.time() - start_time
print('Done training SVM in', train_time, 'seconds.')

# Save model
pickle.dump(text_clf, open(os.path.join(MODEL_PATH, "svm.pkl"), 'wb'))

# SVM
model = pickle.load(open(os.path.join(MODEL_PATH,"svm.pkl"), 'rb'))

def Sentiment_analysis_one_text(text):
    label = model.predict([text])
    print('Predict label:', label_encoder.inverse_transform(label))

def main():

    text = input("Nhập văn bản: ")
    Sentiment_analysis_one_text(text)

if __name__ == "__main__":
    main()