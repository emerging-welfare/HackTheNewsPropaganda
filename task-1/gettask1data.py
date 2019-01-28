import pandas as pd
from sklearn.model_selection import train_test_split

input_train_file = "../datasets-v5/task-1/task1.train.txt"

df = pd.read_csv(input_train_file, sep="\t", header=None, names=["text", "article_id", "label"])

# This splits the train dataset into two. Validation set is used when training to save the best performing model on it
y = df.pop("label")
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1, stratify=y) # Stratified sampling -> Important because data is unbalanced
X_train["label"] = y_train
X_test["label"] = y_test
X_train.to_csv("train.tsv", sep="\t", header=None, index=False)
X_test.to_csv("val.tsv", sep="\t", header=None, index=False)
