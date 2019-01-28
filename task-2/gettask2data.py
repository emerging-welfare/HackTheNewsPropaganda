import pandas as pd
from glob import glob
import re

TRAIN = False # Whether this is the train folder or not
if TRAIN:
    from sklearn.model_selection import train_test_split

path_to_folder = "../test-INPUT/tasks-2-3/test/"
#path_to_folder = "../datasets-v5/tasks-2-3/train/"
all_file_name = "all_test.tsv"
out_filename = "to_be_predicted_test.tsv"

if TRAIN:
    df = pd.DataFrame(columns=["article_sent", "article_id", "label"])
else:
    df = pd.DataFrame(columns=["article_sent", "article_id", "sent_id"])

for filename in glob(path_to_folder + "*.txt"):
    article_id = re.sub(r".*article(\d+)\.txt$", r"\g<1>", filename)
    article_sents = []
    sent_ids = []
    with open(filename, "r", encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()):
            article_sents.append(line.rstrip())
            sent_ids.append(i+1)

    if TRAIN:
        gold_labels = []
        with open(re.sub(r"(\d+)\.txt$", r"\g<1>.task2.labels", filename), "r", encoding="utf-8") as g:
            for line in g.readlines():
                _, _, gold_label = line.rstrip().split("\t")
                gold_labels.append(gold_label)

        df = df.append([{"article_id": article_id, "article_sent": article_sents[j], "label": gold_labels[j]} for j in range(len(sent_ids))], ignore_index=True)
    else:
        df = df.append([{"article_id": article_id, "article_sent": article_sents[j], "sent_id": sent_ids[j]} for j in range(len(sent_ids))], ignore_index=True)

# This splits the train dataset into two. Validation set is used when training to save the best performing model on it
if TRAIN:
    y = df.pop("label")
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1, stratify=y) # Stratified sampling -> Important because data is unbalanced
    X_train["label"] = y_train
    X_test["label"] = y_test
    X_train.to_csv("train.tsv", sep="\t", header=None, index=False)
    X_test.to_csv("val.tsv", sep="\t", header=None, index=False)

else:
    # This is used after we classify the test data. If there is a sentence that is not classified for some reason, we make it non-propaganda manually.
    df.to_csv(all_file_name, sep="\t", header=None, index=False)

    df = df[df.article_sent != ""]
    df.to_csv(out_filename, sep="\t", header=None, index=False)
