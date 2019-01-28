Datathon link : https://www.datasciencesociety.net/datathon/

Requirements :
pandas, pytorch, https://github.com/OsmanMutlu/pytorch-pretrained-BERT (This is a fork of huggingface's original pytorch implementation of BERT), pretrained BERT model with its config and vocab file

For Task 1 :
Run gettask1data.py, which splits the train data into two
Then run bert_finetune.py, which fine-tunes the pretrained BERT model on our training data and saves the best performer model on val data.
Example run of bert_finetune.py :

python bert_finetune.py \
  --task_name hack \
  --do_train \
  --do_eval \
  --data_dir ~/HackTheNewsPropaganda/task-1 \
  --bert_model /path/to/pretrained_bert_model/ \
  --max_seq_length 256 \
  --train_batch_size 20 \
  --seed 43 \
  --learning_rate 2e-5 \
  --num_train_epochs 10.0 \
  --output_file ~/HackTheNewsPropaganda/task-1/model-256-43.pt

Finally run task1_predictor.py, which predicts test data (You need to change bert_model variable)


For Task 2 :
Run gettask2data.py with TRAIN = True, which combines the train data into a single csv and splits the data into two
Then run bert_finetune.py, which fine-tunes the pretrained BERT model on our training data and saves the best performer model on val data.
Example run of bert_finetune.py :

python bert_finetune.py \
  --task_name hack \
  --do_train \
  --do_eval \
  --data_dir ~/HackTheNewsPropaganda/task-2 \
  --bert_model /path/to/pretrained_bert_model/ \
  --max_seq_length 256 \
  --train_batch_size 20 \
  --seed 43 \
  --learning_rate 2e-5 \
  --num_train_epochs 10.0 \
  --output_file ~/HackTheNewsPropaganda/task-2/model-256-43.pt

Run gettask2data.py with TRAIN = False, which combines the test data into a single csv
Finally run task2_predictor.py, which predicts test data (You need to change bert_model variable)