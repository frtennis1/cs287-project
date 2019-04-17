export GLUE_DIR=repo/glue_data
export TASK_NAME=SST-2
export OUTPUT_DIR=repo/output/

python repo/run_classifier.py \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir $OUTPUT_DIR\
  --report_frequency 100\
  --run_name "BERT_testing"\
  --tensorboard_log_dir "repo/tensorboard_data"
