connect:
	ssh -L 6006:localhost:6006 -L 8080:localhost:8888 -L 8081:localhost:8889 Megatron@40.114.38.172 -p 22

train:
	export GLUE_DIR=glue_data
	export TASK_NAME=SST-2
	export OUTPUT_DIR=output/

	python run_classifier.py \
	  --task_name $TASK_NAME \
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
	  --tensorboard_log_dir "tensorboard_data"
			
test:
	export GLUE_DIR=glue_data
	export TASK_NAME=SST-2
	export OUTPUT_DIR=output/

	python run_classifier.py \
	  --task_name $TASK_NAME \
	  --do_eval \
	  --do_lower_case \
	  --data_dir $GLUE_DIR/$TASK_NAME \
	  --bert_model bert-base-uncased \
	  --max_seq_length 128 \
	  --output_dir $OUTPUT_DIR\
	  --report_frequency 100\
	  --run_name "BERT_testing"\
	  --tensorboard_log_dir "tensorboard_data" \
	  --config_file output/bert_config.json \
	  --model_file output/compressed_model.pt \
			


