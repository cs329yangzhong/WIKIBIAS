CUDA_VISIBLE_DEVICES=1 python run_ner_extra_feature_fine_grained.py --data_dir=../../data/tag_finegrained/ --bert_model=bert-base-cased --task_name=joint --output_dir=out_tag_cased_extra_feature_fine_grained_large_seed1000_lr_2e-5_epoch3 --max_seq_length=128 --num_train_epochs 3 --do_eval --eval_on dev --warmup_proportion=0.1 --do_train --seed 1000 --learning_rate 2e-5 --do_joint True --loss_weight 0_1_0
