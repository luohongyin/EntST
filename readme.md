# Entailment as Robust Self-learner

# Dependencies
- Transformers 4.16.2
- Pytorch 1.6.0

# Runing the pipieline
```
./prompt_script.sh $DOMAIN $EVAL_MODEL $FT_MODE $EXP_ID
```
- Domain: qnli, qqp, rte, sst2, cola
- FT_mode: ft (fine-tuning) or st (self-training)

# Files included
- `pretrain_enty.py`: Entailment pretraining on MNLI
- `proc_data.py`: Preprocessing the training / eval corpora and constructing the suppositions.
- `prompt_cst.py`: Running one training pipeline.
- `train_glue.py`: Contains the training functions.
- `eval_prompt.py`: Evaluate the pretrained, fine-tuned, and self-trained models.
- `prompt_script.sh`: Running three independent training-evaluation pipelines.