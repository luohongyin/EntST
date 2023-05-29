# Entailment as Robust Self-learner
The repo of paper [Entailment as Robust Self-learner](#) by Jiaxin Ge*, Hongyin Luo*, Yoon Kim, Jim Glass at ACL 2023 main conference.

# Dependencies
- Transformers >= 4.16.2
- Pytorch >= 1.6.0

# Preparing data
```
bash prep.sh
```

# Reproducing GLUE Experiments
```
bash prompt_script.sh
```
Parameters in `prompt_script.sh`
- `domain`: qnli, qqp, rte, sst2
- `ft_mode`: st (self-training)
- `exp_id`: an `int` number flaging the experiment ID.
- `train_size`: The number of a unit training set. The number of training case is `train_size x exp_id`.
- `model_type_str`: Either `roberta` or `deberta`

The `prompt_script.sh` file describes the entire process, and it runs multiple independent experiments. We also added `Slurm` flags so it can be submitted to slurm as a single job or a job array.

# Reproducing Multi-class Experiments
```
cd multi-class

Train:
python3 ag_news.py -—algo MODEL_NAME -—index 0 --type ST_METHOD
python3 amazon_news.py -—algo MODEL_NAME -—index 0 --type ST_METHOD
python3 emotion.py —-algo MODEL_NAME —-index 0 --type ST_METHOD
python3 copa.py —-algo MODEL_NAME —-index 0 --type ST_METHOD

Test:
python3 test_agnews.py —-algo MODEL_NAME —-index 0
python3 test_amazon.py —-algo MODEL_NAME —-index 0
python3 test_copa.py —-algo MODEL_NAME —-index 0
python3 test_emotion.py —-algo MODEL_NAME —-index 0

—algo: which entailment model backbone to use [“deberta”, “roberta”]
—index : appendix in the model path 
—type: finetune algorithm [
    "pseudo",       # Baseline self training
    "confidence",   # Removing low-confidence cases
    'dropout'       # Simple dropout-based voting
    'unconf',       # Full SimPLE algorithm
]
```

# Files included
- `pretrain_enty.py`: Entailment pretraining on MNLI
- `proc_data.py`: Preprocessing the training / eval corpora and constructing the suppositions.
- `prompt_cst.py`: Running one training pipeline.
- `train_glue.py`: Contains the training functions.
- `eval_prompt.py`: Evaluate the pretrained, fine-tuned, and self-trained models.
- `prompt_script.sh`: Running three independent training-evaluation pipelines.