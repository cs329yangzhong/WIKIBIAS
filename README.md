# WIKIBIAS: Detecting Multi-Span Subjective Biases in Language

This repo contains codes for the following paper:
```
@inproceedings{zhong-etal-2021-wikibias-detecting,
    title = "{WIKIBIAS}: Detecting Multi-Span Subjective Biases in Language",
    author = "Zhong, Yang  and
      Yang, Jingfeng  and
      Xu, Wei  and
      Yang, Diyi",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.155",
    pages = "1799--1814",
    abstract = "Biases continue to be prevalent in modern text and media, especially subjective bias {--} a special type of bias that introduces improper attitudes or presents a statement with the presupposition of truth. To tackle the problem of detecting and further mitigating subjective bias, we introduce a manually annotated parallel corpus WIKIBIAS with more than 4,000 sentence pairs from Wikipedia edits. This corpus contains annotations towards both sentence-level bias types and token-level biased segments. We present systematic analyses of our dataset and results achieved by a set of state-of-the-art baselines in terms of three tasks: bias classification, tagging biased segments, and neutralizing biased text. We find that current models still struggle with detecting multi-span biases despite their reasonable performances, suggesting that our dataset can serve as a useful research benchmark. We also demonstrate that models trained on our dataset can generalize well to multiple domains such as news and political speeches.",
}
```
If you would like to refer to it, please cite the paper mentioned above.

## Prerequisites
- python==3.6.8
- torch=1.5.0
- scikit-learn==0.23.1
- transformers==2.8.0
- pytorch-transformers=1.2.0
- nltk=3.4.5

 If you are using the conda enviroment, you can install all dependencies with the provided ```requirments.txt``` file.

## Data

### Train/Dev/Test Data
Please download all ourd data from [here](https://drive.google.com/drive/folders/1dLKpaVktAojeQ7so1_Seao_e3iLQ-Egy?usp=sharing) and put under ```data```. There should be four folders:

 - ```class_binary```
 - ```class_finegrained```
 - ```tag_binary``` 
 - ```tag_finegrained```

Each folder contains a train, dev, and test split.  We provide the detailed data format in the [README file](data/README.md) under ```data```.

### Data Annotation
We release the data annotation interfaces and instructions under ```data/annotations```.

## Training Models

### Classification

Please go to ```./src/Classification```. 

The following command fine-tunes an bert-base sentence classification model for binary bias detection.
```
python train.py --device [desired cuda id] 
                --batch-size 16 
                --epochs 3 
                --max_len 128 
                --lr 2e-5
                --pretrained-ckpt "ckpt/*saved_ckpt*" [optional]
```

The following command fine-tunes an bert-base sentence classification model for finegrained bias detection.
```
python train_finegrained.py --device 0 [desired cuda device] 
                --batch-size 16 
                --epochs 3 
                --max_len 128 
                --lr 2e-5
                --pretrained-ckpt "ckpt/*saved_ckpt*" [optional]
```

To evaluate the results, run
```
python eval.py/eval_finegrained.py --device 0 [desired cuda device] --eval_ckpt "ckpt/*saved_ckpt*"
```

### Tagging 
Please first change directory to ```./src/Tagging```. 

**Training BERT model**

Please run ```./run_bert_ner.sh``` to train the BERT baseline model without extra linguistic features.

**Training BERT-LING model**

Please run ```./run_bert_ner_extra_feature.sh``` to train the BERT baseline model with extra linguistic features.

**Training BERT-LING model for fine-grained settings**

Please run ```./run_bert_ner_extra_feature_fine_grained.sh``` to train the BERT baseline model with extra linguistic features in the fine-grained setting.

**Evaluating models**

Please follow the example jupyter notebook (```eval.ipynb```) for evaluating the model on test set for Exact and Partial F1 scores.



