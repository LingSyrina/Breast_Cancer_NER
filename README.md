# Breast_Cancer_NER

0. pubmed.py gets queries paper from pubmed with IU id. 
1. NER.py reads in dictionary/database from corpus and perform Re-based data processing
2. model.py takes in preprocessed data from NER and trains a transformer model.
3. label_mapping.json is auto-generated each time by model.py for model prediction translation.
4. NER.bash and train.bash are for bigred200 computing
