# Dualpath Reimplementation
This is the Pytorch implementation of dualpath embedding model for image-text matching for Flickr30k dataset. You will need the GoogleNews word2vec model ([Dowload](https://drive.google.com/a/mail.dcu.ie/uc?id=1lX6iq6_TfngYZKUhJoppEWhqzkS30Dhc&export=download)) and Flickr30k Dataset ([Download](https://drive.google.com/a/mail.dcu.ie/uc?id=12KSjtMLt5gL23aNlqZLigf6jYkjo3Svt&export=download)) to run.

Change parameter in ***config.py*** file, then run ***run_train.py*** file.

### Requirements
- tqdm
- nltk
- pytorch
- PIL
- gensim

### Download files from Drive with gdown
To download file from Drive, you can use gdown (install by ```pip install gdown```)

Syntax: ```gdown https://drive.google.com/uc?id=[FILEID]```

#### FILEID
- word2vec id: 1lX6iq6_TfngYZKUhJoppEWhqzkS30Dhc 
- flickr30k id: 12KSjtMLt5gL23aNlqZLigf6jYkjo3Svt

You can find the original paper [here](https://arxiv.org/abs/1711.05535)
