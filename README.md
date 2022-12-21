{:toc}

# Polarity and Subjectivity Detection
This repository contains my personal solution to the final assignment of the Natural Language Understanding course about Sentiment Analysis.

## Project statement
The requirements for this project were to come up with an algorithm that allows to automatically perform sentiment and polarity classification.

## Datasets
The suggested datasets for carrying out this task are :
- the built-in NLTK movie reviews dataset (for polarity)
- the built-in NLTK subjectivity dataset (for subjectivity).
In addition to these two, I decided to use a third dataset (the [IMDB](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)) to fine-tune a [BertForSequenceClassification](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/bert#transformers.BertForSequenceClassification) model since the samples in the polarity dataset are pretty few. 

## Pipeline
This work focuses on the comparison between a baseline model and a custom model.
### Baseline
The baseline implementation can be found [here](https://github.com/MatteoGuglielmi-tech/Polarity-and-Subjectivity-Detection/blob/main/src/BaselineModel.ipynb) and it is used to have a reference polarity accuracy to surpass with the custom model. This shallow models exploit both [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) and [NaiveBayesClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB).

###Custom model
The custom model implementation can be found [here](https://github.com/MatteoGuglielmi-tech/Polarity-and-Subjectivity-Detection/blob/main/src/Matteo_Guglielmi_232088.ipynb). In addition, the training procedure for the BertForSequenceClassification model, which is part of the pipeline in [Matteo_Guglielmi_232088.ipynb](https://github.com/MatteoGuglielmi-tech/Polarity-and-Subjectivity-Detection/blob/main/src/Matteo_Guglielmi_232088.ipynb), can be found [here](https://github.com/MatteoGuglielmi-tech/Polarity-and-Subjectivity-Detection/blob/main/src/BinaryClassifier.ipynb).
The main notebook consits in:
- training a relative shallow BiLSTM network for subjectivity detection
- using the just trained model to filter out the objective sentences in the polarity dataset
- perform polarity classification, exploiting the previously trained [BinaryClassifier.ipynb](https://github.com/MatteoGuglielmi-tech/Polarity-and-Subjectivity-Detection/blob/main/src/BinaryClassifier.ipynb) and observing the results.

## Results
The baseline models reaches a polarity classification accuracy of $0.832$ ($83\%) while the custom model achieve is able to significantly improve the performance reaching  cumulative accuracy of $0.928$ ($92.83\%)
