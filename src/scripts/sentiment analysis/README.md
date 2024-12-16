## Sentiment Analysis Comparison and Implementation

This README explains how we evaluated and compared multiple sentiment analysis models and selected the most suitable one for our dataset. The models tested include:
* [BERT base multilingual uncased model](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
* [Google Cloud NLP](https://cloud.google.com/natural-language/docs/analyzing-sentiment?hl=fr)
* [GPT-4o mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) 
* [distilbert base multilingual cased model](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student)

### Initial Model Testing

We conducted initial tests using the following Jupyter notebooks:

* **distilbert_test.ipynb** is the first testing ground of the distilbert base model
* **test_google_sent.ipynb** is the first test for the Google Cloud NLP model

## Preparing the data

Since the dataset was too large to test all models on the full dataset, we reduced the sample to the first 6,000 reviews. This number was chosen to balance representativeness with cost-efficiency, especially for API-based models like Google Cloud NLP.

It was carried out in the second cell of **run_sentiment_tests.ipynb**.

## Model Comparison

The next step was to do test and compare every model (also in **run_sentiment_tests.ipynb**): in order to do so we harmonized the outputs of every model between the following : Negative, Neutral or Positive.

We then compared the models’ performances based on exact similarity and similarity within a range of ±1. The latter is particularly relevant because it allows us to identify models that may produce false positives or false negatives. We decided to plot the result in the bar plots and heat maps for the exact and ±1 similarity scores across different pairs.

Looking at the bar plots, the BERT and Distlilbert, the Google Cloud NLP and GPT-4o mini models, as well as the BERT anf Google Cloud NLP models show the highest similarity for exact sentiment, exceeding 65%.

Additionally, the Google Cloud NLP and GPT-4o mini models have nearly 100% similarity when considering a margin, followed closely by the BERT-based model and GPT-4o mini, and then by the DistilBERT model and GPT-4o mini.

Given that GPT-4o mini, the BERT model and Google Cloud NLP are very similar in the second graph, that GPT-4o mini is falling behind in the exact sentiment comparison and the DistilBERT model is behind the other models in both comparisons, we should use either Google Cloud NLP or the BERT-based model. Even though Google Cloud NLP has a slightly better similarity in both comparisons, it also an expensive api for large datasets. We will therefore use the BERT-based model for the final sentiment analysis.

## Applying the BERT-Based Model

The data was preprocessed in the **keep_only_reviews.ipynb** file. We dropped all reviews without comments and split the dataset in 15 batches that were run consecutively, to ensure progress was not lost in case of crashes.

We applied the chosen model in the **sentiment_local.ipynb** file with all the relevant functions being in the **sentiment_utils.py** file.
### Optimizations
* Batch size: 16 
* Workers: 12 
* GPU memory management:
  * Enabled torch.autocast and torch.no_grad to reduce memory usage. 
  * Used torch.mps.empty_cache to clear GPU memory after each batch. 
  * Device: "mps" for macOS devices. If not using macOS, update device="mps" in sentiment_utils.py.

### Hardware Details
The process was optimized for a MacBook Pro with:
* M4 Pro base chip
* 48GB RAM

The sentiment analysis took approximately 30 hours to complete while still having a usable computer.

### Final Step
The 15 resulting CSV files were merged to form the final dataset.

### File Summary
* distilbert_test.ipynb: Initial testing for DistilBERT.
* test_google_sent.ipynb: Initial testing for Google Cloud NLP.
* run_sentiment_tests.ipynb: Comparison of the different models.
* keep_only_reviews.ipynb: Filters and prepares the dataset.
* sentiment_local.ipynb: Applies the BERT-based sentiment analysis model.
* sentiment_utils.py: Contains helper functions for the final sentiment analysis.