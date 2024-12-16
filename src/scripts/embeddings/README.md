## Embedding Analysis: Comparison and potential

This README explains how we evaluated and compared multiple embedding models and selected the most suitable one for our dataset. The models tested include:
* [Multilingual-E5-Large](https://huggingface.co/intfloat/multilingual-e5-large)
* [Multilingual-E5-base](https://huggingface.co/intfloat/multilingual-e5-base)
* [Text-Embedding-3-Large (OpenAI)](https://platform.openai.com/docs/guides/embeddings/) 
* [BGE-M3](https://huggingface.co/BAAI/bge-m3)

## Goal
The primary goal of this evaluation is to identify relevant themes in qualitative reviews, such as flavor, color, and others. Insights derived from this analysis, combined with sentiment analysis and LDA modeling, will help find out why certain characteristics may be controversial. In order to do so we need an accurate embedding model. 

To properly assess the accuracy of such model we will need a labeled dataset. We will use the [amazon_reviews_multi](https://huggingface.co/datasets/defunct-datasets/amazon_reviews_multi) dataset. This dataset is multilingual, with languages from different origins such as English (Germanic), French (Romance), Japanese (Japonic), Chinese (Sino-Tibetan), etc.. This dataset is ideal because embedding models often face challenges with such linguistic diversity.

The dataset columns of interest are:

* review_body: Amazon qualitative product review.
* language: Language of the review.
* product_category: Used as the label column for classification.
## Preparing the data

We first prepared the data in the **prepare_data.ipynb** file to understand the structure and content of the dataframe, ensuring it is ready for testing and embedding.

## Model Comparison

The embedding models selected for evaluation include:

* Multilingual-E5-Large and Multilingual-E5-Base: RoBERTa-based models developed by Microsoft.
* BGE-M3: A model developed by the Beijing Academy of Artificial Intelligence.
* Text-Embedding-3-Large: OpenAI's latest embedding model.

### Similarity Metrics

Cosine similarity was chosen for comparing sentence embeddings because:

* It measures the similarity between normalized vectors.
* It captures the relationships between vector components, unlike Jaccard similarity.
* It normalizes for vector magnitudes, unlike Euclidean similarity.

### Sample Evaluation:
In the first part of **run-embedding_tests.ipynb** we tested each model with some basic sentences to check if they were working properly and have an idea of their outputs. This will help us set the threshold between relevant and non-relevant comments. 

We then did the following: 

* Collected a sample of 1,000 reviews per language.
* Embedded the reviews and their corresponding labels using each model.
* Computed cosine similarity between each review and each label for each model.
* Retained the label with the highest similarity score.
* Calculated the accuracy for each model after excluding reviews that were too vague to be categorized by any model.

In doing so we found out that the most accurate model by a large margin was OpenAI. However, as OpenAI is not free, we recommend the fastest model between the two next best performing model: the **Multilingual-E5-Base** model.

## Potential Applications

With the embedding pipeline in place, there are two main approaches for deriving insights:

1. Thematic Analysis
   * Compare each review against a predefined list of themes (e.g., taste, color, etc.).
   * Generate a list of comments for each theme, that we can summarize providing deeper insights into each theme.

2. Clustering Analysis
   * Use a clustering algorithm to group reviews based on similarity.
   * Assign labels to clusters using:
     * First a summarization model or an LDA
     * Then a large language model (e.g., Llama 2) with a fine-tuned prompt like "Give a label to the following text: [text] in 1 or 2 words".
     
Once labeled, these groups can be analyzed further:

* Identify controversial groups by examining sentiment analysis variance.
* Go deeper into pre-identified controversial themes by applying summarizers to relevant comment clusters.

### File Summary
* prepare_data.ipynb: Data exploration
* test_clustering.ipynb: Testing the concept of clustering
* run_embedding_test.ipynb: Embedding model testing and evaluation