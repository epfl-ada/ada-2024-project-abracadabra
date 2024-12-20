# The hidden aspects of controversial and universal beers

## Abstract

Some beers provoke strong and diverse opinions, while others receive more consistent feedback. This project aims to identify the factors driving such variance, focusing on “controversial” beers, those with high rating variability, and “universal” beers with more uniform reviews. This study uses quantitative ratings and textual analysis to examine how attributes like aroma, appearance, palate, and taste impact the controversiality. Additionally, we evaluate the effect of analyzing sentiments within reviews and the addition of data it provides. Furthermore, we explore beer parameters to evaluate their link with the variation of the reviews. Beyond the beers, we explore regional patterns and compare the user's number of ratings done, to understand who contributes most to each class. By uncovering these insights, the study aims to explain why a beer may be widely appealing or divisive.

## Research questions

- How can we define and classify beer controversiality, and which numerical attributes best predict it? Can we use the textual review to enhance our classification?
- Are there features explaining controversiality, and do they depend on the beer itself?


## Methods

### Data cleaning:

For our analysis, we decided to group each dataset into a single one for every type of information. When merging these datasets, we must carefully handle duplicates to ensure data accuracy. Additionally, we chose to treat matched ratings as separate entries, as they may contain different grades and textual descriptions. This approach should simplify analysis in later stages.

### Basic statistics analysis of the dataset :

To begin our analysis we will use the variance of the features as a metric of controversiality. We analyzed the data from different angles, to identify potential factors that might define controversiality. We gave different definitions of controversiality and analyzed their potential for further analysis, a justification for every choice made is provided. We use statistical tests to try different hypotheses, such as the T-test. Controversy requires a public disagreement. This means we need to consider only the beers with a minimum amount of ratings.

### Sentiment analysis of the textual description

The sentiment analysis’ goal is to give a score that highlights the feeling in a review. We aim to use this score to enhance our definition of controversiality. We compare different models and select one based on the best similarity score.

### Clustering methods

To classify the beers we applied a clustering method on the variance of the ratings. Using the elbow method on a Gaussian Mixture Model, we can classify the different beers and pursue our analysis on the obtained label. 


## Organization

- Alan: Writing the data story, validation of the data story and final notebook, and also came up with the beer icon idea on the website.
- Gustave: Sentiment analysis on the textual reviews, evaluation of the keyword analysis potential, validation of final notebook
- Jehan: Data crawling, plotting graphs, performing various experiments, preliminary data analysis, writing the final notebook
- Mattéo: webmaster, the main writer of the datastory and validation of the final notebook, data merging 
- Valentin: writing up the final analysis and the data story, plotting graphs, crawling the data and performing various experiments, preliminary data analysis, data cleaning and merging
