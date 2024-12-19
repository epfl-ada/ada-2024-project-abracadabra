# The hidden aspects of controversial and universal beers

## Abstract

Some beers provoke strong and diverse opinions, while others receive more consistent feedback. This project aims to identify the factors driving such variance, focusing on “controversial” beers, those with high rating variability, and “universal” beers with more uniform reviews. Using quantitative ratings and textual analysis, this study examines how attributes like aroma, appearance, palate, and taste influence perceptions. Are certain attributes linked to stronger divides in opinion? Additionally, we evaluate the effect of analyzing sentiments within reviews and the addition of data it provides. Furthermore, we explore beer parameters to evaluate their link with the variation of the reviews. Beyond the beers, we explore regional patterns and compare the user's number of ratings done, to understand who contributes most to the variance. By uncovering these insights, the study aims to explain why a beer may be widely appealing or divisive.

## Research questions

- How can we define controversiality for beer reviews and classify the beers based on this definition? Which numerical graded attributes are most associated with beers perceived as controversial? Can we use the textual review to enhance our classification?
- Are there features explaining this classification? Are the features dependent on the beer itself?


## Methods

### Data cleaning:

For our analysis, we decided to group each dataset into a single one for every type of information. When merging these datasets, we must carefully handle duplicates to ensure data accuracy. Additionally, we chose to treat matched ratings as separate entries, as they may contain different grades and textual descriptions. This approach should simplify analysis in later stages.

### Basic statistics analysis of the dataset :

To begin our analysis we will use the variance of the features as a metric of controversiality. We analyzed the data from different angles, to identify potential factors that might define controversiality. We gave different definitions of controversiality and analyze their potential for further analysis, a justification for every choice made is provided. We use statistical tests to try different hypotheses, such as the T-test. Controversy requires a public disagreement. This means we need to consider only the beers with a minimum amount of ratings.

### Sentiment analysis of the textual description

The sentiment analysis’ goal is to give a score that highlights the feeling in a review. If we find that using a single model is unreliable, we can use multiple models and keep only the sentiment they agree on.  

If we do not find a clear and meaningful classifier, we propose to pick the two extremities based on an attribute we choose: overall, or the different attributes. We would then explain the controversiality or not, based on the reviewer, the country of origin, and other sources of explanation.

### Clustering methods

To define the beers of controversiality we applied a clustering method on the variance of the ratings. Using the elbow method on a Gaussian Mixture Model, we can classify the different beers and pursue our analysis on the obtained label. 


## Organization

- Alan: Plotting graphs during data analysis, crawling the data, preliminary data analysis
- Gustave: Problem formulation, coming up with the algorithm
- Jehan: Coding up the algorithm, running tests, tabulating final results
- Mattéo: Writing up the report or the data story, preparing the final presentation
- Valentin: writing up the final analysis and the data story, plotting graphs during data analysis, crawling the data and performing various experiments, preliminary data analysis, data cleaning
