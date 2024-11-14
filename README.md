# Title: The hidden aspects of controversial and universal beers

## Abstract

Some beers provoke strong and diverse opinions, while others receive more consistent feedback. This project aims to identify the factors driving such variance, focusing on “controversial” beers, those with high rating variability, and “universal” beers, which have more uniform reviews. Using both quantitative ratings and textual analysis, this study examines how attributes like aroma, appearance, palate, and taste influence perceptions. Are certain attributes linked to stronger divides in opinion? Additionally, we analyze sentiment within reviews, identifying keywords and phrases associated with these two classifications. Beyond the beers themselves, we explore regional patterns and compare opinions from novices and experts to understand who contributes most to the variance. By uncovering these insights, the study aims to explain why a beer may be widely appealing or divisive.

## Research questions

- What makes a beer controversial?
- What makes a beer “controversial” or “universal”? How can we define controversiality, and can we distinguish these two classes effectively?
- Which attributes, such as aroma, appearance, palate, and taste, are most associated with beer being perceived as controversial?
- Are certain regions more likely to produce controversial beers or to have distinct, less popular opinions? Are locally brewed beers rated more controversially compared to foreign ones?
- Do novice or expert reviewers contribute more to controversial ratings?
- Can we identify specific keywords or descriptions related to taste, smell, or texture that make beer more controversial or universal by analyzing textual reviews?

## Methods

### Data cleaning:
For our analysis, we decided to group each type of information into a single dataset. Specifically, we will combine the Advocate, Matched, and Ratebeer datasets into a single CSV file. When merging these datasets, we must carefully handle duplicates to ensure data accuracy. Furthermore, we decided to consider the matched ratings as two separate comments as they may have different grades and a different textual description. This should enable easier analysis for later on.

### Basic statistics analysis of the dataset :
To begin our analysis we aim to use the variance of the features as a metric of controversiality. The goal is to analyze the data from different angles, to find possible thresholds that could explain controversiality. We aim to give different definitions of controversiality and analyze their potential for further analysis, a justification for every choice made should be provided. We plan on using distribution, mean values, and PCA for a projection on 2 dimensions. We will use statistical tests to test different hypotheses, for example, the T statistic test should be used to compare the variance of different attributes and analyze whether certain attributes have a similarity in grading. A small comparison between the two datasets should be performed to see if and how they differ in the grading. Controversy requires a public disagreement. This means we need to consider only the beers with a certain amount of rating on the websites. One more reason to merge the dataset; we can get more opinions on each beer, thus selecting a higher number of beers for the analysis.

### Sentiment analysis of the textual description
By using multiple models, we aim to give a score that highlights the feeling given by the textual description. We are using various models such as a sentiment analysis model, an embedding model, an LDA model, and potentially a summarizer. One of the models is built by one of the members for his startup. We plan on comparing the results of the models to validate the usage of a model.

If we do not find a clear and meaningful threshold, we aim to propose one to pick the two extremities based on an attribute we choose: ratings, overall, or even the different attributes. For example, the threshold could be 10% of the data on both extremities. Then, we aim to understand what makes a comment and a rating controversial through the previous methods and the next one.

### Clustering methods
Previously we used the PCA in the hope of finding a visual threshold. In this part, we aim to use many dimensions provided by the dataset and use a clustering method. We need to evaluate different techniques such as KNN, and GMM. Furthermore, we aim to look at the keyword coming up in the controversial comments, not only but also analyze the keywords of a beer controversial for a given attribute.

## Timeline of the tasks

Week 1 -  Apply the sentiment analysis by using the model chosen in part 2.3 on the whole dataset. If necessary use a second model and keep the reviews with the same sentiment score. Find a threshold to set a beer as controversial or universal: for example the two extremes of 10%. Continue working on part 2, and for a final point use a clustering method, and visualize the results.
Week 2 - Perform the analysis on the keyword of the comments. Find a match between the attribute of the comment that makes it controversial and the keywords obtained. Perform analysis on the effect of the number of reviews, the origin of the user, the number of ratings of the beer, the style, and the alcohol by volume on the controversy. We want to see this after setting a threshold for setting a beer controversial or not. 
Week 3 - Finalize all the necessary visualizations. Perform analysis of week 2 that couldn’t be done in time. Begin to write the history.
Week 4 - Have a final data story, coherent with the findings we made. Begin the webpage. Add interactive visualizations if possible.
Week 5 - Finalize the webpage, clean the GitHub, and proofread the data story.

## Organization within the team

- Finalize the sentiment analysis, and begin clustering of keywords -> Gustave, Matteo
- Find a threshold to separate the dataset -> Jehan
- Perform a clustering method -> Valentin
- Perform analysis on the effect of the number of reviews, the origin of the user -> Alan
- Perform analysis on the effect of the number of ratings of the beer, the style, and the alcohol by volume -> Valentin, Jehan

## Questions for the TA

Since in part 2, we show that there are no easy thresholds, is it advised to continue to search for a potential delimitation? Is the analysis that we made sufficient to go to part 3? In this part, we set a threshold to separate the two extremes. We want to see how different parameters such as the expertise of the rater, and the country of origin are examined.
In part 2.3 we compare different models and we see that the scores are up to exactly the same for 69%, and up to 97% for +/- 1 score. As we want to measure the variance of the sentiment obtained, does it make sense to use a model? The variance may be meaningless as for 30% of the case the real value may be off.

