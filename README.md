# The hidden aspects of controversial and universal beers

## Abstract

Some beers provoke strong and diverse opinions, while others receive more consistent feedback. This project aims to identify the factors driving such variance, focusing on “controversial” beers, those with high rating variability, and “universal” beers with more uniform reviews. Using quantitative ratings and textual analysis, this study examines how attributes like aroma, appearance, palate, and taste influence perceptions. Are certain attributes linked to stronger divides in opinion? Additionally, we analyze sentiment within reviews, identifying keywords and phrases associated with these two classifications. Beyond the beers, we explore regional patterns and compare opinions from novices and experts to understand who contributes most to the variance. By uncovering these insights, the study aims to explain why a beer may be widely appealing or divisive.

## Research questions

- What makes a beer “controversial” or “universal”? How can we define controversiality, and can we distinguish these two classes effectively?
- Which attributes, such as aroma, appearance, palate, and taste, are most associated with a beer being perceived as controversial?
- Are certain regions more likely to produce controversial beers or to have distinct, less popular opinions? Are locally brewed beers rated more controversially compared to foreign ones?
- Do novice or expert reviewers contribute more to controversial ratings?
- Can we identify specific keywords related to taste, smell, or texture that make a beer more controversial or universal by analyzing textual reviews?


## Methods

### Data cleaning:

For our analysis, we decided to group each dataset into a single one for every type of information. When merging these datasets, we must carefully handle duplicates to ensure data accuracy. Additionally, we chose to treat matched ratings as separate entries, as they may contain different grades and textual descriptions. This approach should simplify analysis in later stages.

### Basic statistics analysis of the dataset :

To begin our analysis we will use the variance of the features as a metric of controversiality. We plan to analyze the data from different angles, to identify potential factors that might define controversiality. We will give different definitions of controversiality and analyze their potential for further analysis, a justification for every choice made should be provided. We plan on using distributions, mean values, PCA for a projection on 2 dimensions. We will use statistical tests to test different hypotheses, such as the T-test. A comparison between the two datasets should be performed to see if and how they differ in the grading. Controversy requires a public disagreement. This means we need to consider only the beers with a minimum amount of ratings.

### Sentiment analysis of the textual description

The sentiment analysis’ goal is to give a score that highlights the feeling in a review. If we find that using a single model is unreliable, we can use multiple models and keep only the sentiment they agree on.  

If we do not find a clear and meaningful classifier, we propose to pick the two extremities based on an attribute we choose: overall, or the different attributes. We would then explain the controversiality or not, based on the reviewer, the country of origin, and other sources of explanation.

### Clustering methods

Previously we used the PCA in hope of finding a visual threshold. Here we are going to use the dimensions provided by the dataset and use a clustering method. We also will analyze keywords that frequently appear in controversial comments, focusing on general keywords and on those related to specific beer attributes. To achieve this, we will first use an embedding model to vectorize all the reviews. We can either use a KNN algorithm to identify themes directly from the reviews or embed predefined themes and use cosine similarity to associate comments with specific themes. Given the dataset’s size, we will need to apply an LDA model or a summarization tool to review the selected comments.

## Timeline

Week 1 -  Apply the sentiment analysis by using the model chosen in part 2.3 on the entire dataset. If necessary use a second model and keep the reviews with the same sentiment score / Define a reasonable classifier to label beers as controversial or universal / Use a clustering method, visualize the results.

Week 2 - Find an embedding and LDA model and prove their reliability. Perform the analysis on the keywords of the comments. Find a match between the attribute of the comment that makes it controversial and the keywords obtained. Perform analysis on the effect of the number of reviews, the origin of the user, the number of ratings of the beer, the style, and the alcohol by volume on the controversy.

Week 3 - Finalize all the necessary visualizations. Perform analysis of week 2 that couldn’t be done in time. Begin to write the story.

Week 4 - Have a final data story, coherent with the findings we made. Begin the webpage. Add interactive visualizations if possible.

Week 5 - Finalize the webpage, clean the GitHub, proofread the data story.


## Organization

- Finalize the sentiment analysis, begin clustering of keywords -> Gustave, Matteo
- Terminate analysis to define a classifier to label the beers -> Jehan
- Perform a clustering method -> Valentin
- Perform analysis on the effect of the number of reviews, the origin of the user -> Alan
- Perform analysis on the effect of the amount of ratings, the style, and the alcohol by volume -> Valentin, Jehan


## Questions

In part 2, we show that there are no evident labelling, is it advised to continue to search for a potential classifier (good definition of controversiality) ?
In part 2.3 we compare different models and we see that the exact similarity value is at maximum 69%, and 97% for +/- 1 score. As we want to measure the variance of the sentiment obtained, does it make sense to use a model? The variance may be meaningless as for 30% of the case the real value may be off.
