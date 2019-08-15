# MovieLens-Recommendation-System

This is the repository for the Recommender challenge by Siraj Raval on [youtube](https://www.youtube.com/watch?v=9gBC9R-msAk&list=PL2-dafEMk2A6QKz1mrk1uIGfHkC1zZ6UU&index=3)

## Description
The program uses the MovieLens 1 million dataset by [GroupLens](https://grouplens.org/). It first downloads the .zip file (5.6mb) locally from my repository. The archive contains 3 files (the rating file has 1000209 entries) with values separated by `::` and a readme file with information about the dataset.

All the files are then loaded into pandas dataframes then parsed. The training data is a sparse matrix of type COO.

## Dependencies
-numpy
-sklearn (for one-hot encoding genres)
-lightFM
-scipy
