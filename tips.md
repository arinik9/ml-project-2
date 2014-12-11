Tips & Remarks
==============

**Organization**:

- The marking distribution over the parts of the project is unknown
- The test set will be made available during the last week. It may be randomized.

**Tips**:

- Always keep the grading scheme in mind. Draw inspiration from the report examples.
- If you apply feature transforms to the training set, always remember to apply them to the test set as well.
- Recommended plots:
  - Compare results (RMSE, etc) with a boxplot for each method used. Shows clearly the progress brought about by different tricks
  - Learning curves (especially for penalized regressions — it legitimates the choice of range for the value of the penalization term)

Person detection
----------------

**Goal**: classify images based on whether it contains a person or not.

This is a difficult problem, but the features are given. We are expected to train a model from those features. Still, feature engineering (feature transforms, feature selection, outlier removal, etc) can still be useful. We just do not need to extract new features from the images.

Feature extraction which was performed: sliding windows over the image (different scales of windows as well). We are given prepared positive and negative examples.

The example code gives a look at the dataset and how the features relate.

We need to use an appropriate error measure: ROC. Indeed, the data examples are *very* skewed: a vast majority of the "sliding windows" are negatives, and only a few are positive. Simply computing the misclassification error, even for trivial constant-valued classifier would give a good looking number, which only reflects the distribution of inputs.
ROC takes into account both true positives and false positives.

Music recommendation
--------------------

**Goal**: from a listening count (discrete values) by user artist and a social graph (binary matrix of users), predict:

- Recommend new artists to existing users (so we have an idea of their test)
- Recommend artists to new users (we have no idea of their tastes, so base the prediction on their social graph)

Issues in the dataset:

- Users will have very large counts because they used the website a lot, but others are just light users. We must be able to adapt to that.
- The listening count data is ambiguous: a count of 0 could either mean an absence of value (the user just haven't come across the artist) *or* that the user dislikes this genre.

Generate a test set from training data by:

- Removing the last columns (make them unknown users)
- On the remaining data, remove some entries (simulating unknown listening counts)

Applicable techniques: kNN, PCA, mixture models. But we need to adapt those techniques to interpret the 0 values correctly.

It's mostly about the *collaborative* nature of the data. Weak prediction for a given user should leverage the users which are similar, and artists who are similar to the one he's listening to.

If you have a nice Gaussian distribution, it's not necessary to normalize it. You can precompute the global mean, and use it only to shift the data back and forth. Making it only at time of computation will reduce headaches with nonzero indices with sparse matrix.

**References**:

- Netflix challenge
