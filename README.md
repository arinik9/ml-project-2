ml-project-2
============

EPFL's Pattern Classification and Machine Learning second course project

Team members
------------

- Jade Copet
- Merlin Nimier-David
- Krishna Sapkota

The project was designed by Prof. Emtiyaz & TAs.

Project structure
-----------------


Project's TODO
--------------

### ML methods

- [ ] Generic k-fold Cross Validation
- [ ] Support Vector Machine
- [ ] Gaussian process and several kernels
- [ ] K-means clustering
- [ ] Gaussian Mixture Model and EM algo
- [ ] Principal Components Analysis (as a low-rank approximation) using alternating least squares
- [ ] Neural Networks (implementation from the [DeepLearn toolbox](https://github.com/rasmusbergpalm/DeepLearnToolbox/archive/))
- [ ] Generic learning curve plotting function
- [ ] Generic ML method comparison function (for each method, plot achieved test error & stability with a boxplot)

### Dataset pre-processing

- [ ] Basic data characteristics (dimensionality, repartition, correlation, ...)
- [ ] Try obtaining helpful visualizations
- [ ] Dimensionality reduction with PCA
- [ ] Try feature transformations (basis expansion) on the reduced set of features
 
### Person detection dataset

- [ ] Implement the relevant error measures
- [ ] Get a baseline error value
- [ ] Experiment with the Neural Network's hyperparameters (number of layers, activation functions)
- [ ] Experiment with clustering
- [ ] Experiment with Gaussian processes
- [ ] Generate learning curves
- [ ] Plot ROC Curves of different models for comparison
- [ ] Minimize the train and test error
- [ ] Check the stability of the results with k-CV

### Song recommendation dataset

- [ ] Get a baseline error value
- [ ] Experiment with SVM
- [ ] Experiment with Gaussian processes
- [ ] Experiment with clustering
- [ ] Generate learning curves
- [ ] Minimize the train and test error
- [ ] Check the stability of the results with k-CV

### Predictions

- [ ] `predictions_detection.csv`: Each row contains probability `p(y=1|data)` for a data example in the test set
- [ ] `predictions_recommendation.csv`: Each row contains predicted count for a data example in the test set
- [ ] `test_errors_detection.csv`: Report expected test error (which error measure?)
- [ ] `test_errors_recommendation.csv`: Report expected test error (which error measure?)

### Report

- [ ] Describe and discuss the methods used and show that we understand their inner working and the influence of each hyperparameter (especially for methods we did not implement ourselves)
- [ ] Produce figures for the detection dataset
- [ ] Report work done for the detection dataset and the corresponding results
- [ ] Produce figures for the recommendation dataset
- [ ] Report work done for the recommendation dataset and the corresponding results
- [ ] Double-check all figures for labels (on each axis and for the figure itself)
- [ ] Clear conclusion and analysis of the results for each dataset
- [ ] Include complete details about each algorithm (initialization values, lambda values, number of folds, number of trials, etc)
- [ ] What worked and what did not? Why do you think are the reasons behind that?
- [ ] Why did you choose the method that you chose?

Tools
-----

- [Piotr's toolbox](http://vision.ucsd.edu/~pdollar/toolbox/doc/)
- [DeepLearn toolbox](https://github.com/rasmusbergpalm/DeepLearnToolbox/archive/)
