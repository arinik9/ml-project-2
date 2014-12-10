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
- [X] Gaussian process and several kernels
- [ ] K-means clustering
- [ ] Gaussian Mixture Model and EM algo
- [x] Principal Components Analysis (as a low-rank approximation) using alternating least squares
- [X] Neural Networks (implementation from the [DeepLearn toolbox](https://github.com/rasmusbergpalm/DeepLearnToolbox/archive/))
- [ ] Generic learning curve plotting function
- [ ] Generic ML method comparison function (for each method, plot achieved test error & stability with a boxplot)

### Dataset pre-processing

- [x] Basic data characteristics (dimensionality, repartition, correlation, ...)
- [ ] Try obtaining helpful visualizations
- [x] Dimensionality reduction with PCA
- [ ] Try feature transformations (basis expansion) on the reduced set of features

### Person detection dataset

- [X] Implement the relevant error measures
- [ ] Compare precision/recall vs ROC
- [ ] Implement Mutual Information and plot against classification threshold theta
- [X] Get a baseline error value
- [X] Implement Logistic Regression
- [X] Experiment with the Neural Network's hyperparameters (number of layers, activation functions, dropout...)
- [ ] Experiment with clustering
- [X] Experiment with Gaussian Processes (check provided toolbox)
- [ ] Experiment with SVM (check provided toolbox)
- [ ] Generate learning curves
- [ ] Plot ROC Curves of different models for comparison
- [ ] Experiment with the threshold once a model decided
- [ ] Minimize the train and test error
- [ ] Check the stability of the results with k-CV

### Song recommendation dataset

Recall we must achieve both **weak** (new ratings for existing users) and **strong** (entirely new users) prediction.

- [x] Implement the relevant error measures
- [x] Implement error diagnostics (which kind of counts do we make the most error on?)
- [x] Train / test split (particular for weak and strong prediction)
- [x] Feature engineering: implement derived variables
- [x] Get a baseline error value
- [ ] Implement the simple [Slope One](http://arxiv.org/pdf/cs/0702144v2.pdf) predictor
- [ ] Experiment with SVM
- [ ] Experiment with Gaussian processes
- [ ] Experiment with clustering
- [ ] Determine if using the social graph helps weak prediction (then, we will be able to know if we can use it for strong prediction as well)
- [ ] Use the social network for strong prediction (clustering ?)
- [ ] Generate learning curves
- [ ] Minimize the train and test error
- [ ] Check the stability of the results with k-CV
- [ ] Use the artists name to output fun facts

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
- [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/code/matlab/doc/)
