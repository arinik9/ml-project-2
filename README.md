ml-project-2
============

EPFL's Pattern Classification and Machine Learning second course project

Team members
------------

- Jade Copet
- Merlin Nimier-David

The project was designed by Prof. Emtiyaz & TAs.

Project structure
-----------------

This project contained two tasks: people detection in images, and a song recommender system from listening counts data.

- `analysis`: simple data exploratory analysis scripts we used to get to know the datasets better.
- `src`:
  - `detection`: code for the people detection dataset. We experimented with Gaussian Processes, Neural Networks, PCA, SVM and Random Forests
  - `recommendation`: code for the song recommendation dataset. We experimented with various feature extractions, ALS-WR, linear regression, K-means clustering, Gaussian Mixture Model clustering, Top-N recommendation and the Pearson similarity measure.
- `toolbox`: place the dependencies there. Our code relies on the DeepLearn toolbox, Piotr toolbox, and the VBGM script (see Tools section).
- `report`: project report (written in LaTeX). Contains references to related papers which were helpful for the project.
- `data` and `results`: input and output data (provided as Matlab `.mat` files).
- `test`: simple test scripts which were provided for us to check the output format of our predictions.


Project's TODO
--------------

### ML methods

- [X] Generic k-fold Cross Validation
- [X] Support Vector Machine
- [X] Gaussian process and several kernels
- [X] K-means clustering
- [X] [Gaussian Mixture Model and EM algo](http://www.mathworks.com/matlabcentral/fileexchange/35362-variational-bayesian-inference-for-gaussian-mixture-model)
- [X] Principal Components Analysis (as a low-rank approximation) using alternating least squares
- [X] Neural Networks (implementation from the [DeepLearn toolbox](https://github.com/rasmusbergpalm/DeepLearnToolbox/archive/))
- [X] Generic ML method comparison function (for each method, plot achieved test error & stability with a boxplot)

### Dataset pre-processing

- [X] Basic data characteristics (dimensionality, repartition, correlation, ...)
- [X] Try obtaining helpful visualizations
- [X] Dimensionality reduction with PCA

### Person detection dataset

- [X] Implement the relevant error measures
- [X] Try feature transformations (basis expansion)
- [X] Get a baseline error value
- [X] Implement PCA
- [X] Implement Logistic Regression
- [X] Experiment with the Neural Network's hyperparameters (number of layers, activation functions, dropout...)
- [X] Experiment with Gaussian Processes (check provided toolbox)
- [X] Experiment with SVM (check provided toolbox)
- [X] Experiment with Random Forest
- [X] Implement kCV fastROC
- [X] Generate feature selection plots (code to replicate for different methods)
- [X] Plot ROC Curves of different models for comparison
- [X] Maximize the train and test avgTPR
- [X] Check the stability of the results with k-CV

### Song recommendation dataset

Recall we must achieve both **weak** (new ratings for existing users) and **strong** (entirely new users) prediction.

- [X] Implement the relevant error measures
- [X] Implement error diagnostics (which kind of counts do we make the most error on?)
- [X] Train / test split (particular for weak and strong prediction)
- [X] Feature engineering: implement derived variables
- [X] Get a baseline error value
- [X] Implement Top-N recommendation (cluster with Pearson similarity measure)
- [X] Experiment with K-Means
- [ ] Implement the simple [Slope One](http://arxiv.org/pdf/cs/0702144v2.pdf) method
- [X] Experiment with Gaussian Mixture Models (soft clustering)
- [X] Cluster the tail items (head / tail cutoff point to be chosen carefully)
- [X] Try clustering in reduced-dimensionality space
- [X] Implement a hybrid head / tail predictor (e.g. Each Item for head, Top-K for tail)
- [ ] Determine if using the social graph helps weak prediction (then, we will be able to know if we can use it for strong prediction as well)
- [X] Use the social network and generic artist informations for strong prediction
- [X] Generate feature selection plots
- [X] Minimize the train and test error
- [X] Check the stability of the results with random train / test splits
- [X] Use the artists name to output fun facts

### Predictions

- [X] `songPred.mat` contains the two matrices `Ytest_weak_pred` (size 1774x15082) and `Ytest_strong_pred` (size 93x15082)
- [X] `personPred.mat` contains a vector 'Ytest_score' (8743x1) with the prediction score for each test sample

### Report

- [X] Describe and discuss the methods used and show that we understand their inner working and the influence of each hyperparameter (especially for methods we did not implement ourselves)
- [X] Produce figures for the detection dataset
- [X] Report work done for the detection dataset and the corresponding results
- [X] Produce figures for the recommendation dataset
- [X] Report work done for the recommendation dataset and the corresponding results
- [ ] Double-check all figures for labels (on each axis and for the figure itself)
- [X] Clear conclusion and analysis of the results for each dataset
- [ ] Include complete details about each algorithm (initialization values, lambda values, number of folds, number of trials, etc)
- [X] What worked and what did not? Why do you think are the reasons behind that?
- [X] Why did you choose the method that you chose?

Tools
-----

- [Piotr's toolbox](http://vision.ucsd.edu/~pdollar/toolbox/doc/)
- [DeepLearn toolbox](https://github.com/rasmusbergpalm/DeepLearnToolbox/archive/)
- [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/code/matlab/doc/)
- [Variational Bayesian Inference for Gaussian Mixture Model](http://www.mathworks.com/matlabcentral/fileexchange/35362-variational-bayesian-inference-for-gaussian-mixture-model) script by Mo Chen
