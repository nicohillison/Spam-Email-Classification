'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
YOUR NAME HERE
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np


class NaiveBayes:
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
     number of classes)'''
    def __init__(self, num_classes):
        '''Naive Bayes constructor

        TODO:
        - Add instance variable for `num_classes`
        '''
        # class_priors: ndarray. shape=(num_classes,).
        #   Probability that a training example belongs to each of the classes
        #   For spam filter: prob training example is spam or ham
        self.class_priors = None
        # class_likelihoods: ndarray. shape=(num_classes, num_features).
        #   Probability that each word appears within class c
        self.class_likelihoods = None

        self.num_classes = num_classes

    def train(self, data, y):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        class likelihoods (the probability of a word appearing in each class — spam or ham)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        TODO:
        - Compute the instance variables self.class_priors and self.class_likelihoods needed for
        Bayes Rule. See equations in notebook.
        '''
        num_samps, num_features = data.shape

        prior = []
        likelihood = []
        for i in range(self.num_classes):
            the_count = np.count_nonzero(y == i, axis = 0)/num_samps
            prior.append(the_count)
            class_likelihood = []
            for j in range(num_features):
                index = np.argwhere(y == i)
                total_count_w = np.sum(data[index, j])
                total_num_w = np.sum(data[index,:])
                bel_wL = (total_count_w + 1)/(total_num_w + num_features)
                class_likelihood.append(bel_wL)
            likelihood.append(class_likelihood)

        self.class_likelihoods = np.array(likelihood)
        self.class_priors = np.array(prior)


    def predict(self, data):
        '''Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `data` is the class that yields the highest posterior
        probability.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.

        TODO:
        - Process test samples one-by-one.
        - For each, we want to compute the log of the numerator of the posterior:
        - (a) Use matrix-vector multiplication (or the dot product) with the log of the likelihoods
          and the test sample (transposed, but no logarithm taken)
        - (b) Add to it the log of the priors
        - Predict the class of each test sample according to the class that produces the largest
        log(posterior) probability (use argmax)
        '''
        num_test_samps, num_features = data.shape

        sum_mtrx = data @ np.log(self.class_likelihoods.T)
        add_mtrx = np.log(self.class_priors) + sum_mtrx

        return add_mtrx.argmax(axis = 1)




    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        difference = y_pred - y
        non_zero = np.count_nonzero(difference)
        accuracy = (y.shape[0]-non_zero)/y.shape[0]
        return accuracy

    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Recall: the rows represent the "actual" ground truth labels, the columns represent the
        predicted labels.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''
        # To get the number of classes, you can use the np.unique
        # function to identify the number of unique categories in the
        # y matrix.

        mtrx = np.zeros(shape = (self.num_classes, self.num_classes))

        for i in range(len(y)):
            row = np.int(y[i])
            columns = np.int(y_pred[i])
            mtrx[row][columns] += 1
        
        return mtrx





