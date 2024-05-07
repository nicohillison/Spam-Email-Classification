'''email_preprocessor.py
Preprocess Enron email dataset into features for use in supervised learning algorithms
Nico Hillison
CS 251 Data Analysis Visualization, Spring 2020
'''
import re
import os
import numpy as np


def tokenize_words(text):
    '''Transforms an email into a list of words.

    Parameters:
    -----------
    text: str. Sentence of text.

    Returns:
    -----------
    Python list of str. Words in the sentence `text`.

    This method is pre-filled for you (shouldn't require modification).
    '''
    # Define words as lowercase text with at least one alphabetic letter
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())


def count_words(email_path='data/enron'):
    '''Determine the count of each word in the entire dataset (across all emails)

    Parameters:
    -----------
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_emails: int. Total number of emails in the dataset.

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use the `tokenize_words` function above to chunk it into a list of words.
    - Update the counts of each word in the dictionary.

    Hints:
    - Check out Python functions in the os and os.path modules for walking the directory structure.
    '''
    names = []
    dic = {}
    for (dirpath, dirnames, filenames) in os.walk(email_path):
        names += [os.path.join(dirpath, f) for f in filenames if f[-4:] == '.txt']
    for i in range(len(names)):
        with open(names[i], encoding='latin-1') as f:
            content = tokenize_words(f.read())
            for j in content:
                if j in dic:
                    dic[j] += 1
                else:
                    dic[j] = 1
        f.close()

    return dic, len(names)



def find_top_words(word_freq, num_features=200):
    '''Given the dictionary of the words that appear in the dataset and their respective counts,
    compile a list of the top `num_features` words and their respective counts.

    Parameters:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_features: int. Number of top words to select.

    Returns:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    counts: Python list. Counts of the `num_features` words in high-to-low count order.
    '''
    d = sorted(word_freq, key = word_freq.get, reverse= True)
    print(d)
    counter = 0
    top_words = []
    count = []
    for k in d:
        top_words.append(k)
        count.append(word_freq[k])

    
    return top_words[:num_features], count[:num_features]

def make_feature_vectors(top_words, num_emails, email_path='data/enron'):
    '''Count the occurance of the top W (`num_features`) words in each individual email, turn into
    a feature vector of counts.

    Parameters:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    num_emails: int. Total number of emails in the dataset.
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    feats. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use `tokenize_words` to chunk it into a list of words.
    - Count the occurance of each word, ONLY THOSE THAT APPEAR IN `top_words`.

    HINTS:
    - Start with your code in `count_words` and modify as needed.
    '''
    num_features = len(top_words)
    y = np.zeros((num_emails, 1))
    feats = np.zeros((num_emails, num_features))
    num = 0
    counter = 0

    for i in os.listdir(email_path):
        if i != '.DS_Store':
            for name in os.listdir(os.path.join(email_path, i)):
                ind_name = os.path.join(email_path, i, name)

                f = open(ind_name)
                str_mail = f.read().replace("\n"," ")
                f.close()

                lst_to_wrds = tokenize_words(str_mail)
                print('lst_to_wrds: ', lst_to_wrds)
                counter += 1
                count = []
                for j in lst_to_wrds:
                    if j in top_words:
                        idx_tw = top_words.index(j)
                        feats[counter, idx_tw] += 1


                    count_w = np.count_nonzero(j in lst_to_wrds)
                    count.append(count_w)
                print('count: ', count)
                feats[:,] = count
                
                
                if i == 'spam':
                    y[num,:] = 0
                    num += 1
                else:
                    y[num,:] = 1
                    num += 1
    
    y = y.flatten()
    print('y: ', len(y))

    return feats, y


def make_train_test_sets(features, y, test_prop=0.2, shuffle=True):
    '''Divide up the dataset `features` into subsets ("splits") for training and testing. The size
    of each split is determined by `test_prop`.

    Parameters:
    -----------
    features. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)
    test_prop: float. Value between 0 and 1. What proportion of the dataset samples should we use
        for the test set? e.g. 0.2 means 20% of samples are used for the test set, the remaining
        80% are used in training.
    shuffle: boolean. Should we shuffle the data before splitting it into train/test sets?

    Returns:
    -----------
    x_train: ndarray. shape=(num_train_samps, num_features).
        Training dataset
    y_train: ndarray. shape=(num_train_samps,).
        Class values for the training set
    inds_train: ndarray. shape=(num_train_samps,).
        The index of each training set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    x_test: ndarray. shape=(num_test_samps, num_features).
        Test dataset
    y_test:ndarray. shape=(num_test_samps,).
        Class values for the test set
    inds_test: ndarray. shape=(num_test_samps,).
        The index of each test set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    '''
    inds = np.arange(y.size)
    if shuffle:
        features = features.copy()
        y = y.copy()

        inds = np.arange(y.size)
        np.random.shuffle(inds)
        features = features[inds]
        y = y[inds]

    # Your code here:
    n_samp = y.size
    n_test_samps = int(test_prop * n_samp)
    n_train_samps = y.size - n_test_samps

    x_train = features[0:n_train_samps,:]
    x_test = features[n_train_samps:n_samp, :]
    y_train = features[0:n_train_samps][:,0]
    y_test = features[n_train_samps:n_samp][:,0]
    inds_train = inds[0:n_train_samps]
    inds_test = inds[n_train_samps:n_samp]


    return x_train, y_train, inds_train, x_test, y_test, inds_test


def retrieve_emails(inds, email_path='data/enron'):
    '''Obtain the text of emails at the indices `inds` in the dataset.

    Parameters:
    -----------
    inds: ndarray of nonnegative ints. shape=(num_inds,).
        The number of ints is user-selected and indices are counted from 0 to num_emails-1
        (counting does NOT reset when switching to emails of another class).
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    Python list of str. len = num_inds = len(inds).
        Strings of entire raw emails at the indices in `inds`
    '''
    raw_emails = []
    counter = 0
    for i in os.listdir(email_path):
        if i != '.DS_Store':
            for name in os.listdir(os.path.join(email_path, i)):
                ind_name = os.path.join(email_path, i, name)

                f = open(ind_name)
                str_mail = f.read().replace("\n"," ")
                f.close()

                
                if counter in inds:
                    raw_emails.append(str_mail)
                counter += 1

    return  raw_emails
