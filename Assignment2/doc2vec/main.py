import gensim
from Polarity import Polarity
from os import listdir
from sklearn import svm

POSITIVE_TAGGED_DIR = 'data/pos/'
NEGATIVE_TAGGED_DIR = 'data/neg/'

polarities = dict()
validation = []
training = []

def parse(path, file_name):
    with open(path + file_name, 'r') as f:
        line = f.readlines()
    assert(len(line) == 1)
    return gensim.models.doc2vec.TaggedDocument(gensim.utils.
        simple_preprocess(line[0]), file_name)

def get_data():
    positive_reviews = []
    negative_reviews = []
    for file_name in listdir(POSITIVE_TAGGED_DIR):
        positive_reviews.append(parse(POSITIVE_TAGGED_DIR, file_name))
        polarities[file_name] = Polarity.POS
    for file_name in listdir(NEGATIVE_TAGGED_DIR):
        negative_reviews.append(parse(NEGATIVE_TAGGED_DIR, file_name))
        polarities[file_name] = Polarity.NEG
    for i in range(len(positive_reviews)):
        if i % 10 == 0:
            validation.append(positive_reviews[i])
        else:
            training.append(positive_reviews[i])
    for i in range(len(negative_reviews)):
        if i % 10 == 0:
            validation.append(negative_reviews[i])
        else:
            training.append(negative_reviews[i])

def cross_val(labels, data):
    print "Starting cross validation"
    accuracies = []
    for i in range(10):
        print "Using fold %d" % i
        training = []
        training_labels = []
        testing = []
        testing_labels = []
        for j in range(len(labels)):
            if j % 10 == i:
                testing_labels.append(labels[j])
                testing.append(data[j])
            else:
                training_labels.append(labels[j])
                training.append(data[j])
        print "Split data"
        clf = svm.SVC(gamma="scale")
        clf.fit(training, training_labels)
        accuracies.append(clf.score(testing, testing_labels))
    print "Cross validation complete"
    return sum(accuracies) / len(accuracies)

    # for each fold
    #   get the folds
    #   train on training and then find the accuracy on testing
    # return the average accuracy

def main():
    # Separate the data into validation and training
    get_data()
    # Prepare a list of parameters
    parameters = [{'dm' : 1, 'dim' : 100, 'epch' : 20, 'wdw' : 5, 'hs' : 1}]
    '''    {'dm' : 0, 'dim' : 100, 'epch' : 20, 'wdw' : 5, 'hs' : 1},
        {'dm' : 1, 'dim' : 100, 'epch' : 20, 'wdw' : 5, 'hs' : 0},
        {'dm' : 0, 'dim' : 100, 'epch' : 20, 'wdw' : 5, 'hs' : 0},
        {'dm' : 1, 'dim' : 100, 'epch' : 20, 'wdw' : 10, 'hs' : 1},
        {'dm' : 0, 'dim' : 100, 'epch' : 20, 'wdw' : 10, 'hs' : 1},
        {'dm' : 1, 'dim' : 100, 'epch' : 20, 'wdw' : 10, 'hs' : 0},
        {'dm' : 0, 'dim' : 100, 'epch' : 20, 'wdw' : 10, 'hs' : 0}]
    '''# For each set of parameters, train the training set and evaluate
    #   on the validation set.
    '''
    for each parameter set:
        train doc2vec on the training set
        generate vectors for the validation set
        perform cross validation of SVMs on the validation set
        store the accuracy
    '''
    accuracies = []
    for par in parameters:
        print "Testing parameters"
        print par
        model = gensim.models.doc2vec.Doc2Vec(dm=par['dm'],
            vector_size=par['dim'], epochs=par['epch'], window=par['wdw'],
            hs=par['hs'])
        model.build_vocab(training)
        model.train(training, total_examples=len(training),
            epochs=model.epochs)
        validation_vectors = []
        for v in validation:
            validation_vectors.append((model.infer_vector(v[0]), v[1]))
        labels = [v[1] for v in validation_vectors]
        data = [v[0] for v in validation_vectors]
        accuracies.append(cross_val(labels, data))
    print accuracies


    # Choose the parameters with the best results on the validation set.
    # Perform some permutation tests.
    # Using the best parameters, run cross-validation on the training set.

if __name__ == "__main__":
    main()
