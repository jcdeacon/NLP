import gensim
from Polarity import Polarity
from os import listdir
from sklearn import svm
import random
import numpy

POSITIVE_TAGGED_DIR = 'data/pos/'
NEGATIVE_TAGGED_DIR = 'data/neg/'

polarities = dict()
ground_truth = dict()
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
        polarities[file_name] = 1
    for file_name in listdir(NEGATIVE_TAGGED_DIR):
        negative_reviews.append(parse(NEGATIVE_TAGGED_DIR, file_name))
        polarities[file_name] = -1
    for i in range(len(positive_reviews)):
        if i % 10 == 0:
            validation.append(positive_reviews[i])
            ground_truth[positive_reviews[i][1]] = 1
        else:
            training.append(positive_reviews[i])
    for i in range(len(negative_reviews)):
        if i % 10 == 0:
            validation.append(negative_reviews[i])
            ground_truth[negative_reviews[i][1]] = -1
        else:
            training.append(negative_reviews[i])

def cross_val(labels, data, filenames):
    print("Starting cross validation")
    parameters = [{"kern" : 'linear'},
        {"kern" : 'poly'},
        {"kern" : 'rbf'},
        {"kern" : 'sigmoid'}]
    all_param_acc = []
    all_param_pred = []
    for par in parameters:
        print("Using parameters")
        print(par)
        accuracies = []
        predictions = dict()
        for i in range(10):
            print("Using fold %d" % i)
            training = []
            training_labels = []
            testing = []
            testing_labels = []
            testnames = []
            for j in range(len(labels)):
                if j % 10 == i:
                    testing_labels.append(labels[j])
                    testing.append(data[j])
                    testnames.append((filenames[j], j))
                else:
                    training_labels.append(labels[j])
                    training.append(data[j])
            clf = svm.SVC(kernel=par["kern"])
            clf.fit(training, training_labels)
            score = clf.score(testing, testing_labels)
            accuracies.append(score)
            for name in testnames:
                predictions[name[0]] = clf.predict([data[name[1]]])[0]
        all_param_acc.append(sum(accuracies) / len(accuracies))
        all_param_pred.append(predictions)
    print("Cross validation complete")
    return (all_param_acc, all_param_pred)

def eval_svm(labels, data):
    train = []
    train_labels = []
    test = []
    test_labels = []
    for i in range(len(labels)):
        if i % 10 == 0:
            test_labels.append(labels[i])
            test.append(data[i])
        else:
            train_labels.append(labels[i])
            train.append(data[i])
    clf = svm.SVC(kernel='sigmoid')
    clf.fit(train, train_labels)
    return clf.score(test, test_labels)


def perform_permutation_test(results, truth, pairs):
    statistics = []
    for pair in pairs:
        print("Working on pair")
        print(pair)
        (i, j) = pair
        statistics.append((permutation_test(results[i], results[j], truth),
            pair))
    print(statistics)

def permutation_test(results1, results2, truth):
    s = 0
    R = 5000
    sum_acc1, sum_acc2 = 0, 0
    for key in results1:
        sum_acc1 += (results1[key] == truth[key])
        sum_acc2 += (results2[key] == truth[key])
    mean_diff = (sum_acc1 - sum_acc2) / len(truth)
    if mean_diff < 0:
        mean_diff *= -1
    for i in range(R):
        new1 = dict()
        new2 = dict()
        for key in results1:
            switch = random.randint(0, 1)
            if switch:
                new1[key] = results2[key]
                new2[key] = results1[key]
            else:
                new1[key] = results1[key]
                new2[key] = results2[key]
        newsum1, newsum2 = 0, 0
        for key in new1:
            newsum1 += (new1[key] == truth[key])
            newsum2 += (new2[key] == truth[key])
        new_mean_diff = (newsum1 - newsum2) / len(truth)
        if new_mean_diff < 0:
            new_mean_diff *= 1
        s += (new_mean_diff >= mean_diff)
    return (s + 1) / (R + 1)



# for each fold
    #   get the folds
    #   train on training and then find the accuracy on testing
    # return the average accuracy

def main1():
    # Separate the data into validation and training
    get_data()
    # Prepare a list of parameters
    dim = 100
    epch = 20
    dm = [1, 0]
    wdw = [5, 10]
    hs = [1, 0]
    parameters = []
    for d in dm:
        for w in wdw:
            for h in hs:
                parameters.append({'dm':d, 'dim':dim, 'epch':epch, 'wdw':w,
                    'hs':h})
    # For each set of parameters, train the training set and evaluate
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
        print("Testing parameters")
        print(par)
        model = gensim.models.doc2vec.Doc2Vec(dm=par['dm'],
            vector_size=par['dim'], epochs=par['epch'], window=par['wdw'],
            hs=par['hs'])
        model.build_vocab(training)
        model.train(training, total_examples=len(training),
            epochs=model.epochs)
        validation_vectors = []
        for v in validation:
            validation_vectors.append((model.infer_vector(v[0]), v[1]))
        labels = [polarities[v[1]] for v in validation_vectors]
        data = [v[0] for v in validation_vectors]
        filenames = [v[1] for v in validation_vectors]
        accuracies.append(cross_val(labels, data, filenames))
    print([a[0] for a in accuracies])
    all_accuracies = []
    all_predictions = []
    for item in accuracies:
        all_accuracies.extend(item[0])
        all_predictions.extend(item[1])
    indices = numpy.argsort(all_accuracies)
    print(indices)
    pairs = [(indices[i], indices[i+1]) for i in range(len(indices)-1)]
    print("pairs")
    print(pairs)
    perform_permutation_test(all_predictions, ground_truth, pairs)


    # Choose the parameters with the best results on the validation set.
    # Perform some permutation tests.
    # Using the best parameters, run cross-validation on the training set.

def main2():
    get_data()
    accuracies = []
    for i in range(10):
        print("beginning fold %d" % i)
        new_train = []
        new_test = []
        for j in range(len(training)):
            if j % 10 == i:
                new_test.append(training[j])
            else: new_train.append(training[j])
        model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=100,
            epochs=20, window=5, hs=0)
        model.build_vocab(new_train)
        model.train(new_train, total_examples = len(new_train),
            epochs=model.epochs)
        test_vectors = []
        for t in new_test:
            test_vectors.append((model.infer_vector(t[0]), t[1]))
        labels = [polarities[t[1]] for t in test_vectors]
        data = [t[0] for t in test_vectors]
        accuracies.append(eval_svm(labels, data))
    print(sum(accuracies) / len(accuracies))





if __name__ == "__main__":
    main1()
