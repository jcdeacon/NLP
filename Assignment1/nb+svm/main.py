from collections import Counter

from Polarity import Polarity
from NaiveBayesClassifier import NaiveBayesClassifier
from SVM import SVM
from Review import Review
from os import listdir, path
from math import factorial, ceil
from functools import partial
from decimal import Decimal

POSITIVE_TAGGED_DIR = 'data/POS'
NEGATIVE_TAGGED_DIR = 'data/NEG'

'''
Sign test things
'''
def two_tail_cumulative_binomial(n, k, q):
    print('Two tailed cumulative binomial with n=', n, 'k=', k, 'q=', q)
    q1 = lambda i: Decimal(q)     ** Decimal(i)
    q2 = lambda i: Decimal(1 - q) ** Decimal(n - i)
    acc = sum([nCr(n, i) * q1(i) * q2(i) for i in range(k)])
    return 2 * acc


def nCr(n, r):
    numerator   = Decimal(factorial(n))
    denominator = Decimal(factorial(r) * factorial(n - r))
    return numerator / denominator


def get_sign_values(reviews, system1, system2, classifications, polarities):
    sign_values = {'plus': 0, 'minus': 0, 'null': 0}

    for review in reviews:
        system1_correct = classifications[review][system1] == polarities[review]
        system2_correct = classifications[review][system2] == polarities[review]

        if system1_correct == system2_correct:
            sign_values['null'] += 1
        elif system1_correct:
            sign_values['plus'] += 1
        elif system2_correct:
            sign_values['minus'] += 1

    return sign_values


def perform_sign_test(reviews, system1, system2, classifications, polarities):
    print('Performing sign test between %s and %s:' % (system1, system2))
    sign_values = get_sign_values(reviews, system1, system2, classifications, polarities)
    print('Null:', sign_values['null'], 'Plus:', sign_values['plus'], 'Minus:', sign_values['minus'], )
    n = 2 * ceil(sign_values['null'] / 2.) + sign_values['plus'] + sign_values['minus']
    k = ceil(sign_values['null'] / 2.) + min(sign_values['plus'], sign_values['minus'])

    z = two_tail_cumulative_binomial(n, k, 0.5)
    p = 0.01
    if z < p:
        print(z, '<', p, '=> Statistically significant')
    else:
        print(z, '>=', p, '=> Not statistically significant')

'''
End sign test things
'''

'''
Parse all the files in a directory into Review objects and return them
'''
def parse_reviews(directory, polarity):
    reviews = list()
    for file_name in listdir(directory):
        with open(path.join(directory, file_name), 'r') as f:
            review = Review(file_name, polarity, f.readlines())
            reviews.append(review)
    return reviews
'''
End parsing
'''

'''
Classifier related stuff
'''
def execute_naive_bayes(reviews, f_is_test_instance, smoothing):
    test_set = filter(f_is_test_instance, reviews)
    train_set = filter(lambda x: not f_is_test_instance(x), reviews)

    classifier = NaiveBayesClassifier()
    for r in train_set:
        classifier.feed(r.bag_of_words, r.polarity)

    classifier.train()
    return {r: classifier.classify(r.bag_of_words, smoothing) for r in test_set}

def execute_svm(reviews, f_is_test_instance):
    test_set = filter(f_is_test_instance, reviews)
    train_set = filter(lambda x: not f_is_test_instance(x), reviews)

    # TODO: implement SVM
    classifier = SVM()
    for r in train_set:
        classifier.feed(r.bag_of_words, r.polarity)

    classifier.train()
    return {r: classifier.classify(r.bag_of_words) for r in test_set}

def interpret_results(results):
    correct = 0
    wrong = 0
    unclassifiable = 0
    for (review, classification) in results.items():
        if classification is None:
            unclassifiable += 1
        elif review.polarity == classification:
            correct += 1
        else:
            wrong += 1

    interpretation = dict()
    interpretation['correct'] = correct
    interpretation['wrong'] = wrong
    interpretation['unclassifiable'] = unclassifiable
    interpretation['accuracy'] = 100 * correct / (correct + wrong)
    return interpretation

'''
End Naive Bayes things
'''

'''
Utility functions to decide whether a given test instance
is in the test or train set when doing cross validation.

Both assume 10-fold cross validation.
'''
def is_test_consecutive_splitting(split_number, test):
    lower_limit = 100 * split_number
    upper_limit = 100 * (split_number + 1)
    return lower_limit <= test.file_id < upper_limit


def is_test_round_robin(split_number, test):
    x = test.file_id % 10
    return x == split_number
'''
End cross validation utility functions.
'''

'''
Begin different types of classifier
'''

def holdout_naive_bayes(tagged_reviews, smoothing=1):
    is_test_function = lambda x: x.file_id >= 900
    nb_results = execute_naive_bayes(tagged_reviews, is_test_function, smoothing)

    interpreted_results = interpret_results(nb_results)
    print('Correct:', interpreted_results['correct'], 'Wrong:', interpreted_results['wrong'], ', Unclassifiable:', interpreted_results['unclassifiable'], ', Accuracy:', interpreted_results['accuracy'])
    return nb_results

def holdout_svm(tagged_reviews):
    is_test_function = lambda x: x.file_id >= 900
    svm_results = execute_svm(tagged_reviews, is_test_function)
    interpreted_results = interpret_results(svm_results)
    print('Correct:', interpreted_results['correct'], 'Wrong:', interpreted_results['wrong'], ', Unclassifiable:', interpreted_results['unclassifiable'], ', Accuracy:', interpreted_results['accuracy'])
    return svm_results


def cross_val_naive_bayes(tagged_reviews, cv_method, smoothing=1):
    cv_nb_results = list()
    for k in range(10):
        print("\nTraining {}th fold".format(k))
        is_test_function = partial(cv_method, k)
        raw_results = execute_naive_bayes(tagged_reviews, is_test_function, smoothing)
        interpreted_results = interpret_results(raw_results)
        print('Correct:', interpreted_results['correct'], 'Wrong:', interpreted_results['wrong'], ', Accuracy:', interpreted_results['accuracy'])
        cv_nb_results.append(interpreted_results)

    cv_accuracy = sum(map(lambda x: x['accuracy'], cv_nb_results))
    cv_accuracy /= len(cv_nb_results)

    cv_variance = sum(map(lambda x: x['accuracy'] ** 2, cv_nb_results))
    cv_variance /= len(cv_nb_results)
    cv_variance -= cv_accuracy ** 2

    print('Cross validation accuracy:', cv_accuracy, 'variance:', cv_variance)

def cross_val_svm(tagged_reviews, cv_method):
    cv_svm_results = list()
    for k in range(10):
        print("\nTraining {}th fold".format(k))
        is_test_function = partial(cv_method, k)
        raw_results = execute_svm(tagged_reviews, is_test_function)
        interpreted_results = interpret_results(raw_results)
        print('Correct:', interpreted_results['correct'], 'Wrong:', interpreted_results['wrong'], ', Accuracy:', interpreted_results['accuracy'])
        cv_svm_results.append(interpreted_results)

    cv_accuracy = sum(map(lambda x: x['accuracy'], cv_svm_results))
    cv_accuracy /= len(cv_svm_results)

    cv_variance = sum(map(lambda x: x['accuracy'] ** 2, cv_svm_results))
    cv_variance /= len(cv_svm_results)
    cv_variance -= cv_accuracy ** 2

    print('Cross validation accuracy:', cv_accuracy, 'variance:', cv_variance)

'''
End different types of classifier
'''


def main():
    print('----')
    print('Holdout naive Bayes classification')
    print('----')

    tagged_reviews = parse_reviews(POSITIVE_TAGGED_DIR, Polarity.POS)
    tagged_reviews.extend(parse_reviews(NEGATIVE_TAGGED_DIR, Polarity.NEG))

    print('With 0-smoothing')
    nb_0_smooth = holdout_naive_bayes(tagged_reviews, 0)
    print('With 1-smoothing')
    nb_1_smooth = holdout_naive_bayes(tagged_reviews, 1)

    ground_polarities = {r.file_name: r.polarity for r in nb_0_smooth.keys()}

    classifications = dict()
    for r in nb_0_smooth:
        classifications[r.file_name] = {'0-smoothing': nb_0_smooth[r]}
    for r in nb_1_smooth:
        classifications[r.file_name]['1-smoothing'] = nb_1_smooth[r]

    perform_sign_test(ground_polarities.keys(), '0-smoothing', '1-smoothing', classifications, ground_polarities)

    print('----')
    print('Holdout SVM classification')
    print('----')

    svm = holdout_svm(tagged_reviews)
    for r in svm:
        classifications[r.file_name] = {'svm': svm[r]}

    perform_sign_test(ground_polarities.keys(), 'svm', '1-smoothing',
        classifications, ground_polarities)

    print('----')
    print('Cross-validated naive Bayes classification')
    print('----')
    # Different possibilities for cross-validation splitting
    cv_method = is_test_round_robin
    # cv_method = is_test_consecutive_splitting
    cross_val_naive_bayes(tagged_reviews, cv_method, 1)

    print('----')
    print('Cross-validated SVM classification')
    print('----')
    cross_val_svm(tagged_reviews, cv_method)

if __name__ == '__main__':
    main()
