from collections import Counter

from Polarity import Polarity
from NaiveBayesClassifier import NaiveBayesClassifier
from SVM import SVM
from Review import Review
from os import listdir, path
from math import factorial, ceil
from functools import partial
from decimal import Decimal

POSITIVE_TAGGED_DIR = 'stemmed_data/POS'
NEGATIVE_TAGGED_DIR = 'stemmed_data/NEG'

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

    z = two_tail_cumulative_binomial(int(n), int(k), 0.5)
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
def execute_naive_bayes(reviews, f_is_test_instance, smoothing, bigrams):
    test_set = filter(f_is_test_instance, reviews)
    train_set = filter(lambda x: not f_is_test_instance(x), reviews)

    classifier = NaiveBayesClassifier()
    for r in train_set:
        if bigrams:
            classifier.feed(r.all_features, r.polarity)
        else:
            classifier.feed(r.bag_of_words, r.polarity)

    classifier.train()
    if bigrams:
        return {r: classifier.classify(r.all_features, smoothing) for r in test_set}
    else:
        return {r: classifier.classify(r.bag_of_words, smoothing) for r in test_set}

def execute_svm(reviews, f_is_test_instance, bigrams):
    test_set = filter(f_is_test_instance, reviews)
    train_set = filter(lambda x: not f_is_test_instance(x), reviews)

    classifier = SVM()
    for r in train_set:
        if bigrams:
            classifier.feed(r.all_features, r.polarity)
        else:
            classifier.feed(r.bag_of_words, r.polarity)

    classifier.train()
    if bigrams:
        return {r: classifier.classify(r.all_features) for r in test_set}
    else:
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
End Classification things
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

def holdout_naive_bayes(tagged_reviews, bigrams=False, smoothing=1):
    is_test_function = lambda x: x.file_id >= 900
    nb_results = execute_naive_bayes(tagged_reviews, is_test_function, smoothing, bigrams)

    interpreted_results = interpret_results(nb_results)
    print('Correct:', interpreted_results['correct'], 'Wrong:', interpreted_results['wrong'], ', Unclassifiable:', interpreted_results['unclassifiable'], ', Accuracy:', interpreted_results['accuracy'])
    return nb_results

def holdout_svm(tagged_reviews, bigrams=False):
    is_test_function = lambda x: x.file_id >= 900
    svm_results = execute_svm(tagged_reviews, is_test_function, bigrams)
    interpreted_results = interpret_results(svm_results)
    print('Correct:', interpreted_results['correct'], 'Wrong:', interpreted_results['wrong'], ', Unclassifiable:', interpreted_results['unclassifiable'], ', Accuracy:', interpreted_results['accuracy'])
    return svm_results


def cross_val_naive_bayes(tagged_reviews, cv_method, bigrams = False, smoothing=1):
    cv_nb_results = list()
    classifications = dict()
    for k in range(10):
        print("\nTraining {}th fold".format(k))
        is_test_function = partial(cv_method, k)
        raw_results = execute_naive_bayes(tagged_reviews, is_test_function, smoothing, bigrams)
        classifications.update(raw_results)
        interpreted_results = interpret_results(raw_results)
        print('Correct:', interpreted_results['correct'], 'Wrong:', interpreted_results['wrong'], ', Accuracy:', interpreted_results['accuracy'])
        cv_nb_results.append(interpreted_results)

    cv_accuracy = sum(map(lambda x: x['accuracy'], cv_nb_results))
    cv_accuracy /= len(cv_nb_results)

    cv_variance = sum(map(lambda x: x['accuracy'] ** 2, cv_nb_results))
    cv_variance /= len(cv_nb_results)
    cv_variance -= cv_accuracy ** 2

    print('Cross validation accuracy:', cv_accuracy, 'variance:', cv_variance)
    return classifications

def cross_val_svm(tagged_reviews, cv_method, bigrams=False):
    cv_svm_results = list()
    classifications = dict()
    for k in range(10):
        print("\nTraining {}th fold".format(k))
        is_test_function = partial(cv_method, k)
        raw_results = execute_svm(tagged_reviews, is_test_function, bigrams)
        classifications.update(raw_results)
        interpreted_results = interpret_results(raw_results)
        print('Correct:', interpreted_results['correct'], 'Wrong:', interpreted_results['wrong'], ', Accuracy:', interpreted_results['accuracy'])
        cv_svm_results.append(interpreted_results)

    cv_accuracy = sum(map(lambda x: x['accuracy'], cv_svm_results))
    cv_accuracy /= len(cv_svm_results)

    cv_variance = sum(map(lambda x: x['accuracy'] ** 2, cv_svm_results))
    cv_variance /= len(cv_svm_results)
    cv_variance -= cv_accuracy ** 2

    print('Cross validation accuracy:', cv_accuracy, 'variance:', cv_variance)
    return classifications

'''
End different types of classifier
'''


def main():
    print('----')
    print('Holdout naive Bayes classification without bigrams')
    print('----')

    tagged_reviews = parse_reviews(POSITIVE_TAGGED_DIR, Polarity.POS)
    tagged_reviews.extend(parse_reviews(NEGATIVE_TAGGED_DIR, Polarity.NEG))

    print('With 1-smoothing')
    nb_1_smooth = holdout_naive_bayes(tagged_reviews, False, 1)

    ground_polarities = {r.file_name: r.polarity for r in nb_1_smooth.keys()}

    classifications = dict()
    for r in nb_1_smooth:
        classifications[r.file_name] = {'1-smooth-uni': nb_1_smooth[r]}
    
    print('----')
    print('Holdout naive Bayes classification with bigrams')
    print('----')

    print('With 1-smoothing')
    nb_1_smooth = holdout_naive_bayes(tagged_reviews, True, 1)

    for r in nb_1_smooth:
        classifications[r.file_name]['1-smooth-bi'] = nb_1_smooth[r]

    print('----')
    print('Holdout SVM classification without bigrams')
    print('----')

    svm = holdout_svm(tagged_reviews, False)
    for r in svm:
        classifications[r.file_name]['svm-uni'] = svm[r]

    print('----')
    print('Holdout SVM classification with bigrams')
    print('----')

    svm = holdout_svm(tagged_reviews, True)
    for r in svm:
        classifications[r.file_name]['svm-bi'] = svm[r]

    models = ['1-smooth-uni', '1-smooth-bi', 'svm-uni', 'svm-bi']

    for model1 in models:
        for model2 in models:
            if model1 != model2:
                perform_sign_test(ground_polarities.keys(), model1,
                    model2, classifications, ground_polarities)
    
    new_class = dict()
    cv_method = is_test_round_robin
    print('----')
    print('Cross-validated naive Bayes without bigrams classification')
    print('----')
    classifications_nb_uni = \
        cross_val_naive_bayes(tagged_reviews, cv_method, False, 1)
    for r in classifications_nb_uni:
        new_class[r.file_name] = {'nb-uni': classifications_nb_uni[r]}

    print('----')
    print('Cross-validated naive Bayes with bigrams classification')
    classifications_nb_bi = \
        cross_val_naive_bayes(tagged_reviews, cv_method, True, 1)
    for r in classifications_nb_bi:
        new_class[r.file_name]['nb-bi'] = classifications_nb_bi[r]
    
    print('----')
    print('Cross-validated SVM without bigrams classification')
    print('----')
    classifications_svm_uni = \
        cross_val_svm(tagged_reviews, cv_method, False)
    for r in classifications_svm_uni:
        new_class[r.file_name]['svm-uni'] = classifications_svm_uni[r]
    print('----')
    print('Cross-validated SVM with bigrams classification')
    print('----')
    classifications_svm_bi = \
        cross_val_svm(tagged_reviews, cv_method, True)
    for r in classifications_svm_bi:
        new_class[r.file_name]['svm-bi'] = classifications_svm_bi[r]
    
    models = ['nb-uni', 'nb-bi', 'svm-uni', 'svm-bi']
    ground_polarities = {r.file_name: r.polarity for r in classifications_svm_bi.keys()}    
    for model1 in models:
        for model2 in models:
            if model1 != model2:
                perform_sign_test(ground_polarities.keys(), model1,
                        model2, new_class, ground_polarities)

if __name__ == '__main__':
    main()
