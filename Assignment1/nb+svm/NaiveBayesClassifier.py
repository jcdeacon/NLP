from random import randint
from functools import reduce
from Word import Word
from math import log
from collections import Counter


class NaiveBayesClassifier:
    def __init__(self):
        self.bank = dict()
        self.class_counts = Counter()
        self.total_count = 0
        self.class_priors = dict()
        self.class_wordcounts = dict()

    def feed(self, words, clazz):
        # Split text into words and increase frequencies as necessary

        for word in words:
            word_obj = self.bank.get(word, Word())
            word_obj.counts[clazz] += 1
            self.bank[word] = word_obj

        # Increase count of documents of this class we have seen
        self.class_counts[clazz] += 1
        self.total_count += 1

    def train(self):
        self.class_priors = {clazz: count / self.total_count for (clazz, count) in self.class_counts.items()}
        print('Training Naive Bayes with class counts:', self.class_counts, 'yielding priors', self.class_priors)

        # Sum of counts of all words, per class
        self.class_wordcounts = {
            clazz: reduce(lambda x, y: x + y.counts[clazz], self.bank.values(), 0)
            for clazz in self.class_counts
        }
        print('Training complete')

    def classify(self, bag_of_words, laplace):
        # Initialise the class probabilities as their priors
        probs = {clazz: log(prior) for (clazz, prior) in self.class_priors.items()}

        for clazz in probs:
            for word in filter(lambda x: x in self.bank, bag_of_words):
                word_count = self.bank[word].counts[clazz] + laplace
                class_count = self.class_wordcounts[clazz] + (laplace * len(self.bank))

                if word_count is 0:
                    continue

                probs[clazz] += log(word_count / class_count)

        # Check that the probabilities aren't the same for each class
        # If they are, return a random class
        if len(set(probs.values())) == 1:
            return list(probs.keys())[randint(0, len(probs.keys()) - 1)]

        return max(probs, key=probs.get)
