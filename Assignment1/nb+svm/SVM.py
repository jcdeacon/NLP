from pysvmlight.src import SvmLightEstimator as svm_light

from Polarity import Polarity
import random

class SVM:
    def __init__(self):
        self.learner = svm_light.SvmLightEstimator()
        self.data = []
        self.classifications = []

    def feed(self, words, clazz):
        #place = random.randrange(len(self.data)+1)
        place = len(self.data)+1
        self.data.insert(place, self.learner.process(words))
        if clazz == Polarity.POS:
            self.classifications.insert(place, 1)
        else:
            self.classifications.insert(place, -1)

    def train(self):
        print('Training SVM')
        for item in self.data:
            print(item)
            print("\n\n\n\n")
        self.learner.fit(self.data, self.classifications)

    def classify(self, bag_of_words):
        prediction = \
            self.learner.predict([self.learner.process(bag_of_words)])[0]
        if prediction > 0:
            return Polarity.POS
        else:
            return Polarity.NEG
