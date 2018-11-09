import svmlight

from Polarity import Polarity
import sets

class SVM:
    def __init__(self):
        self.bank = {}
        self.total_count = 0
        self.training_data = []
        self.classifications = []

    def feed(self, words, clazz):
        features = []
        word_set = sets.Set()
        for word in words:
            if word in word_set:
                continue
            else:
                word_set.add(word)
            if word in self.bank:
                features.append((self.bank[word], 1))
            else:
                self.total_count += 1
                self.bank[word] = self.total_count
                features.append((self.bank[word], 1))
        features.sort()
        if clazz == Polarity.POS:
            self.training_data.append((1, features))
        else:
            self.training_data.append((-1, features))

    def train(self):
        print('Training SVM')
        self.model = svmlight.learn(self.training_data,
            type='classification')

    def classify(self, bag_of_words):
        test_example = []
        word_set = sets.Set()
        for word in bag_of_words:
            if word in word_set:
                continue
            else:
                word_set.add(word)
            if word in self.bank:
                test_example.append((self.bank[word], 1))
        prediction = svmlight.classify(self.model, [(0, test_example)])
        if prediction[0] > 0:
            return Polarity.POS
        else:
            return Polarity.NEG
