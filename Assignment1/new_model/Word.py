from collections import Counter


class Word:
    def __init__(self):
        self.counts = Counter()

    def add_instance(self, clazz):
        self.counts[clazz] += 1

    def has_counts_for_classes(self, classes):
        return all(clazz in self.counts for clazz in classes)
