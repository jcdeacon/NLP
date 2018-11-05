import re

'''
Holds a bag of words, bundled with two different tokenization functions
for each type of file (data/{POS,NEG}/*.txt and data-tagged/{POS,NEG}/*.tag).
'''


class Review:
    untagged_regex = re.compile('[A-Za-z\']+')
    tagged_regex = re.compile('^[A-Za-z]+$')

    def __init__(self, file_name, polarity, raw_text, force_lowercase=False):
        self.polarity = polarity
        self.file_name = file_name
        
        is_tagged = '.tag' in file_name
        self.file_id = int(file_name.split('_')[0].replace('cv', ''))

        # Use different tokenizer dependent on what type of source file it is
        if is_tagged:
            results = map(lambda x: x.strip().split('\t')[0], raw_text)
            self.bag_of_words = list(filter(lambda x: Review.tagged_regex.search(x), results))
            self.bigrams = list(zip(self.bag_of_words, self.bag_of_words[1:]))
            self.all_features = self.bag_of_words + self.bigrams
        else:
            raise NotImplementedError
        #     tokenize_function = self.tokenize_untagged

    # def tokenize_untagged(self, text):
    #     lines = map(lambda x: x.strip().split(' '), text)
    #     # The previous gives us a list of lists which is inconvenient, so flatten
    #     results = filter(lambda x: x != '', [val for sublist in lines for val in sublist])
    #     results = filter(lambda x: Review.untagged_regex.search(x), results)
    #     results = map(lambda x: Review.untagged_regex.search(x).group(0), results)
    #     self.bag_of_words = results

