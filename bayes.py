import re
from collections import defaultdict
import cPickle
import codecs


def strip_garbage(text):
    """ Removes any substrings matching any regex contained in the regex
        file. Used for cleaning the input data prior to training."""

    try:
        regexes = codecs.open('regex', encoding='utf-8')
        for regex in regexes.readlines():
            strip = re.compile(regex.rstrip())
            text = strip.sub('', text)
    except IOError:
        print "Regex file not found."

    return text


def get_unigrams(doc):
    splitter = re.compile('\\W*')
    unigrams = [s.lower() for s in splitter.split(doc)]
    return unigrams


def get_ngrams(doc):
    """ Returns a dictionary containing unigram, bigram,
        and trigram counts for your wealth and amusement."""

    unigrams = get_unigrams(doc)

    ngrams = []

    bigrams, trigrams = [], []
    for i in xrange(len(unigrams)):
        if (i+1) % 2 == 0 and i > 0:
            ngrams.append(unigrams[i-1] + unigrams[i])
        if (i+1) % 3 == 0 and i > 0:
            ngrams.append(unigrams[i-2] + unigrams[i-1] + unigrams[i])

    ngrams.extend(unigrams)
    ngram_counts = defaultdict(int)

    for gram in ngrams:
        ngram_counts[gram] += 1

    return dict(ngram_counts)


class classifier(object):

    def __init__(self, get_features):
        self.fc_pairs = {}
        self.cc = {}
        self.get_features = get_features

    # Increase the count of a feature/category pair
    def inc_fc_pair(self, f, cat):
        self.fc_pairs.setdefault(f, {})
        self.fc_pairs[f].setdefault(cat, 0)
        self.fc_pairs[f][cat] += 1

    # Increaes the count of a category
    def inc_cat_count(self, cat):
        self.cc.setdefault(cat, 0)
        self.cc[cat] += 1

    # The number of times a feature has appeared in a category
    def f_count(self, f, cat):
        if f in self.fc_pairs and cat in self.fc_pairs[f]:
            return float(self.fc_pairs[f][cat])
        return 0.0

    # The number of items in a category
    def cat_count(self, cat):
        if cat in self.cc:
            return float(self.cc[cat])

    # The total number of items:
    def total_count(self):
        return sum(self.cc.values())

    # The list of all categories:
    def categories(self):
        return self.cc.keys()

    def train(self, item, cat):
        features = self.get_features(item)

        # Increment the count for every feature with this category
        for f in features:
            self.inc_fc_pair(f, cat)

        # Increment the count for this category
        self.inc_cat_count(cat)

    def f_prob(self, f, cat):
        if self.cat_count(cat) == 0:
            return
        # The total number of times this feature appeared in this category
        # divided by the total number of items in this category
        return self.f_count(f, cat) / self.cat_count(cat)

    def weighted_prob(self, f, cat, prf, weight=1.0, ap=0.5):
        # Calculate the current probability
        basicprob = prf(f, cat)

        # Count the number of times this feature has appeared in all cats
        totals = sum([self.f_count(f, c) for c in self.categories()])

        # Calculate the weighted average
        bp = ((weight * ap) + (totals * basicprob)) / (weight + totals)
        return bp

    def sample_train(self):
        self.train('Nobody owns the water.', 1)
        self.train('the quick rabbit jumps fences', 1)
        self.train('buy pharmaceuticals now', 0)
        self.train('make quick money at the online casino', 0)
        self.train('the quick brown fox jumps', 1)


class naive_bayes(classifier):
    def __init__(self, get_features):
        classifier.__init__(self, get_features)
        self.thresholds = {}

    def doc_prob(self, item, cat):
        features = self.get_features(item)

        # Multiply the probabilities of all the features together
        p = 1
        for f in features:
            p *= self.weighted_prob(f, cat, self.f_prob)
        return p

    def set_threshold(self, cat, t):
        self.thresholds[cat] = t

    def get_threshold(self, cat):
        if cat not in self.thresholds:
            return 1.0
        else:
            return self.thresholds[cat]

    # Pr(A|B) = Pr(B|A) x Pr(A)/Pr(B)
    # Pr(cat|doc) = Pr(doc|cat) * Pr(cat)/Pr(doc)
    # We can ignore Pr(doc) for our purposes
    def prob(self, item, cat):
        cat_prob = self.cat_count(cat) / self.total_count()
        doc_prob = self.doc_prob(item, cat)
        return doc_prob * cat_prob

    def classify(self, item, default=None):
        probs = {}
        # Find the category with the highest probability
        maxprob = 0.0
        for cat in self.categories():
            probs[cat] = self.prob(item, cat)
            if probs[cat] > maxprob:
                maxprob = probs[cat]
                best = cat

        # Make sure the probability exceeds threshold*next best
        for cat in probs:
            if cat == best:
                continue
            if probs[cat] * self.get_threshold(best) > probs[best]:
                return default

        return best

    def save(self):
        try:
            cPickle.dump(self.fc_pairs, open('fc_pairs.pickle', 'w'))
            cPickle.dump(self.c, open('cc.pickle', 'w'))
            cPickle.dump(self.thresholds, open('thresholds.pickle', 'w'))
        except IOError:
            print "Could not save files."

    def load(self):
        try:
            self.fc_pairs = cPickle.load(open('fc_pairs.pickle'))
            self.cc = cPickle.load(open('cc.pickle'))
            self.thresholds = cPickle.load(open('thresholds.pickle'))
        except IOError:
            print "Could not load files."
