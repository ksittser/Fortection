import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
#from nltk import NaiveBayesClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB,CategoricalNB,ComplementNB
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM,MLPClassifier
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import math
import random
import pickle
import pandas

class Formatter:
    def __init__(self,rawData=[]):
        self.rawData = rawData
        self.data = None
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def loadDataFromFile(self, f):
        '''Unpickle data from file'''
        file = open(f, 'rb')
        self.data = pickle.load(file)
        file.close()

    def writeDataToFile(self, f):
        '''Pickle data to file'''
        file = open(f, 'wb')
        pickle.dump(self.data, file)
        file.close()

    def reformatJson(self):
        '''Take JSON data and reformat to [(passage,label),(passage,label), ...] format'''
        data = []  # list of (passage,label) pairs
        for book in self.rawData:
            for tech in book['technologies']:
                if 'excerpt' in book['technologies'][tech] and 'category' in book['technologies'][tech]:
                    data.append((book['technologies'][tech]['excerpt'], book['technologies'][tech]['category']))

        # Currently only using the labels with the most data, to minimize bias
        self.data = [(text, label) for (text, label) in data]

    def removePunc(self,tokens):
        '''Return text with punctuation removed'''
        # https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/
        translator = str.maketrans('', '', string.punctuation)
        return ' '.join(tokens).translate(translator).split()

    def tokenize(self,text):
        '''Return list of tokens from text'''
        return word_tokenize(text)

    def removeStop(self,tokens):
        '''Return string with NLTK standard stop words removed'''
        # https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/
        stopWords = set(stopwords.words("english"))
        stopWords.update(['\n','\r'])
        filtered = [word for word in tokens if word not in stopWords]
        return filtered

    def stem(self,tokens):
        '''Use Porter Stemmer on token list'''
        # https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/
        stems = [self.stemmer.stem(word) for word in tokens]
        return stems

    def lemmatize(self,tokens):
        '''Use WordNet lemmatizer on token list'''
        # https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/
        lemmas = [self.lemmatizer.lemmatize(word,pos='v') for word in tokens]
        return lemmas

    def preprocess(self,text):
        '''Preprocess a passage'''
        tokens = self.removeStop(self.removePunc(self.tokenize(text.lower())))
        preprocessed = self.stem(tokens)
        return preprocessed

    def rudimentaryRegroup(self):
        '''Group texts into fewer, larger categories based on my rudimentary grouping that should probably be revised'''
        for i in range(len(self.data)):
            if self.data[i][1] in ['Spacecraft', 'Space Tech']:
                self.data[i] = (self.data[i][0],'Space')
            elif self.data[i][1] in ['Vehicle', 'Transportation', 'Travel', 'Miscellaneous']:
                self.data[i] = (self.data[i][0],'Transport')
            elif self.data[i][1] in ['Output Devices', 'Input Devices', 'Displays', 'Computers', 'Data Storage']:
                self.data[i] = (self.data[i][0],'Computer')
            elif self.data[i][1] in ['Armor', 'Weapon', 'Warfare', 'Security', 'Surveillance']:
                self.data[i] = (self.data[i][0],'War')
            elif self.data[i][1] in ['Lifestyle', 'Culture', 'Living Space', 'Work']:
                self.data[i] = (self.data[i][0],'Life')
            elif self.data[i][1] in ['Manufacturing', 'Material', 'Engineering', 'Clothing']:
                self.data[i] = (self.data[i][0],'Making')
            elif self.data[i][1] in ['Entertainment', 'Media', 'Communication']:
                self.data[i] = (self.data[i][0],'Communications')
            elif self.data[i][1] in ['Medical', 'Biology']:
                self.data[i] = (self.data[i][0],'Bio')
            elif self.data[i][1] in ['Artificial Intelligence', 'Virtual Person', 'Robotics']:
                self.data[i] = (self.data[i][0],'AI')

    def formatToWordFeatures(self,quantized=False):
        '''Format text as lists of word features, to fit NLTK's NB and Scikit input format'''
        '''Input passages as [(passage,label),(passage,label), ... ]'''
        '''Set quantized to True to count instances of a word in a passage, or False to do binary True/False'''
        '''Output is list of (dict, label) pairs, where dict is dictionary of the frequencies of each word in dataset'''
        # https://stackoverflow.com/questions/20827741/nltk-naivebayesclassifier-training-for-sentiment-analysis
        #self.rudimentaryRegroup()
        prepLabeledTexts = [(self.preprocess(text),label) for (text,label) in self.data]
        words = set(word for (text,label) in prepLabeledTexts for word in text)
        if quantized:
            formatted = [({word: (math.floor(math.log(text.count(word),2)) if text.count(word) > 0 else -1) for word in words}, label) for (text,label) in prepLabeledTexts]
        else:
            formatted = [({word: (word in text) for word in words}, label) for (text, label) in prepLabeledTexts]
        self.data = formatted

    def formatToTfidfFeatures(self):
        '''Find TF-IDF for each word in each document and format to fit NLTK's format'''
        '''Input passages as [(passage,label),(passage,label), ... ]'''
        '''Output is list of (dict, label) pairs, where dict is dictionary of the TF-IDFs in dataset'''
        # https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/
        #self.rudimentaryRegroup()
        prepLabeledTexts = [(self.preprocess(text), label) for (text, label) in self.data]
        vectorizer = TfidfVectorizer(ngram_range=(1,3),max_features=500,sublinear_tf=True)
        vecs = vectorizer.fit_transform([' '.join(entry[0]) for entry in prepLabeledTexts])
        wordList = vectorizer.get_feature_names()
        wordDict = {wordList[i]: i for i in range(len(wordList))}
        formatted = [({word: vecs[i,wordDict[word]] for word in wordList},prepLabeledTexts[i][1]) for i in range(len(prepLabeledTexts))]
        self.data = formatted

class Classifier:
    def __init__(self,data=[]):
        self.data = data
        self.classifier = None

    def loadClassifierFromFile(self,f):
        '''Unpickle classifier data from file'''
        file = open(f, 'rb')
        self.classifier = pickle.load(file)
        file.close()

    def writeClassifierToFile(self,f):
        '''Pickle classifier data to file'''
        file = open(f, 'wb')
        pickle.dump(self.classifier, file)
        file.close()

    def train(self,classifier,trainData):
        '''Train classifier on data with input classifier'''
        self.classifier = classifier.train(trainData)

    def test(self,testData):
        '''Test classifier on input data'''
        correct = 0
        correctByClass = {cat: 0 for cat in self.classifier.labels()}
        totalByClass = {cat: 0 for cat in self.classifier.labels()}
        for t in testData:
            if self.classifier.classify(t[0]) == t[1]:
                correct += 1
                correctByClass[t[1]] += 1
            totalByClass[t[1]] += 1
        print('Correct:', correct, '/', len(testData), '=', correct / len(testData) * 100, '%')
        print('By class:')
        for c in self.classifier.labels():
            if totalByClass[c] > 0:
                print(c, ':', correctByClass[c], '/', totalByClass[c], '=', correctByClass[c] / totalByClass[c] * 100, '%')
            else:
                print(c, ':', correctByClass[c], '/', totalByClass[c])


if __name__ == '__main__':
    print('Loading data ... ',end='')
    rawData = None
    with open('alldata.json') as f:
        rawData = json.load(f)
    print('Done')

    print('Reformatting data ... ', end='')
    fmtr = Formatter(rawData)
    fmtr.reformatJson()
    '''
    # Print statistics of document lengths for each category (mean/median length etc.)
    catInfo = {}
    for doc in fmtr.data:
        if doc[1] in catInfo:
            catInfo[doc[1]]['lens'].append(len(doc[0].split()))
        else:
            catInfo[doc[1]] = {'lens':[len(doc[0].split())]}
    import statistics
    for cat in catInfo:
        print(cat,'\t',len(catInfo[cat]['lens']),'\t',statistics.mean(catInfo[cat]['lens']),'\t',statistics.median(catInfo[cat]['lens']),'\t',min(catInfo[cat]['lens']),'\t',max(catInfo[cat]['lens']),sep='')
    '''
    fmtr.formatToTfidfFeatures()
    #fmtr.formatToWordFeatures(True)
    print('Done')

    print('Assembling training data ... ', end='')
    classifiers = {
        #'nb': {'name': 'Naive Bayes', 'classifier': NaiveBayesClassifier},
        'mnb': {'name': 'Multinomial Naive Bayes', 'classifier': MultinomialNB()},
        'bnb': {'name': 'Bernoulli Naive Bayes', 'classifier': BernoulliNB()},
        'gnb': {'name': 'Gaussian Naive Bayes', 'classifier': GaussianNB()},
        'cnb': {'name': 'Complement Naive Bayes', 'classifier': ComplementNB()},
        'catnb': {'name': 'Categorical Naive Bayes', 'classifier': CategoricalNB()},
        'lsvc': {'name': 'Linear Support Vector', 'classifier': LinearSVC(penalty='l2',multi_class='ovr',dual=False,C=0.1)},
        'svc': {'name': 'Support Vector', 'classifier': SVC(probability=True,kernel='rbf',gamma='scale',degree=4,C=1)},
        'nsvc': {'name': 'Nu Support Vector', 'classifier': NuSVC(probability=True,kernel='rbf',gamma='scale',degree=4,nu=0.7)},
        'sgd': {'name': 'Stochastic Gradient Descent', 'classifier': SGDClassifier()},
        'knc': {'name': 'K-Neighbors Classifier', 'classifier': KNeighborsClassifier(weights='uniform',p=2,n_neighbors=5,algorithm='auto')},
        'rnc': {'name': 'Radius Neighbors Classifier', 'classifier': RadiusNeighborsClassifier()},
        'gbc': {'name': 'Gradient Boosting Classifier', 'classifier': GradientBoostingClassifier(subsample=0.8,loss='deviance',learning_rate=0.1)},
        'lr': {'name': 'Logistic Regression', 'classifier': LogisticRegression(solver='liblinear',penalty='l2',multi_class='auto',fit_intercept=False,dual=True,C=0.9)},
        'mlp': {'name': 'Multi-Layer Perception', 'classifier': MLPClassifier(solver='lbfgs',learning_rate='constant',alpha=0.00001,activation='tanh',max_iter=1000)}
    }
    chosenClsfr = 'mlp'
    clsfr = classifiers[chosenClsfr]['classifier']

    data = [(list(d[0].values()), d[1]) for d in fmtr.data]
    random.shuffle(data)
    trainProportion = 0.85
    trainSize = int(len(fmtr.data) * trainProportion)
    dataTrain = [d[0] for d in data[:trainSize]]
    labelsTrain = [d[1] for d in data[:trainSize]]
    dataTest = [d[0] for d in data[trainSize:]]
    labelsTest = [d[1] for d in data[trainSize:]]
    print('Done')

    '''
    print('Finding best parameters ... ', end='')
    random_grid = {
        'activation': ['identity','logistic','tanh','relu'],
        'solver': ['lbfgs','sgd','adam'],
        'alpha': [0.00001,0.0001,0.001,0.01],
        'learning_rate': ['constant','invscaling','adaptive']
    }
    randomSearch = RandomizedSearchCV(estimator=clsfr,
                                      param_distributions=random_grid,
                                      n_iter=50,
                                      scoring='accuracy',
                                      cv=3,
                                      verbose=1,
                                      random_state=8)
    randomSearch.fit(dataTrain, labelsTrain)
    print('Done')
    print('  Best params:')
    print(' ',randomSearch.best_params_)
    print('  Score with these params:')
    print(' ',randomSearch.best_score_)
    '''

    print('Training',classifiers[chosenClsfr]['name'],'classifier ... ', end='')
    #clsfr = Classifier(fmtr.data)
    #clsfr.train(classifiers[chosenClsfr]['classifier'],fmtr.data[:trainSize])

    clsfr.fit(dataTrain,labelsTrain)
    print('Done')

    '''
    print('Saving data ... ',end='')
    #fmtr.writeDataToFile('data.pickle')
    #clsfr.writeClassifierToFile('classifierdata.pickle')
    print('Done')
    '''

    print('Performing',len(fmtr.data)-trainSize,'tests for',classifiers[chosenClsfr]['name'],'...')
    #clsfr.test(fmtr.data[trainSize:])
    predTrain = clsfr.predict(dataTrain)
    predTest = clsfr.predict(dataTest)
    print('Done')
    print('Train accuracy:', accuracy_score(labelsTrain, predTrain))
    print('Test accuracy:',accuracy_score(labelsTest,predTest))

    print(confusion_matrix(labelsTest, predTest))