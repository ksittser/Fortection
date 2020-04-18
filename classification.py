import json
import string
import math
import random
import statistics
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import NaiveBayesClassifier
from nltk import classify
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB,CategoricalNB,ComplementNB
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

random.seed(104)

class Formatter:
    def __init__(self,rawData=[]):
        self.rawData = rawData
        self.data = None
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.labels = []

    def reformatJson(self):
        '''Reformat JSON data to [(passage,label),(passage,label), ...] format'''
        data = []  # list of (passage,label) pairs
        for book in self.rawData:
            for tech in book['technologies']:
                if 'excerpt' in book['technologies'][tech] and 'category' in book['technologies'][tech]:
                    data.append((book['technologies'][tech]['excerpt'], book['technologies'][tech]['category']))

        self.data = [(text, label) for (text, label) in data]
        self.labels = list(set([d[1] for d in data]))

    def removePunc(self,tokens):
        '''Remove punctuation tokens from token list
           :param tokens: list of strings
           :returns: list of strings'''
        # https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/
        translator = str.maketrans('', '', string.punctuation)
        return ' '.join(tokens).translate(translator).split()

    def tokenize(self,text):
        '''Split passage into list of tokens
           :param text: string
           :returns: list of strings'''
        return word_tokenize(text)

    def removeStop(self,tokens):
        '''Remove stopwords from list, based on NLTK's list of stopwords
           :param tokens: list of strings
           :returns: list of strings'''
        # https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/
        stopWords = set(stopwords.words("english"))
        stopWords.update(['\n','\r'])
        filtered = [word for word in tokens if word not in stopWords]
        return filtered

    def stem(self,tokens):
        '''Stem each token, using Porter Stemmer
           :param tokens: list of strings
           :returns: list of strings, but stemmed'''
        # https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/
        stems = [self.stemmer.stem(word) for word in tokens]
        return stems

    def lemmatize(self,tokens):
        '''Lemmatize each token, using WordNet lemmatizer
           :param tokens: list of strings
           :returns: list of strings'''
        # https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/
        lemmas = [self.lemmatizer.lemmatize(word,pos='v') for word in tokens]
        return lemmas

    def preprocessPsg(self,text):
        '''Preprocess a passage
           :param text: string
           :returns: list of tokens'''
        tokens = self.removeStop(self.removePunc(self.tokenize(text.lower())))
        preprocessed = self.stem(tokens)
        return preprocessed

    def fixLabel(self,label):
        '''Fix label formatting (for instance, one category is "Artificial\r\n   Intelligence)"
           :param label: string
           :returns: string'''
        return ' '.join(self.removeStop(self.tokenize(label)))

    def recat(self,*cats,newName=None):
        '''Merge categories to make one with name newName
           :param cats: any number of strings, which are category names to merge
           :param newName: string, which is name of newly created category, or None to use first category's name'''
        if newName is None:
            newName = cats[0]
        for i in range(len(self.data)):
            if self.data[i][1] in cats:
                self.data[i] = (self.data[i][0], newName)
        for cat in cats:
            self.labels.remove(cat)
        self.labels.append(newName)

    def preprocess(self):
        '''Preprocess all passages
           In: [(passage,label),(passage,label), ... ]
           Out: [([tokens],label),([tokens],label), ... ]'''
        self.data = [(self.preprocessPsg(text), self.fixLabel(label)) for (text, label) in self.data]

    def mergeCats(self):
        '''Group texts into fewer, larger categories'''
        self.recat('Virtual Person', 'Displays', 'Input Devices', 'Media', newName='VirDispInput')
        self.recat('Spacecraft', 'Space Tech', newName='SpacecSpacet')
        self.recat('Warfare', 'Weapon', 'Security', newName='WarWeapSec')
        self.recat('Data Storage', 'Computers', 'Artificial Intelligence', newName='DataCompArtif')
        self.recat('Vehicle', 'Transportation', newName='VehTrans')
        self.recat('Manufacturing', 'Material', newName='ManufMat')
        self.recat('Armor', 'Clothing', newName='ArmCloth')
        self.recat('Living Space', 'Lifestyle', 'Entertainment', 'Travel', newName='LivLifeEntTrav')
        self.recat('Biology', 'Medical', newName='BioMedic')
        self.labels.sort()

    def formatToVecFeatures(self, vecType='tfidf'):
        '''Reformat passages as features vector
           :param vecType: 'tfidf', 'count', or 'binary', to do TF-IDF, word count, or binary word occurrence vectors
           In: [([tokens],label),([tokens],label), ... ]
           Out: [(dict, label), (dict, label), ... ] where dict is dictionary of word:value in dataset'''
        # TF-IDF from https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/
        vectorizer = None
        if vecType == 'tfidf':
            vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=2000, sublinear_tf=True)
        elif vecType == 'binary':
            vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=2000, binary=True)
        elif vecType == 'count':
            vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=2000, binary=False)
        vecs = vectorizer.fit_transform([' '.join(entry[0]) for entry in self.data])
        wordList = vectorizer.get_feature_names()
        wordDict = {wordList[i]: i for i in range(len(wordList))}
        formatted = [({word: vecs[i,wordDict[word]] for word in wordList},self.data[i][1]) for i in range(len(self.data))]
        random.shuffle(formatted)
        self.data = formatted

class Classifier:
    def __init__(self,data=[],clsfrName=None,clsfr=None):
        self.data = data
        self.classifierName = clsfrName
        self.classifier = clsfr
        self.dataTrain = None
        self.dataTest = None
        self.labelsTrain = None
        self.labelsTest = None
        self.predictionsTrain = None
        self.predictionsTest = None

    def getDocStats(self):
        '''Print document stats per class (count of docs in each class, mean/median length of docs, etc.)'''
        catInfo = {}
        for doc in self.data:
            if doc[1] in catInfo:
                catInfo[doc[1]]['lens'].append(len(doc[0].split()))
            else:
                catInfo[doc[1]] = {'lens': [len(doc[0].split())]}
        for cat in catInfo:
            print(cat, '\t', len(catInfo[cat]['lens']), '\t', statistics.mean(catInfo[cat]['lens']), '\t',
                  statistics.median(catInfo[cat]['lens']), '\t', min(catInfo[cat]['lens']), '\t',
                  max(catInfo[cat]['lens']), sep='')

    def filter(self,cats):
        '''Filter data by category, keeping only those in input categories
           :param cats: list of strings, which are category names to keep'''
        self.data = [d for d in self.data if d[1] in cats]

    def assembleData(self,k=8,i=0):
        '''Reformat data and split datasets based on k-fold cross-validation
           :param k: number of sections data is split into
           :param i: 0<=i<k, to indicate which section to assemble'''
        data = [(list(d[0].values()), d[1]) for d in self.data]
        testInterval = (len(data)*i//k, len(data)*(i+1)//k)
        self.dataTrain = [d[0] for d in data[:testInterval[0]]+data[testInterval[1]:]]
        self.labelsTrain = [d[1] for d in data[:testInterval[0]]+data[testInterval[1]:]]
        self.dataTest = [d[0] for d in data[testInterval[0]:testInterval[1]]]
        self.labelsTest = [d[1] for d in data[testInterval[0]:testInterval[1]]]

    def normalizeTrainCatSize(self,percentile):
        '''Make each category have the same size in training data, by deleting random data from categories with many
           and duplicating random data from categories with few
           :param percentile: int, give each category size that is this percentile of actual sizes of categories'''
        sortedData = {}
        trainData = zip(self.dataTrain,self.labelsTrain)
        for d in trainData:
            if d[1] in sortedData:
                sortedData[d[1]].append(d[0])
            else:
                sortedData[d[1]] = [d[0]]
        sizes = []
        for cat in sortedData:
            sizes.append(len(sortedData[cat]))
        sizes.sort()
        newSize = sizes[len(sizes)*percentile//100]
        for cat in sortedData:
            if len(sortedData[cat]) < newSize:
                for _ in range(newSize-len(sortedData[cat])):
                    sortedData[cat].append(sortedData[cat][random.randrange(len(sortedData[cat]))])
            else:
                random.shuffle(sortedData[cat])
                sortedData[cat] = sortedData[cat][:newSize]
        data = []
        for cat in sortedData:
            data += [(d,cat) for d in sortedData[cat]]
        random.shuffle(data)
        self.dataTrain = [d[0] for d in data]
        self.labelsTrain = [d[1] for d in data]

    def findBestParams(self):
        '''Use RandomizedSearchCV to determine optimal parameters for classifier
           Modify randomGrid based on the parameters of the classifier and likely values for them'''
        print('Finding best parameters ... ', end='')
        randomGrid = {
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.00001, 0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'invscaling', 'adaptive']
        }
        randomSearch = RandomizedSearchCV(estimator=self.classifier,
                                          param_distributions=randomGrid,
                                          n_iter=50,
                                          scoring='accuracy',
                                          cv=3,
                                          verbose=1,
                                          random_state=8)
        randomSearch.fit(self.dataTrain, self.labelsTrain)
        print('Done')
        print('  Best params:')
        print(' ', randomSearch.best_params_)
        print('  Score with these params:')
        print(' ', randomSearch.best_score_)

    def train(self):
        '''Train classifier'''
        if self.classifierName == 'LSTM':
            self.trainLSTM()
        elif self.classifierName == 'NLTK Naive Bayes':
            for i in range(len(self.dataTrain)):
                self.dataTrain[i] = {str(j):self.dataTrain[i][j] for j in range(len(self.dataTrain[i]))}
            self.classifier = NaiveBayesClassifier.train(zip(self.dataTrain,self.labelsTrain))
        else:
            self.classifier.fit(self.dataTrain, self.labelsTrain)

    def test(self):
        '''Test classifier'''
        res = None
        if self.classifierName == 'LSTM':
            res = self.testLSTM()
        elif self.classifierName == 'NLTK Naive Bayes':
            for i in range(len(self.dataTest)):
                self.dataTest[i] = {str(j):self.dataTest[i][j] for j in range(len(self.dataTest[i]))}
            testSet = zip(self.dataTest,self.labelsTest)
            testAcc = classify.accuracy(self.classifier, testSet)
            res = (None,testAcc)
        else:
            self.predictionsTrain = self.classifier.predict(self.dataTrain)
            self.predictionsTest = self.classifier.predict(self.dataTest)
            trainAcc = accuracy_score(self.labelsTrain, self.predictionsTrain)
            testAcc = accuracy_score(self.labelsTest, self.predictionsTest)
            res = trainAcc, testAcc
        return res

    def confMat(self,l=None):
        '''Generate confusion matrix'''
        if l is None:
            return confusion_matrix(self.labelsTest, self.predictionsTest)
        else:
            return confusion_matrix(self.labelsTest, self.predictionsTest,labels=l)

class Fortection:
    def __init__(self):
        self.formatter = None
        self.classifier = None
        self.trainAccs = {}
        self.testAccs = {}
        self.precs = {}
        self.recs = {}
        self.f1s = {}
        self.confMats = {}
        self.classifiers = {
            'nltknb': {'name': 'NLTK Naive Bayes', 'classifier': None},
            'mnb': {'name': 'Multinomial Naive Bayes', 'classifier': MultinomialNB()},
            'bnb': {'name': 'Bernoulli Naive Bayes', 'classifier': BernoulliNB()},
            'gnb': {'name': 'Gaussian Naive Bayes', 'classifier': GaussianNB()},
            'cnb': {'name': 'Complement Naive Bayes', 'classifier': ComplementNB()},
            'catnb': {'name': 'Categorical Naive Bayes', 'classifier': CategoricalNB()},
            'svc': {'name': 'Support Vector',
                    'classifier': SVC(probability=True, kernel='rbf', gamma='scale', degree=4, C=1)},
            'lsvc': {'name': 'Linear Support Vector',
                     'classifier': LinearSVC(penalty='l2', multi_class='ovr', dual=False, C=0.1)},
            'nsvc': {'name': 'Nu Support Vector',
                     'classifier': NuSVC(probability=True, kernel='rbf', gamma='scale', degree=4, nu=0.7)},
            'sgd': {'name': 'Stochastic Gradient Descent', 'classifier': SGDClassifier()},
            'knc': {'name': 'K-Neighbors Classifier',
                    'classifier': KNeighborsClassifier(weights='uniform', p=2, n_neighbors=5, algorithm='auto')},
            'rnc': {'name': 'Radius Neighbors Classifier', 'classifier': RadiusNeighborsClassifier()},
            'gbc': {'name': 'Gradient Boosting Classifier',
                    'classifier': GradientBoostingClassifier(subsample=0.8, loss='deviance', learning_rate=0.1)},
            'lr': {'name': 'Logistic Regression',
                   'classifier': LogisticRegression(solver='liblinear', penalty='l2', multi_class='auto',
                                                    fit_intercept=False, dual=True, C=0.9)},
            'mlp': {'name': 'Multi-Layer Perception',
                    'classifier': MLPClassifier(solver='lbfgs', learning_rate='constant', alpha=0.00001,
                                                activation='tanh', max_iter=1000)}
        }

    def getData(self,fileName=None,featureType=None):
        '''Reformat data to feature vectors
           :param fileName: file to load feature vectors from; or None to format from scratch
           :param featureType: 'tfidf', 'count', or 'binary', to format from scratch; None to load from file'''
        print('Loading data ... ', end='')
        if fileName is not None:
            # load from file
            f = open('tfidf.pickle', 'rb')
            self.formatter = pickle.load(f)
            f.close()
        elif featureType is not None:
            # generate from scratch
            with open('alldata.json') as f:
                rawData = json.load(f)
            print('Done')
            print('Reformatting data ... ', end='')
            self.formatter = Formatter(rawData)
            self.formatter.reformatJson()
            self.formatter.preprocess()
            self.formatter.formatToVecFeatures('tfidf')
            # self.formatter.formatToWordFeatures(True,True)
            f = open('tfidf.pickle', 'wb')
            pickle.dump(self.formatter, f)
            f.close()
        else:
            pass
        print('Done')

    def kFoldCrossTest(self,chosenClsfrs,catList,mergeCats=False,normCatSize=False,numSections=8,numTests=8):
        '''k-fold cross testing on data
           :param chosenClsfrs: list of strings, which indicate which classifiers to try
           :param catList: list of strings, which are categories to filter by; data from other categories are removed
           :param mergeCats: boolean, whether categories should be merged based on scheme in Classifier.mergeCats()
           :param normCatSize: boolean, whether categories should be undersampled and oversampled to make them same size
           :param showConfMat: boolean, whether a confusion matrix should be printed
           :param sections: int, how many sections to partition data into for k-fold cross test
           :param tests: int, 0<tests<=sections, how many train/test cycles to actually perform'''

        if mergeCats:
            self.formatter.mergeCats()
        for clsfr in chosenClsfrs:
            # for each classifier
            clsfrType = self.classifiers[clsfr]['name']
            print('Beginning classification for ',clsfrType,':',sep='')
            confMat = 0
            trainAccs, testAccs, precs, recs, f1s = [], [], [], [], []
            cats = catList
            classifier = Classifier(self.formatter.data, clsfrType, self.classifiers[clsfr]['classifier'])
            classifier.filter(cats)
            for i in range(numTests):
                print('(',i+1,'/',numTests,')',sep='',end=' ')
                print('Assembling ... ', end='')
                classifier.assembleData(numSections, i)
                if normCatSize:
                    classifier.normalizeTrainCatSize(92)
                # classifier.findBestParams()
                print('Training ... ', end='')
                classifier.train()
                print('Testing ... ', end='')
                trainAcc, testAcc = classifier.test()
                trainAccs.append(trainAcc)
                testAccs.append(testAcc)
                precision, recall, f1Score, _ = precision_recall_fscore_support(classifier.labelsTest,
                                                                                classifier.predictionsTest, labels=cats)
                precs.append(precision)
                recs.append(recall)
                f1s.append(f1Score)

                if confMat is None:
                    confMat = classifier.confMat(cats)
                else:
                    confMat += classifier.confMat(cats)
                print('Done')
            self.trainAccs[clsfr] = trainAccs
            self.testAccs[clsfr] = testAccs
            self.precs[clsfr] = precs
            self.recs[clsfr] = recs
            self.f1s[clsfr] = f1s
            self.confMats[clsfr] = confMat

    def showResults(self,showAccs=True,showPrecRecs=False,showF1s=True,showConfMat=False,showBestPairings=False):
        '''Print results for each classifier
           :param showAccs: boolean, whether to show mean and median accuracy scores for train and test data
           :param showPrecRecs: boolean, whether to show precisions and recalls for test data
           :param showF1s: boolean, whether to show F1 scores for test data
           :param showConfMat: boolean, whether to show confusion matrices
           :param showBestPairings: boolean, whether to show a list of best category merging possibilities'''
        #print(len(self.formatter.labels), 'cats:', '\t'.join(self.formatter.labels))
        for clsfr in self.trainAccs.keys():
            print('\n==================================================\n')
            print('Showing results for:',self.classifiers[clsfr]['name'])
            print()
            if showAccs:
                print('Mean Train Acc:', statistics.mean(self.trainAccs[clsfr]))
                print('  Median Train Acc:', statistics.median(self.trainAccs[clsfr]))
                print('Mean Test Acc:', statistics.mean(self.testAccs[clsfr]))
                print('  Median Test Acc:', statistics.median(self.testAccs[clsfr]))
            if showPrecRecs:
                precs = self.precs[clsfr]
                avgPrecs = [statistics.mean([precs[i][j] for i in range(len(precs))]) for j in range(len(precs[0]))]
                print('Precisions:', avgPrecs)
                print('  Mean Precision:', statistics.mean(avgPrecs))
                print('  Median Precision:', statistics.median(avgPrecs))
                recs = self.recs[clsfr]
                avgRecs = [statistics.mean([recs[i][j] for i in range(len(recs))]) for j in range(len(recs[0]))]
                print('Recalls:', avgRecs)
                print('  Mean Recall:', statistics.mean(avgRecs))
                print('  Median Recall:', statistics.median(avgRecs))
            if showF1s:
                f1s = self.f1s[clsfr]
                avgF1s = [statistics.mean([f1s[i][j] for i in range(len(f1s))]) for j in range(len(f1s[0]))]
                print('F1 Scores:', avgF1s)
                print('  Mean F1:', statistics.mean(avgF1s))
                print('  Median F1:', statistics.median(avgF1s))
            if showConfMat:
                print('\nConfusion Matrix (Percentages):')
                cats = self.formatter.labels
                print(' '.rjust(8),end='\t')
                print('\t'.join([c[:8].rjust(8) for c in cats]))
                for cat,row in zip(cats,self.confMats[clsfr]):
                    sum = 0
                    for d in row:
                        sum += d
                    print(cat[:8].ljust(8),end='\t')
                    print('\t'.join(['%8.4f'%(d/sum*100) for d in row]))

            if showBestPairings:
                print('\nBest pairings:')
                res = []
                for row in range(len(self.confMats[clsfr])):
                    sum = 0
                    for col in range(len(self.confMats[clsfr][row])):
                        sum += self.confMats[clsfr][row][col]
                    for col in range(len(self.confMats[clsfr][row])):
                        if row != col:
                            res.append((self.confMats[clsfr][row][col]/sum*100, row, col))
                res.sort(reverse=True)
                cats = self.formatter.labels
                for i in range(25):
                    print(cats[res[i][1]][:8].ljust(8), cats[res[i][2]][:8].ljust(8), '%8.4f'%(res[i][0]), sep='   ')
                print('(',str(len(self.formatter.labels)),' Categories: ',', '.join(self.formatter.labels),')',sep='')

if __name__ == '__main__':
    fortection = Fortection()

    # Uncomment exactly one of these two lines
    loadFile = 'tfidf.pickle'  # Load feature vectors from a file
    #featureType = 'tfidf'  # Build feature vectors from scratch

    fortection.getData(fileName=loadFile, featureType=None)

    classifiers = ['lsvc', 'lr']  # List as many classifiers as you want; see abbreviations in Fortection.kFoldCrossTest()
    # Uncomment one of these lines based on which categories' data to use
    labelList = fortection.formatter.labels  # Use all data
    #labelList = ['Engineering', 'Medical', 'Robotics', 'Weapon']
    #labelList = ['Artificial Intelligence','Biology','Communication','Computers','Culture','Displays','Engineering','Lifestyle','Living Space','Material','Medical','Robotics','Space Tech','Spacecraft','Surveillance','Transportation','Vehicle','Weapon']

    mergeCats = True  # Specify whether to merge categories based on scheme in Formatter.mergeCats()
    normCatSize = True  # Specify whether to undersample and oversample so that categories have same amount of data
    numSections = 8  # Specify how many partitions of data for k-fold cross testing
    numTests = 8  # Specify how many train/test cycles to actually do in k-fold cross testing

    fortection.kFoldCrossTest(chosenClsfrs=classifiers,catList=fortection.formatter.labels,mergeCats=mergeCats,
                                  normCatSize=normCatSize,numSections=numSections,numTests=numTests)

    showAccs = True  # Specify whether to show accuracy scores
    showPrecRecs = True  # Specify whether to show precision and recall scores
    showF1s = True  # Specify whether to show F1 scores
    showConfMats = False  # Specify whether to show confusion matrices
    showBestPairings = False  # Specify whether to show best pairings for category merging

    fortection.showResults(showAccs=showAccs,showPrecRecs=showPrecRecs,showF1s=showF1s,showConfMat=showConfMats,
                           showBestPairings=showBestPairings)