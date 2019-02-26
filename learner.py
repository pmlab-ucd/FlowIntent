import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.naive_bayes import BernoulliNB
import pydotplus
from statistics import *
import pickle
from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy as np
from sklearn.linear_model import LogisticRegression
from time import time
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import set_logger

import string
import random

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
from nltk.stem.porter import PorterStemmer
import jieba

import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
logger = set_logger('Learner')


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([PorterStemmer().stem(w) for w in analyzer(doc)])


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: ([PorterStemmer().stem(w) for w in analyzer(doc)])


class Learner:
    class LabelledDocs:
        @staticmethod
        def stem_tokens(tokens, stemmer):
            return [stemmer.stem(item) for item in tokens]

        @staticmethod
        def vectorize(instances, vec=None, tf=False, ngrams_range=None):
            # Initialize the "CountVectorizer" object, which is scikit-learn's
            # bag of words tool.
            if vec is not None:
                train_data = vec.transform(instances)
                vocab = vec.get_feature_names()
                return train_data, vocab, vec
            if not tf:
                if ngrams_range is None:
                    vectorizer = StemmedCountVectorizer(analyzer="word",
                                                        tokenizer=None,
                                                        preprocessor=None,
                                                        stop_words=['http'])
                else:
                    vectorizer = StemmedCountVectorizer(analyzer='char_wb',
                                                        tokenizer=None,
                                                        preprocessor=None,
                                                        stop_words=['http'],
                                                        ngram_range=ngrams_range)
            else:
                if ngrams_range is None:
                    vectorizer = StemmedTfidfVectorizer(analyzer="word",
                                                        tokenizer=None,
                                                        preprocessor=None,
                                                        stop_words=['http'])
                else:
                    vectorizer = StemmedTfidfVectorizer(analyzer='char_wb',
                                                        tokenizer=None,
                                                        preprocessor=None,
                                                        stop_words=None,
                                                        ngram_range=ngrams_range)
            # fit_transform() does two functions: First, it fits the model
            # and learns the vocabulary; second, it transforms our training data
            # into feature vectors. The input to fit_transform should be a list of
            # strings.
            train_data = vectorizer.fit_transform(instances)

            # Numpy arrays are easy to work with, so convert the result to an
            # array
            # train_data = train_data.toarray()
            logger.info(train_data.shape)
            # Take a look at the words in the vocabulary
            vocab = vectorizer.get_feature_names()
            # log.info(vocab)
            # train_data, labels = Learner.feature_filter_by_prefix(vocab, docs)

            return train_data, vocab, vectorizer

        @staticmethod
        def tokenize(text: str) -> [str]:
            """
            Tokenize the given text.
            :param text: a string
            :return: The vocabulary.
            """
            vectorizer = CountVectorizer(analyzer='word')
            vectorizer.fit_transform([text])
            tokens = vectorizer.get_feature_names()
            # stems = self.stem_tokens(tokens, stemmer)
            return tokens

        def __init__(self, doc, label, numeric_features=None, real_label=None, char_wb=False,
                     filtered_words: [] = None):
            """
            Create a instance representing a text string to be used by ML.
            :param doc: The text string.
            :param label: The assigned label for training.
            :param numeric_features: The numeric feature values.
            :param real_label: The ground truth label of this instance.
            :param char_wb: Used to n-grams only from characters inside word boundaries (padded with space on each side)
            :param filtered_words: Filter doc based on the given list.
            https://scikit-learn.org/stable/modules/feature_extraction.html#limitations-of-the-bag-of-words-representation
            """
            if filtered_words is not None:
                for word in filtered_words:
                    doc = doc.replace(word, ' ')
            self.doc = doc
            self.label = label
            self.real_label = real_label
            self.numeric_features = numeric_features
            tokens = self.tokenize(doc)
            if char_wb:
                self.doc = ''.join(tokens)
            else:
                self.doc = ' '.join(tokens)

    @staticmethod
    def same_prefix(str_a, str_b):
        for i, c in enumerate(str_a):
            if i > 6:
                return True
            if c == str_b[i]:
                continue
            else:
                return False

    @staticmethod
    def feature_filter_by_prefix(vocab, docs):
        examined = []
        for i in range(len(vocab)):
            logger.info('i: ' + vocab[i] + ' ' + str(i))
            if len(vocab[i]) < 6 or vocab[i] in examined:
                continue
            for j in range(i + 1, len(vocab)):
                # log.info('j: ' + vocab[j] + ' ' + str(j))
                if len(vocab[j]) < 6:
                    examined.append(vocab[j])
                    continue
                if vocab[i] in vocab[j] or vocab[j] in vocab[i]:  # Learner.same_prefix(vocab[i], vocab[j]):
                    # log.info('Found ' + vocab[i] + ' ' + vocab[j] + ' ' + str(i))
                    examined.append(vocab[j])
                    for doc in docs:
                        if vocab[j] in doc.doc:
                            doc.doc = str(doc.doc).replace(vocab[j], vocab[i])
        instances = []
        labels = []
        for doc in docs:
            instances.append(doc.doc)
            labels.append(doc.label)
        vectorizer = StemmedCountVectorizer(analyzer="word",
                                            tokenizer=None,
                                            preprocessor=None,
                                            stop_words=None)
        train_data = vectorizer.fit_transform(instances)

        # Numpy arrays are easy to work with, so convert the result to an
        # array
        # train_data = train_data.toarray()
        logger.info(train_data.shape)
        return train_data, labels

    @staticmethod
    def ocsvm(train_data, labels, n_fold=0):
        nu = float(np.count_nonzero(labels == -1)) / len(labels)
        clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=0.1)
        results = None
        if n_fold != 0:
            folds = Learner.n_folds(train_data, labels, fold=n_fold)
            results = Learner.cross_validation(clf, train_data, labels, folds=folds)
            # simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'),
            # separators=(',', ':'), sort_keys=True, indent=4)
            logger.info('OCSVM: ' + str(results['duration']))
            logger.info('mean scores:' + str(results['mean_scores']))
            logger.info('mean_conf:' + str(results['mean_conf_mat']))

        clf.fit(train_data)

        return clf, results

    @staticmethod
    def train_bayes(train_data, labels, n_fold=0):
        clf = BernoulliNB()
        results = None
        if n_fold != 0:
            folds = Learner.n_folds(train_data, labels, fold=n_fold)
            results = Learner.cross_validation(clf, train_data, labels, folds=folds)
            # simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'),
            # separators=(',', ':'), sort_keys=True, indent=4)
            logger.info('Bayes: ' + str(results['duration']))
            logger.info('mean scores:' + str(results['mean_scores']))
            logger.info('mean_conf:' + str(results['mean_conf_mat']))

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        clf = clf.fit(train_data, labels)

        return clf, results

    @staticmethod
    def class_report(conf_mat):
        tp, fp, fn, tn = conf_mat.flatten()
        return Learner.measure(tp, fp, tn, fn)

    @staticmethod
    def measure(tp, fp, tn, fn):
        measures = {'accuracy': (tp + tn) / (tp + fp + fn + tn), 'fp_rate': fp / (tn + fp), 'recall': tp / (tp + fn),
                    'precision': tp / (tp + fp), 'f1score': 2 * tp / (2 * tp + fp + fn)}
        # measures['tn_rate'] = tn / (tn + fp)  # (true negative rate)
        return measures

    @staticmethod
    def n_folds(data, labels, shuffle=True, fold=5):
        """
        Split the dataset into $fold folds.
        :param data:
        :param labels:
        :param shuffle: whether shuffle the original dataset
        :param fold: the number of folds
        :return: A list of splited datasets
        """
        X = data
        y = labels
        ''' Run x-validation and return scores, averaged confusion matrix, and df with false positives and negatives '''
        # cv = KFold(n_splits=5, shuffle=True)
        # I generate a KFold in order to make cross validation
        kf = StratifiedKFold(n_splits=fold, shuffle=shuffle, random_state=42)
        folds = {}
        for fold_index, (train_index, test_index) in enumerate(kf.split(X, y)):
            fold = dict()
            fold['train_index'] = train_index
            fold['test_index'] = test_index
            # fold['X_train'], fold['X_test'] = X[train_index], X[test_index]
            # fold['y_train'], fold['y_test'] = y[train_index], y[test_index]
            folds[fold_index] = fold
        return folds

    @staticmethod
    def cross_validation(clf, X, y, folds, shuffle=True, scoring='f1', n=5):
        t0 = time()
        results = dict()
        scores = []
        conf_mat = np.zeros((2, 2))  # Binary classification

        # I start the cross validation
        results['fold'] = []
        for fold_id in folds:
            result = dict()
            fold = folds[fold_id]
            train_index = fold['train_index']
            test_index = fold['test_index']
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # I train the classifier
            clf.fit(X_train, y_train)

            # I make the predictions
            predicted = clf.predict(X_test)
            # print(fold)
            # print(predicted)
            y_plabs = np.squeeze(predicted)
            if hasattr(clf, 'predict_proba'):
                y_pprobs = clf.predict_proba(X_test)  # Predicted probability
                result['roc'] = metrics.roc_auc_score(y_test, y_pprobs[:, 1])
            else:  # for SVM
                y_decision = clf.decision_function(X_test)
                if not isinstance(clf, svm.OneClassSVM):
                    result['roc'] = metrics.roc_auc_score(y_test, y_decision[:, 1])
                else:  # OCSVM
                    result['roc'] = metrics.roc_auc_score(y_test, y_decision)
            # metrics.roc_curve(y_test, y_pprobs[:, 1])
            scores.append(result['roc'])

            # Learner.perf_measure(predicted, y_test)

            # obtain the accuracy of this fold
            # ac = accuracy_score(predicted, y_test)

            # obtain the confusion matrix
            confusion = metrics.confusion_matrix(y_test, predicted)
            conf_mat += confusion
            result['conf_mat'] = confusion.tolist()

            # collect indices of false positive and negatives, effective only shuffle=False, or backup the original data
            if not isinstance(clf, svm.OneClassSVM):
                fp_i = np.where((y_plabs == 1) & (y_test == 0))[0]
                fn_i = np.where((y_plabs == 0) & (y_test == 1))[0]
                result['fp_item'] = test_index[fp_i]
                result['fn_item'] = test_index[fn_i]
                # print(result['fp_item'])
                # print(result['fn_item'])
            results['fold'].append(result)

        # cv_res = cross_val_score(clf, data, labels, cv=cv, scoring='f1').tolist()
        # simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'),
        # separators=(',', ':'), sort_keys=True, indent=4)
        duration = time() - t0
        results['duration'] = duration
        # results['cv_res'] = cv_res
        # results['cv_res_mean'] = sum(cv_res) / n_splits

        # print "\nMean score: %0.2f (+/- %0.2f)" % (np.mean(scores), np.std(scores) * 2)
        results['mean_scores'] = np.mean(scores)
        results['std_scores'] = np.std(scores)
        conf_mat /= n
        # print "Mean CM: \n", conf_mat

        # print "\nMean classification measures: \n"
        results['mean_conf_mat'] = Learner.class_report(conf_mat)
        # return scores, conf_mat, {'fp': sorted(false_pos), 'fn': sorted(false_neg)}
        logger.info(str(clf) + ': ' + str(results['duration']))
        logger.info('mean scores:' + str(results['mean_scores']))
        logger.info('mean_conf:' + str(results['mean_conf_mat']))
        return results

    @staticmethod
    def train_svm(train_data, labels, n_fold=5):
        clf = svm.SVC(class_weight='balanced', probability=True)
        results = None
        if n_fold != 0:
            folds = Learner.n_folds(train_data, labels, fold=n_fold)
            results = Learner.cross_validation(clf, train_data, labels, folds=folds)
            # simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'),
            # separators=(',', ':'), sort_keys=True, indent=4)
            logger.info('SVM: ' + str(results['duration']))
            logger.info('mean scores:' + str(results['mean_scores']))
            logger.info('mean_conf:' + str(results['mean_conf_mat']))

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        clf = clf.fit(train_data, labels)

        return clf, results

    @staticmethod
    def train_logistic(train_data, labels, n_fold=5):
        clf = LogisticRegression(class_weight='balanced')
        results = None
        if n_fold != 0:
            folds = Learner.n_folds(train_data, labels, fold=n_fold)
            results = Learner.cross_validation(clf, train_data, labels, folds=folds)
            # simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'),
            # separators=(',', ':'), sort_keys=True, indent=4)
            logger.info('Logistic: %s', str(results['duration']))
            logger.info('mean scores: %s', str(results['mean_scores']))
            logger.info('mean_conf:%s ', str(results['mean_conf_mat']))

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        clf = clf.fit(train_data, labels)
        return clf, results

    @staticmethod
    def train_tree(train_data, labels, n_fold=5, res=None, output_dir=None, tree_name='tree'):
        clf = DecisionTreeClassifier(class_weight='balanced')
        results = None
        if n_fold != 0:
            print(n_fold)
            folds = Learner.n_folds(train_data, labels, fold=n_fold)
            results = Learner.cross_validation(clf, train_data, labels, folds=folds)
            # simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'),
            # separators=(',', ':'), sort_keys=True, indent=4)
            logger.info('Tree: %.2f', results['duration'])
            logger.info('mean scores: %.2f', results['mean_scores'])
            logger.info('mean_conf: %.2f', results['mean_conf_mat'])

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        clf = clf.fit(train_data, labels)
        if output_dir is not None:
            tree.export_graphviz(clf, out_file=output_dir + '/' + tree_name + '.dot',
                                 # feature_names=feature_names,
                                 label='root', impurity=False, special_characters=True)  # , max_depth=5)
            dot_file = open(output_dir + '/' + tree_name + '.dot', 'r')
            graph = pydotplus.graph_from_dot_data(dot_file.read())
            graph.write_pdf(output_dir + '/' + tree_name + '.pdf')
            dot_file.close()

        if res is not None:
            res['tree'] = results
        return clf, results

    @staticmethod
    def train_classifier(func, X, y, cv, result_dict, tag):
        result_dict[tag] = func(X, y, cv)

    @staticmethod
    def rand_str(size=6, chars=string.ascii_uppercase + string.digits):
        url = ''.join(random.choice(chars) for _ in range(size))
        if url[0] < 'k':
            url = url + 'net'
        else:
            url = url + 'com'
        url = 'www.' + url
        return url

    @staticmethod
    def simulate_flows(size, label):
        docs = []
        for _ in range(size):
            docs.append(Learner.LabelledDocs('www.' + Learner.rand_str() + '', label))
        return docs

    @staticmethod
    def tree_info(clf):
        info = dict()
        n_nodes = clf.tree_.node_count
        # children_left = clf.tree_.children_left
        # children_right = clf.tree_.children_right
        # feature = clf.tree_.max_features
        # n_feature = clf.tree_.n_features_
        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        depth = clf.tree_.max_depth
        info['n_nodes'] = n_nodes
        info['depth'] = depth
        logger.info(info)
        return info

    @staticmethod
    def predict(model, vec, instances, labels=None, src_name='', model_name=''):
        # loaded_vec = CountVectorizer(decode_error="replace", vocabulary=voc)
        data = vec.transform(instances)
        y_1 = model.predict(data)

        # log.info(y_1)
        if labels is not None:
            return accuracy_score(labels, y_1)

    @staticmethod
    def feature_selection(X, y, k, count_vectorizer, instances, tf=False, ngram_range=None):
        ch2 = SelectKBest(chi2, k=k)
        X_new = ch2.fit_transform(X, y)
        feature_names = count_vectorizer.get_feature_names()
        if feature_names:
            feature_names = [feature_names[i] for i
                             in ch2.get_support(indices=True)]
        '''
        dict = np.asarray(count_vectorizer.get_feature_names())[ch2.get_support()]
        if tf:
            if ngram_range is not None:
                count_vectorizer = StemmedTfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, vocabulary=dict)
            else:
                count_vectorizer = StemmedTfidfVectorizer(analyzer='char_wb', vocabulary=dict)
        else:
            if ngram_range is not None:
                count_vectorizer = StemmedCountVectorizer(analyzer='word', vocabulary=dict, ngram_range=ngram_range)
            else:
                count_vectorizer = StemmedCountVectorizer(analyzer="word", vocabulary=dict)
        X_new = count_vectorizer.fit_transform(text_fea)
        # cPickle.dump(count_vectorizer.vocabulary, open(output_dir + '/' + "vocabulary.pkl", "wb"))
        '''
        return X_new, feature_names, ch2

    @staticmethod
    def pipe_feature_selection(X, y):
        clf = Pipeline([
            ('feature_selection', SelectKBest(chi2, k=2).fit_transform(X, y)),
            ('classification', RandomForestClassifier())
        ])
        clf.fit(X, y)

    @staticmethod
    def save2file(obj, path):
        # save the obj
        with open(path, 'wb') as fid:
            pickle.dump(obj, fid)

    @staticmethod
    def obj_from_file(path):
        return pickle.load(open(path, 'rb'))

    @staticmethod
    def chinese(content):
        """
        判断是否是中文需要满足u'[\u4e00-\u9fa5]+'，
        需要注意如果正则表达式的模式中使用unicode，那么
        要匹配的字符串也必须转换为unicode，否则肯定会不匹配。
        """
        zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
        return zhPattern.search(content)

    @staticmethod
    def str2words(string):
        jieba.load_userdict('user.dict')
        string = re.sub('°', 'DegreeMark', string)
        if Learner.chinese(string):
            logger.debug('Chinese Detected!')
            # string = re.sub(u'[^\u4e00-\u9fa5]', '', string)
            words = jieba.cut(string, cut_all=False)
            # words = [w for w in words if not w in stopwords.words("chinese")]
        else:
            logger.debug('English Detected!')
            string = re.sub('[^a-zA-Z]', ' ', string)  # if English only
            words = string.lower().split()
            # words = [w for w in words if not w in stopwords.words("english")]
            logger.debug(words)

        word_list = []
        for word in words:
            word_list.append(word)
        # return ' '.join(words)
        return word_list

    @staticmethod
    def stat_fea_cal(x):
        return [min(x), max(x), mean(x), median(x), s_dev(x), skewness(x), kurtosis(x)]

    @staticmethod
    def voting(clfs, X, y, folds: {}):
        """
        Let each classifier train on the n-1 folds and predict on the rest fold.
        Then for each fold, pick the text_fea where all clfs say pos/neg.
        :param clfs:
        :param X:
        :param y:
        :param folds:
        :return:
        """
        t0 = time()
        results = dict()
        scores = []
        conf_mat = np.zeros((2, 2))  # Confusion matrix for binary classification

        # Start the cross validation
        for clf in clfs:
            clf_name = type(clf).__name__
            results[clf_name] = {}
            results[clf_name]['folds'] = {}
            logger.debug('clf: %s', clf_name)
            for fold_id in folds:
                result = dict()
                fold = folds[fold_id]
                train_index = fold['train_index']
                test_index = fold['test_index']
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # Train the classifier
                clf.fit(X_train, y_train)

                # Make the predictions
                predicted = clf.predict(X_test)
                y_plabs = np.squeeze(predicted)
                result['predicted_1'] = test_index[np.where(y_plabs == 1)[0]]
                result['predicted_0'] = test_index[np.where(y_plabs == 0)[0]]
                if hasattr(clf, 'predict_proba'):
                    y_pprobs = clf.predict_proba(X_test)  # Predicted probability
                    result['roc'] = metrics.roc_auc_score(y_test, y_pprobs[:, 1])
                else:  # for SVM
                    y_decision = clf.decision_function(X_test)
                    if not isinstance(clf, svm.OneClassSVM):
                        result['roc'] = metrics.roc_auc_score(y_test, y_decision[:, 1])
                    else:  # OCSVM
                        result['roc'] = metrics.roc_auc_score(y_test, y_decision)
                # metrics.roc_curve(y_test, y_pprobs[:, 1])
                scores.append(result['roc'])

                # Learner.perf_measure(predicted, y_test)

                # Obtain the accuracy of this fold
                # ac = accuracy_score(predicted, y_test)

                # Obtain the confusion matrix
                confusion = metrics.confusion_matrix(y_test, predicted)
                conf_mat += confusion
                result['conf_mat'] = confusion.tolist()

                # Collect indices of false positive and negatives, effective only shuffle=False,
                # or backup the original data
                if not isinstance(clf, svm.OneClassSVM):
                    fp_i = np.where((y_plabs == 1) & (y_test == 0))[0]
                    fn_i = np.where((y_plabs == 0) & (y_test == 1))[0]
                    result['fp_item'] = test_index[fp_i]
                    result['fn_item'] = test_index[fn_i]
                results[clf_name]['folds'][fold_id] = result
                logger.debug('fold: ')
                fps = result['fp_item']
                logger.debug("FP: %s", str(fps))
                fns = result['fn_item']
                logger.debug("FN: %s", str(fns))
            # cv_res = cross_val_score(clf, data, labels, cv=cv, scoring='f1').tolist()
            # simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'),
            # separators=(',', ':'), sort_keys=True, indent=4)
            duration = time() - t0
            results[clf_name]['duration'] = duration
            # results['cv_res'] = cv_res
            # results['cv_res_mean'] = sum(cv_res) / n_splits

            # print "\nMean score: %0.2f (+/- %0.2f)" % (np.mean(scores), np.std(scores) * 2)
            results[clf_name]['mean_scores'] = np.mean(scores)
            results[clf_name]['std_scores'] = np.std(scores)
            conf_mat /= len(folds)
            logger.info("Mean CM: %s \n", str(conf_mat))

            logger.info("\nMean classification measures: \n")
            results[clf_name]['mean_conf_mat'] = Learner.class_report(conf_mat)
            # return scores, conf_mat, {'fp': sorted(false_pos), 'fn': sorted(false_neg)}
            logger.info('duration of %s: %0.2f', str(clf), results[clf_name]['duration'])
            logger.info('mean scores: %0.2f', results[clf_name]['mean_scores'])
            logger.info('mean_conf: %s', str(results[clf_name]['mean_conf_mat']))

        def overlap_pred(fold_id, label):
            res = set()
            for j in range(0, len(clfs)):
                c_name = type(clfs[j]).__name__
                s = set(results[c_name]['folds'][fold_id]['predicted_' + label])
                if j == 0:
                    res = s
                else:
                    res = res.intersection(s)
            return res

        tp_v = []
        fp_v = []
        tn_v = []
        fn_v = []
        overlap_pred_ind = set()
        for fold_id in folds:
            overlap_predicted_pos_i = overlap_pred(fold_id, '1')
            overlap_predicted_neg_i = overlap_pred(fold_id, '0')
            overlap_pred_ind = overlap_pred_ind.union(overlap_predicted_neg_i)
            overlap_pred_ind = overlap_pred_ind.union(overlap_predicted_pos_i)
            # logger.debug(len(overlap_predicted_pos_i))
            folds[fold_id]['vot_pred_pos'] = [int(i) for i in overlap_predicted_pos_i]
            folds[fold_id]['vot_pred_neg'] = [int(i) for i in overlap_predicted_neg_i]
            for neg in overlap_predicted_neg_i:
                if y[neg] == 1:
                    fn_v.append(neg)
                else:
                    tn_v.append(neg)
            for pos in overlap_predicted_pos_i:
                if y[pos] == 0:
                    fp_v.append(pos)
                else:
                    tp_v.append(pos)
        results['voting'] = {}
        results['voting']['tp'] = tp_v
        results['voting']['fp'] = fp_v
        results['voting']['tn'] = tn_v
        results['voting']['fn'] = fn_v
        results['voting']['all'] = overlap_pred_ind
        results['voting']['conf_mat'] = Learner.measure(len(tp_v), len(fp_v), len(tn_v), len(fn_v))
        logger.info('voting conf_mat: %s', str(results['voting']['conf_mat']))
        return results


if __name__ == '__main__':
    logger.setLevel(20)
