from AppInspector.context import Context, Object, contexts
import json
import os
from learner import Learner
import learner
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils import set_logger, file_name_no_ext, handle_remove_readonly
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import shutil
import pandas as pd
import logging
from multiprocessing import Manager, Pool
from argparse import ArgumentParser

logger = set_logger('ContextProcessor')


class ContextProcessor:
    """
    Gather the labelled app contexts and build the ML models.
    """
    @staticmethod
    def preprocess(dir_path: str) -> None:
        """
        If the sens_http_flows.json/xml does not have the corresponding png, del it.
        :param dir_path:
        """

        def del_file(ext: str):
            path = os.path.join(root, base_name + ext)
            if os.path.exists(path):
                os.remove(path)
                logger.info('rm %s', path)

        for root, dirs, files in os.walk(dir_path):
            for d in dirs:
                d = os.path.join(root, d)
                if not any(fname.endswith('.png') for fname in os.listdir(d)):
                    logger.info('rm %s', d)
                    shutil.rmtree(d, ignore_errors=False, onerror=handle_remove_readonly)
            for file_name in files:
                if not file_name.endswith('.xml'):
                    continue
                base_name = file_name_no_ext(os.path.join(root, file_name))
                if os.path.exists(os.path.join(root, base_name + '.png')):
                    continue
                del_file('.xml')
                del_file('.json')
                del_file('.pcap')
                del_file('_sens_http_flows.json')

    @staticmethod
    def docs(instances: [Context], additional_doc: [str] = None) -> [[], []]:
        """
        Convert SharingInstances into the <string, label> pairs.
        :param instances:
        :param additional_doc
        :return:
        """
        docs = []
        labels = []
        for instance in instances:
            doc = []
            for string in instance.ui_doc:
                doc.append(' '.join(Learner.str2words(str(string))))
            if instance.topic is not '':
                doc.append('t_' + instance.topic)  # use another feature space
            if instance.app_name is not '':
                doc.append(' '.join(['n_' + i for i in Learner.str2words(instance.app_name)]))
            for d in additional_doc:
                doc.append(d)
            docs.append(' '.join(doc))
            labels.append(int(instance.label))
        return docs, np.array(labels)

    @staticmethod
    def subprocess_mp_wrapper(args):
        return ContextProcessor.subprocess(*args)

    @staticmethod
    def subprocess(dir_path, instances):
        for root, dirs, files in os.walk(dir_path):
            for file_name in files:
                if not file_name.endswith('.json'):
                    continue
                with open(os.path.join(root, file_name), 'r', encoding="utf8", errors='ignore') as my_file:
                    instance = json.load(my_file)
                    instances.append(instance)
                    logger.debug(instance['dir'])

    @staticmethod
    def process(root_dir, pos_dir_name='1', neg_dir_name='0', reset_out_dir=False, sub_dir_name=''):
        """
        Given the dataset of legal and illegal sharing text_fea
        Perform cross-validation on them
        :param root_dir:
        :param pos_dir_name:
        :param neg_dir_name:
        :param reset_out_dir:
        :param sub_dir_name:
        """
        # Load the contexts stored in the hard disk.
        contexts_dir = os.path.join('data', sub_dir_name)  # output dir
        pos_dir = os.path.join(root_dir, pos_dir_name)
        pos_out_dir = os.path.join(contexts_dir, pos_dir_name)
        neg_dir = os.path.join(root_dir, neg_dir_name)
        neg_out_dir = os.path.join(contexts_dir, neg_dir_name)
        if reset_out_dir:
            shutil.rmtree(contexts_dir)
        if not os.path.exists(contexts_dir):
            os.makedirs(pos_out_dir)
            instances = contexts(pos_dir)
            logger.info('pos: %d', len(instances))
            for instance in instances:
                with open(os.path.join(pos_out_dir, instance.id + '.json'), 'w', encoding="utf8") as outfile:
                    outfile.write(instance.json())

            os.makedirs(neg_out_dir)
            instances = contexts(neg_dir)
            logger.info('neg: %d', len(instances))
            for instance in instances:
                with open(os.path.join(neg_out_dir, instance.id + '.json'), 'w', encoding="utf8") as outfile:
                    outfile.write(instance.json())
        m = Manager()
        pos_instances = m.list()
        neg_instances = m.list()
        p = Pool(2)
        p.map(ContextProcessor.subprocess_mp_wrapper,
              [(pos_out_dir, pos_instances), (neg_out_dir, neg_instances)])
        p.close()
        pos_instances.extend(neg_instances)
        instances = [i for i in pos_instances]
        with open(os.path.join(contexts_dir, 'contexts.json'), 'w', encoding="utf8") as outfile:
            json.dump(instances, outfile)
            logger.info("Generate contexts.json at %s", str(os.path.curdir))
            # pd.Series(text_fea).to_json(outfile, orient='values')
        return [Object(ins) for ins in instances], contexts_dir

    @staticmethod
    def train(instances, contexts_dir, resource='Location'):
        logger.info('The number of instances: %d', len(instances))
        # Convert the text_fea into the <String, label> pairs.
        docs, y = ContextProcessor.docs(instances, ['r_' + resource])
        # Transform the strings into the np array.
        train_data, voc, vec = Learner.LabelledDocs.vectorize(docs)
        logger.info('neg: %d', len(np.where(y == 0)[0]))
        logger.info('pos: %d', len(np.where(y == 1)[0]))
        # Split the data set into 10 folds.
        folds = Learner.n_folds(train_data, y, fold=10)  # [Fold(f) for f in Learner.n_folds(train_data, y, fold=10)]
        # Wrap a bunch of classifiers and let them vote on every fold.
        clfs = [svm.SVC(kernel='linear', class_weight='balanced', probability=True),
                RandomForestClassifier(class_weight='balanced'),
                LogisticRegression(class_weight='balanced')]
        res = Learner.voting(clfs, train_data, y, folds)
        for clf in clfs:
            clf_name = type(clf).__name__
            logger.debug('CLF: %s', clf_name)
            for fold_id in res[clf_name]['folds']:
                fold = res[clf_name]['folds'][fold_id]
                if 'fp_item' not in fold:
                    continue
                for fp in fold['fp_item']:
                    logger.debug('FP: %s, %s, %s', str(instances[fp].ui_doc), instances[fp].topic,
                                 instances[fp].app_name)
                for fn in fold['fn_item']:
                    logger.debug('FN: %s, %s, %s', str(instances[fn].ui_doc), instances[fn].topic,
                                 instances[fn].app_name)
        with open(os.path.join(contexts_dir, 'folds.json'), 'w') as json_file:
            for fold_id in folds:
                fold = folds[fold_id]
                fold['train_index'] = fold['train_index'].tolist()
                fold['test_index'] = fold['test_index'].tolist()
            # pd.Series(folds).to_json(json_file, orient='values')
            logger.info('The number of folds: %d', len(folds))
            json.dump(folds, json_file)

        with open(os.path.join(contexts_dir, 'voting_res.json'), 'w') as json_file:
            pd.Series(res).to_json(json_file, orient='split')
            logger.info('The total number of overlapping instances after voting: %d', len(res['voting']['all']))
            logger.info('The number of fp: %d', len(res['voting']['fp']))
            logger.info('The number of fn: %d', len(res['voting']['fn']))
            logger.info('conf_mat: %s', str(res['voting']['conf_mat']))
        #   json.dump(res, json_file)
        # with open(os.path.join(contexts_dir, 'voting_predicted_pos.json'), 'w') as json_file:
        # json.dump(predicted_pos_instances, json_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--preprocess", dest="preprocess", action='store_true',
                        help="whether preprocess the dest folder")
    parser.add_argument("-d", "--dir", dest="dir",
                        help="the dest folder")
    parser.add_argument("-c", "--clean", dest="clean", action='store_true',
                        help="whether clear the dest folder")
    parser.add_argument("-r", "--res", dest="res", help="the resource type")
    args = parser.parse_args()
    dest_dir = args.dir
    if args.preprocess:
        ContextProcessor.preprocess(dest_dir)
        exit()
    logger.setLevel(logging.DEBUG)
    logger.info('The data stored at: %s', dest_dir)
    learner.logger.setLevel(logging.INFO)
    samples, samples_dir = ContextProcessor.process(dest_dir, reset_out_dir=args.clean,
                                                    sub_dir_name=os.path.basename(dest_dir))
    ContextProcessor.train(samples, samples_dir, resource=args.res)
