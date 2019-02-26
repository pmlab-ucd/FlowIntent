#!/usr/bin/python3
# -*- coding:utf8 -*-
import os
from bs4 import BeautifulSoup
from xml.dom.minidom import parseString
import json
from learner import Learner
from utils import set_logger, file_name_no_ext

logger = set_logger('Context')


class Context:
    """
    App-level context of each running text_fea collected.
    """
    word_topics = {'topic_health': [u'健身', u'运动', u'健康', u'体重', u'身体', u'锻炼'],
                   'topic_sports': [u'足球', u'队员', u'篮球', u'跑步'],
                   'topic_weather': [u'天气', u'预报', u'温度', u'湿度', 'PM2\\.5'],
                   'topic_map': [u'旅行', u'地图', u'地理', u'GPS', u'导航', u'旅游']}

    def __init__(self, data_dir, label, xml, ui_doc=None):
        logger.debug(data_dir)
        self.dir = data_dir
        self.id = os.path.basename(data_dir) + '_' + file_name_no_ext(xml)
        self.label = label
        # Collect the topic and the app name from the html
        self.html = find_html(data_dir)
        self.topic = ''
        self.app_name = ''
        # self.views = []
        self.xml = xml
        if self.html:
            self.topic, self.app_name = description(self.html)
            logger.debug('%s, %s', self.topic, self.app_name)
        # Parse user interfaces
        if ui_doc is None:
            views, self.ui_doc = hierarchy_xml(xml)
        elif ui_doc is not None:
            self.ui_doc = ui_doc
        else:
            self.ui_doc = ''

    def json(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4, ensure_ascii=False)


def find_html(data_dir: str) -> str:
    for filename in os.listdir(data_dir):
        if filename.endswith('.html'):
            return os.path.join(data_dir, filename)


def hierarchy_xml(xml_path):
    """
    Extract views and texts of the given XML.
    :param xml_path: The path of the hierarchy XML file.
    :return: A list of views included in the XML and a list of texts shown on the views.
    """
    all_views = []
    doc = []
    if os.path.exists(xml_path):
        try:
            with open(xml_path, 'rb') as f:
                dom = parseString(f.read())
                nodes = dom.getElementsByTagName('node')
                # Iterate over all the uses-permission nodes
                for node in nodes:
                    if node.getAttribute('text') != '':
                        doc.append(node.getAttribute('text'))
                    if node.getAttribute('package') in str(xml_path):
                        all_views.append(node)
                logger.debug(doc)
        except Exception as e:
            logger.warning('Errors in processing %s: %s', xml_path, str(e))
    else:
        logger.warning('XML %s does not exist!', xml_path)
    return all_views, doc


def contexts(app_cxt_rdir):
    """
    Extract app contexts from the given dir.
    :type app_cxt_rdir: The dest_dir directory that stores the app contexts.
    """
    label = os.path.basename(app_cxt_rdir)
    collection = []
    for root, dirs, files in os.walk(app_cxt_rdir):
        logger.info(app_cxt_rdir)
        for dir_name in dirs:
            logger.info('processing %s', dir_name)
            xmls = find_xmls(os.path.join(root, dir_name))
            # To handle old data generated in SECON.
            hier_word = False
            for xml in xmls:
                if 'hier' in xml:
                    hier_word = True
                break
            if hier_word:
                length = 0
                final_xml = ''
                ui_doc = ''
                for xml in xmls:
                    views, doc = hierarchy_xml(xml)
                    if len(views) >= length:
                        final_xml = xml
                        # self.views = views
                        ui_doc = doc
                        length = len(views)
                cxt = Context(os.path.join(root, dir_name), label, xml=final_xml, ui_doc=ui_doc)
                if len(cxt.ui_doc) == 0 and len(cxt.app_name) == 0 and len(cxt.topic) == 0:
                    continue
                collection.append(cxt)
                continue
            # To handle new data.
            for xml in xmls:
                cxt = Context(os.path.join(root, dir_name), label, xml=xml)
                if len(cxt.ui_doc) == 0 and len(cxt.app_name) == 0 and len(cxt.topic) == 0:
                    continue
                collection.append(cxt)
    return collection


def find_xmls(data_dir):
    xmls = []
    for fname in os.listdir(data_dir):
        if fname.endswith('.xml'):
            xmls.append(os.path.join(data_dir, fname))
    return xmls


def description(html):
    category = ''
    app_name = ''
    try:
        soup = BeautifulSoup(open(html, 'r', encoding="utf8", errors='ignore'), "html.parser")
        app_name_soup = BeautifulSoup(str(soup.select('.app-name')), "html.parser")
        app_name = app_name_soup.span.string
        category_soup = BeautifulSoup(str(soup.select('.nav')), "html.parser")
        category = category_soup.select('span')[2].a.string
        desc_soup = BeautifulSoup(str(soup.select('.brief-long')), "html.parser")
        desc = str(desc_soup.select('p'))
        unseen = []
        word_list = Learner.str2words(desc)
        topic_word_counter = {}
        for word in word_list:
            if word in unseen:
                continue
            else:
                unseen.append(word)
                for topic in Context.word_topics.keys():
                    if word not in Context.word_topics[topic]:
                        continue
                    logger.debug(word)
                    if topic not in topic_word_counter.keys():
                        topic_word_counter[topic] = 1
                    else:
                        topic_word_counter[topic] += 1
                    break
        if len(topic_word_counter) == 0:
            return [category, app_name]
        for topic in sorted(topic_word_counter, key=topic_word_counter.get, reverse=True):
            return [topic, app_name]
    except Exception as e:
        logger.warning(e)
        return [category, app_name]


class Object(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [Object(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, Object(b) if isinstance(b, dict) else b)

    def json(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4, ensure_ascii=False)
