import unittest
from TrafficAnalyzer.analyzer import Analyzer
from utils import set_logger
import os
import json
import shutil
import time

log = set_logger('TestPcapTaintDroidMatcher', 'DEBUG')


class TestPcapTaintDroidMatcher(unittest.TestCase):
    def test_pred_pos_contexts(self):
        pass

