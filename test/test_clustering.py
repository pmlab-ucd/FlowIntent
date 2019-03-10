import unittest
from TrafficAnalyzer.analyzer import Analyzer
from utils import set_logger
from statistics import jaccard


log = set_logger('TestClustering', 'DEBUG')


class TestClustering(unittest.TestCase):
    def test_pattern2set(self):
        url = '/api/analytics/v1/user[\-]android[\.]json[\?]latlon=[\.][%]5C[%]2A&isLAT=[\.][%]5C[%]2A&deviceId=[\.]' \
              '[%]5C[%]2A&v=[\.][%]5C[%]2A'
        a = Analyzer.url_pattern2set(url)

        url = '/api/c2dm/register[\.]json[\?]v=[\.][%]5C[%]2A&latlon=[\.][%]5C[%]2A&installId=[\.][%]5C[%]2A&tz=[\.]' \
              '[%]5C[%]2A&devregid=[\.][%]5C[%]2A&deviceId=[\.][%]5C[%]2A&deviceName=[\.][%]5C[%]2A'
        b = Analyzer.url_pattern2set(url)
        d = jaccard(a, b)
        self.assertTrue(0.33 < d < 0.34)
