import unittest
from pcap_processor import tcp_stream_number, http_trace, logger
import logging
from utils import set_logger

log = set_logger('TestPcapProcessor', 'DEBUG')


class TestPcapProcessor(unittest.TestCase):
    def test_tcp_stream_number(self):
        number = tcp_stream_number(
            'data/0897d40edb8b6b585f38ca1a9866bd03cd70a5035cc0ec28f933d702f9a38a03/com.gp.mahjongg0710-03-25-16.pcap')
        self.assertEqual(number, 4)

    def test_http_trace(self):
        pcap = 'data/0897d40edb8b6b585f38ca1a9866bd03cd70a5035cc0ec28f933d702f9a38a03/com.gp.mahjongg0710-03-25-16.pcap'
        logger.setLevel(logging.INFO)
        for i in range(tcp_stream_number(pcap) + 1):
            flow = http_trace(pcap, i)
            log.debug(flow)
