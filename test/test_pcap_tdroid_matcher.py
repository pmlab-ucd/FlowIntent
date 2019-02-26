import unittest
from AppInspector.pcap_tdroid_matcher import *
from utils import set_logger
import os
import json
import shutil
import time

log = set_logger('TestPcapTaintDroidMatcher', 'DEBUG')


class TestPcapTaintDroidMatcher(unittest.TestCase):
    def test_parse_logs(self):
        taints, pkg = parse_logs('data/raw/0897d40edb8b6b585f38ca1a9866bd03cd70a5035cc0ec28f933d702f9a38a03')
        self.assertEqual(pkg, 'com.gp.mahjongg')
        self.assertEqual(len(taints), 4)
        for taint in taints:
            log.debug(taint)
            self.assertEqual(taint['process_name'], 'com.gp.mahjongg')
            self.assertEqual(taint['channel'], 'HTTP')

    def test_http_taints(self):
        taints, pkg = parse_logs('data/raw/0897d40edb8b6b585f38ca1a9866bd03cd70a5035cc0ec28f933d702f9a38a03')
        tgt_taints = http_taints(taints)
        self.assertEqual(len(tgt_taints), 4)
        found = False
        for taint in tgt_taints:
            log.debug(taint)
            if 'Location' in taint['type']:
                self.assertEqual(taint['ip'], '120.55.192.233')
                self.assertEqual(taint['data'],
                                 "/getAdByClient.action?type=0&version=1&moblieType=GalaxyNexus&imei=351565054929465"
                                 "&appId=BC1DF56")
                found = True
        self.assertTrue(found)

    def test_parse_dir(self):
        target_json = "data/raw/0897d40edb8b6b585f38ca1a9866bd03cd70a5035cc0ec28f933d702f9a38a03/" \
                      "com.gp.mahjongg0710-03-25-16_sens_http_flows.json"
        if os.path.exists(target_json):
            os.remove(target_json)
        parse_dir('data')
        self.assertTrue(os.path.exists(target_json))
        with open(target_json, 'r') as infile:
            flows = json.load(infile)
        self.assertEqual(len(flows), 3)
        for flow in flows:
            self.assertTrue('IMEI' in flow['taint'])
            log.debug('Flow: ' + str(flow))
            if 'Location' in flow['taint']:
                self.assertTrue('location' in flow['url'])

    def test_organize_dir_by_taint(self):
        out_base_dir = 'data/ground/'
        if os.path.exists('data/ground'):
            shutil.rmtree(out_base_dir)
            time.sleep(3)  # Give it some time to delete.
        dataset = 'raw'
        base_dir = os.path.join('data', dataset)

        taint_type = 'Location'
        out_dir = os.path.join(out_base_dir, taint_type)
        out_dir = os.path.join(out_dir, dataset)

        organize_dir_by_taint(base_dir, out_dir, taint_type, False)
        out_dir = 'data/ground/Location/raw/0897d40edb8b6b585f38ca1a9866bd03cd70a5035cc0ec28f933d702f9a38a03'
        target_json = os.path.join(out_dir, 'com.gp.mahjongg0710-03-25-16_sens_http_flows.json')
        self.assertTrue(os.path.exists(target_json))
        with open(target_json, 'r') as infile:
            flows = json.load(infile)
        self.assertEqual(len(flows), 3)
        for flow in flows:
            self.assertTrue('IMEI' in flow['taint'])
            log.debug('Flow: ' + str(flow))
            if 'Location' in flow['taint']:
                self.assertTrue('location' in flow['url'])
        shutil.rmtree(out_base_dir)

    def test_parse_old_logs(self):
        taints, pkg = parse_logs('data/raw/com.pdw.yw')
        self.assertEqual(pkg, 'com.pdw.yw')
        self.assertEqual(len(taints), 4)
        for taint in taints:
            log.debug(taint)
            self.assertEqual(taint['channel'], 'HTTP')

    def test_old_http_taints(self):
        taints, pkg = parse_logs('data/raw/com.pdw.yw')
        tgt_taints = http_taints(taints)
        self.assertEqual(len(tgt_taints), 4)
        found = False
        for taint in tgt_taints:
            log.debug(taint)
            if 'Location' in taint['type']:
                self.assertEqual(taint['ip'], '106.185.38.147')
                self.assertEqual(taint['data'], "/yinwei/yw/devices")
                found = True
        self.assertTrue(found)

    def test_old_parse_dir(self):
        target_json = "data/raw/com.pdw.yw/" \
                      "com.pdw.yw0712-23-07-16_sens_http_flows.json"
        if os.path.exists(target_json):
            os.remove(target_json)
        parse_dir('data')
        self.assertTrue(os.path.exists(target_json))
        with open(target_json, 'r') as infile:
            flows = json.load(infile)
        self.assertEqual(len(flows), 4)
        for flow in flows:
            self.assertTrue('IMEI' in flow['taint'])
            log.debug('Flow: ' + str(flow))
            if 'Location' in flow['taint']:
                self.assertTrue('devices' in flow['url'])
