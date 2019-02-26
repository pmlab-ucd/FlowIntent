__author__ = 'hao'

from learner import Learner
from utils import set_logger
import os
import json

logger = set_logger('pcap_processor', 'INFO')

"""
The utilities to process pcap files. 
"""


def tcp_stream_number(pcap_path):
    """
    Get the number of number tcp streams, with the help of tshark.
    :param pcap_path: The input directory.
    :return: The number of tcp streams identified by tshark.
    """
    cmd = 'tshark -r ' + pcap_path + ' -T fields -e tcp.stream'  # | sort -n | tail -1'
    lines = os.popen(cmd).readlines()
    max_index = 0
    for line in lines:
        if line.rstrip().isdigit():
            max_index = max(max_index, int(line))
    logger.debug(cmd)
    logger.debug(max_index)
    return max_index


def tcp_stream(pcap_path, stream_index, out_dir='./', out_name=None, overwrite=True):
    """
    Retrieve tcp packets of a HTTP trace from a pcap and output to a pcap, with the help of tshark.
    tshark is able to retrieve the whole tcp stream, including the packet in HTTP response.
    :param pcap_path: The input directory.
    :param stream_index: The index of the HTTP trace.
    :param out_dir: The output directory.
    :param out_name: The output pcap name.
    :param overwrite:
    :return:
    """
    if out_name is None:
        out_name = os.path.basename(pcap_path).replace('.pcap', '') + '_ts_' + str(stream_index) + '.pcap'
    elif not out_name.endswith('.pcap'):
        out_name += '.pcap'
    out_pcap = os.path.join(out_dir, out_name)
    if (not overwrite) and os.path.exists(out_pcap):
        return
    cmd = 'tshark -r ' + pcap_path + ' -Y "tcp.stream==' + str(stream_index) + '" -w ' + out_pcap
    logger.debug(cmd)
    os.system(cmd)


def http_trace(pcap, stream_index=0, label='', matching_funcs=None, args=None):
    """
    The implementation of http_requests, which extract the interested http requests from a given pcap.
    :param: pcap: pcap path.
    :param: stream_index: The tcp stream index labelled by tshark.
    :param: label: The supervised learning label of the extracted http requests.
    :return flow: The feature value of the HTTP trace.
    """
    cmd = 'tshark -r ' + pcap + ' -Y "tcp.stream eq ' + str(stream_index) + '" -T fields ' \
                                                                            '-e frame.len ' \
                                                                            '-e tcp.srcport ' \
                                                                            '-e frame.protocols ' \
                                                                            '-e frame.time_epoch ' \
                                                                            '-e ip.dst ' \
                                                                            '-e http.request.full_uri ' \
                                                                            '-e http.content_length ' \
                                                                            '-e http.response '
    logger.debug(cmd)
    lines = os.popen(cmd).readlines()
    i = 0
    frame_lengths = []
    epochs = []
    up_count = 0
    up_port = -1
    up_frames = []
    down_frames = []
    non_http_tcp_num = 0
    ip_dst = ''
    url = ''
    for line in lines:
        if line.rstrip() is not '':
            i += 1
            logger.debug(str(i) + ' ' + line)
            values = line.split('\t')
            frame_len = int(values[0])
            frame_lengths.append(frame_len)
            non_http_tcp_num = (non_http_tcp_num + 1) if 'http' in values[2] else non_http_tcp_num
            epochs.append(float(values[3]))
            if i == 1:
                up_port = values[1]
                ip_dst = values[4]
            if values[1] == up_port:
                up_count += 1
                up_frames.append(frame_len)
            else:
                down_frames.append(frame_len)
            url = values[5] if values[5] is not '' else url
    if url == '':
        # Not a proper http flow, may only be a tcp stream, or be truncated invalidly.
        return None
    taint = ''
    if matching_funcs is not None:
        # Iterate over the matching funcs (and the corresponding args) and see whether this trace matches any func.
        for i in range(len(matching_funcs)):
            if matching_funcs[i](args[i], [ip_dst, url]):
                if args[i][2] in taint:
                    continue
                else:
                    taint = args[i][2]
    if matching_funcs is not None and taint == '':
        return None
    intervals = []
    for i in range(1, len(epochs)):
        intervals.append(epochs[i] - epochs[i - 1])
    try:
        flow = dict()
        flow['frame_num'] = i
        flow['up_count'] = up_count
        flow['non_http_num'] = non_http_tcp_num
        flow['len_stat'] = Learner.stat_fea_cal(frame_lengths)
        flow['epoch_stat'] = Learner.stat_fea_cal(intervals)
        flow['up_stat'] = Learner.stat_fea_cal(up_frames)
        flow['down_stat'] = Learner.stat_fea_cal(down_frames)
        flow['url'] = url
        flow['label'] = label
        flow['taint'] = taint
        flow['pcap'] = pcap + '_steam_' + str(stream_index)
        logger.debug(flow)
    except Exception as e:
        logger.warning('Error in processing ' + pcap)
        logger.warning(e)
        return None
    return flow


def flows2json(sub_dir, filename, label=None, filter_funcs=None, args=None,
               fn_filter=None, json_ext='_sens_http_flows.json'):
    if (fn_filter is None or fn_filter not in filename) and filename.endswith('.pcap'):
        sub_flows = []
        pcap_path = os.path.join(sub_dir, filename)
        for i in range(tcp_stream_number(pcap_path) + 1):
            try:
                flow = http_trace(pcap_path, i, label=label, matching_funcs=filter_funcs, args=args)
                if flow is not None:
                    sub_flows.append(flow)
            except UnicodeDecodeError as e:
                logger.warning('Errors in processing %s', pcap_path)
                logger.warning(e)
        if len(sub_flows) != 0:
            with open(os.path.join(sub_dir, os.path.splitext(filename)[0] + json_ext), 'w',
                      encoding="utf8", errors='ignore') as outfile:
                json.dump(sub_flows, outfile)
            return sub_flows


def flows2jsons(sub_dir, flows, label=None, filter_funcs=None, args=None,
                fn_filter='filter', json_ext='_sens_http_flows.json'):
    """
    Generate the jsons from the flows.
    :param sub_dir:
    :param flows:
    :param label:
    :param filter_funcs:
    :param args:
    :param fn_filter:
    :param json_ext:
    """
    for filename in os.listdir(sub_dir):
        if (fn_filter is None or fn_filter not in filename) and filename.endswith('.pcap'):
            sub_flows = flows2json(sub_dir, filename, label, filter_funcs, args, fn_filter, json_ext)
            if sub_flows is not None:
                flows.extend(sub_flows)


if __name__ == '__main__':
    input_pcap_path = 'H:\\FlowIntent\\test\\0\\com.anforen.voicexf' \
                      '\\com.anforen.voicexf0713-00-01-55_ts_13.pcap'
    http_trace(input_pcap_path)
