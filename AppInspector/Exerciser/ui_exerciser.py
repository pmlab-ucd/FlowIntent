#!/usr/bin/env python
# -*-encoding:utf-8-*-

from uiautomator import Device
from xml.dom.minidom import parseString
from utils import ISO_TIME_FORMAT, set_logger, run_method, set_file_log, kill_by_name, current_time
import os
from subprocess import STDOUT, check_output, Popen, PIPE
import re
import csv
import time
import psutil
import json
from AppInspector.Exerciser.sign_apks import sign_apk
from AppInspector.Exerciser.ViewClientHandler import ViewClientHandler
from AppInspector.Exerciser.online_tdroid_log_collector import OnlineTaintDroidLogHandler

logger = set_logger('UIExerciser', 'INFO')


class UIExerciser:
    emu_proc = None
    emu_loc = None
    emu_name = None
    series = None

    @staticmethod
    def start_activity(package, activity):
        """
        Open the given activity.
        :param package:
        :param activity:
        :return:
        """
        logger.info("Start Activity " + activity)
        # cmd = 'adb -s ' + series + ' shell am start -D -n ' + package + '/' + activity
        # os.popen('adb -s ' + series + ' shell am start -n ' + package + '/.' + activity)
        # cmd = self.monkeyrunner_loc + ' ' + os.getcwd() + '/run_activity_monkeyrunner.py ' + ' ' + \
        # self.series + ' ' + package + '/' + activity
        cmd = ' shell am start -n ' + package + '/' + activity
        return UIExerciser.run_adb_cmd(cmd)

    @staticmethod
    def get_package_name(aapt, apk):
        cmd = aapt + ' dump badging ' + apk
        try:
            output = check_output(cmd, stderr=STDOUT)
        except Exception as e:
            logger.error(e)
            return None
        for line in output.split('\n'):
            logger.info(line)
            if 'package: name=' in line:
                # the real code does filtering here
                package = re.findall('\'([^\']*)\'', line.rstrip())[0]
                logger.info('Package: ' + package)
                return package
            else:
                break

    @staticmethod
    def launchable_activities(aapt, apk):
        cmd = aapt + ' dump badging ' + apk
        activities = []
        try:
            output = check_output(cmd, stderr=STDOUT)
        except Exception as e:
            logger.error(e)
            return None
        for line in output.split('\n'):
            if 'launchable-activity: name=' in line:
                # the real code does filtering here
                activity = re.findall('\'([^\']*)\'', line.rstrip())[0]
                logger.info('Launchable activity: ' + activity)
                activities.append(activity)

        return activities

    @staticmethod
    def is_crashed(dev, xml_data):
        dom = parseString(xml_data.encode("utf-8"))
        nodes = dom.getElementsByTagName('node')
        # Iterate over all the uses-permission nodes
        # crashed = True
        for node in nodes:
            if node.getAttribute('text') != '':
                if ' has stopped.' in node.getAttribute('text'):
                    if dev(resourceId="android:id/button1", text="OK").exists:
                        dev(resourceId="android:id/button1", text="OK").click()
                    logger.warn('Crashed!')
                    return True
                    # print(node.getAttribute('text'))
                    # print(node.toxml())
                    # if node.getAttribute('package') == package:
                    # crashed = False
        return False

    @staticmethod
    def is_sms_alarm(dev: classmethod, xml_data: object) -> object:
        """
        Whether the current dialog is a SMS permission request, after Android 5.0.
        :param dev:
        :param xml_data:
        :return:
        """
        dom = parseString(xml_data.encode("utf-8"))
        nodes = dom.getElementsByTagName('node')
        # Iterate over all the uses-permission nodes
        # crashed = True
        for node in nodes:
            if node.getAttribute('text') != '' and 'would like to send a message to ' in node.getAttribute('text'):
                    if dev(resourceId='android:id/button2', text="Cancel").exists:
                        # print dev.press.back()
                        # print dev(resourceId='android:id/sms_short_code_remember_choice_checkbox').click()
                        logger.info(dev(resourceId='android:id/button2', text="Cancel").click())
                        print('Send SMS alarm')
                    return True
        return False

    @staticmethod
    def touch(dev, node_bounds):
        node_bounds = node_bounds[1: len(node_bounds) - 1]
        node_bounds = node_bounds.split('][')
        node_bounds[0] = node_bounds[0].split(',')
        node_bounds[0] = map(float, node_bounds[0])
        node_bounds[1] = node_bounds[1].split(',')
        node_bounds[1] = map(float, node_bounds[1])
        x = 0.5 * (node_bounds[1][0] - node_bounds[0][0]) + node_bounds[0][0]
        y = 0.5 * (node_bounds[1][1] - node_bounds[0][1]) + node_bounds[0][1]
        dev.click(x, y)

    @staticmethod
    def tcpdump_begin(package=None, current_time=None, nohup=False):
        current_time = current_time if current_time is not None else current_time()
        package = package if package is not None else 'collect'
        if not nohup:
            cmd = ' shell /data/local/tcpdump -w /sdcard/' + package + '_' + current_time + '.pcap'
        else:
            sub = "nohup /data/local/tcpdump -w /sdcard/" + package + "_" + current_time + ".pcap"
            cmd = ' shell "' + sub + '"'
        UIExerciser.run_adb_cmd(cmd)

    @staticmethod
    def tcpdump_end(output_dir, package=None, current_time=None):
        current_time = current_time if current_time is not None else current_time()
        package = package if package is not None else 'collect'
        UIExerciser.run_adb_cmd(
            'shell ps | grep tcpdump | awk \'{print $2}\' | xargs adb -s ' + UIExerciser.series + ' shell kill')
        out_pcap = output_dir + package + current_time + '.pcap'
        cmd = 'pull /sdcard/' + package + '_' + current_time + '.pcap ' + out_pcap
        UIExerciser.run_adb_cmd(cmd)

    @staticmethod
    def screenshot(dir_data, activity, first_page, dev=None, pkg=''):
        """
        Take the screenshot of the given activity.
        :param dir_data:
        :param activity:
        :param first_page:
        :param dev:
        :param pkg:
        :return:
        """
        logger.info('Try to dump layout XML of ' + activity)
        if dev is None:
            dev = Device(UIExerciser.series)
        dev.screen.on()
        if activity == '':
            activity = 'first_page'
        activity = str(activity).replace('\"', '')
        # dev.wait.idle()
        logger.info('Dumping...' + activity)
        if first_page:
            UIExerciser.pass_first_page(dev)
        xml_data = None
        try:
            xml_data = dev.dump()
        except Exception as e:
            logger.error(e)
            # The dev may be died, force exit.
            exit(1)

        UIExerciser.is_crashed(dev, xml_data)
        while UIExerciser.is_sms_alarm(dev, xml_data):
            xml_data = dev.dump()

        xml_data = ViewClientHandler.fill_ids(xml_data, pkg)
        logger.info(xml_data)

        f = open(dir_data + activity + '.xml', "wb", )
        f.write(xml_data.encode('utf-8'))
        f.close()
        try:
            logger.info(dev.screenshot(dir_data + activity + '.png'))
        except Exception as e:
            logger.error(e)
            UIExerciser.run_adb_cmd('shell /system/bin/screencap -p /sdcard/screenshot.png')
            UIExerciser.run_adb_cmd('pull /sdcard/screenshot.png ' + dir_data + activity + '.png')

        return True

    @staticmethod
    def start_activities(package, csv_path, output_dir, traffic=False):
        """

        :param package:
        :param csv_path:
        :param output_dir:
        :param traffic: Whether record the traffic pcaps.
        """
        logger.info("Try to read csv " + csv_path)
        csv_file = open(csv_path, 'rb')
        spam_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
        for row in spam_reader:
            activity = row[0]
            activity = str(activity).replace('\"', '')
            if 'com.google.ads.AdActivity' in activity:
                logger.error('Cannot start Activity: ' + activity)
                continue
            if UIExerciser.start_activity(package, activity):
                cur_time = current_time()
                if traffic:
                    UIExerciser.run_adb_cmd('logcat -c')
                    logger.info('clear logcat')  # self.screenshot(output_dir, activity)

                    # UIExerciser.run_adb_cmd('shell "nohup /data/local/tcpdump -w /sdcard/' + package + current_time  + '.pcap &"')
                    # UIExerciser.run_adb_cmd('shell "nohup logcat -v threadtime -s "UiDroid_Taint" > /sdcard/' + package + current_time +'.log &"')

                    # cmd = 'adb -s ' + series + ' shell "nohup /data/local/tcpdump -w /sdcard/' + package + current_time + '.pcap &"'
                    # log.info('tcpdump begins')
                    # cmd = 'adb -s ' + UIExerciser.series + ' shell /data/local/tcpdump -w /sdcard/' +  activity + '.pcap'
                    # os.system(cmd)
                    # log.info(cmd)
                    # process = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
                    UIExerciser.tcpdump_begin(package, cur_time, nohup=False)
                time.sleep(2)
                # self.screenshot(output_dir, activity)
                for i in range(1, 3):
                    if not UIExerciser.check_dev_online(UIExerciser.series):
                        if UIExerciser.emu_proc:
                            UIExerciser.close_emulator(UIExerciser.emu_proc)
                            UIExerciser.emu_proc = UIExerciser.open_emu(UIExerciser.emu_loc, UIExerciser.emu_name)
                        else:
                            raise Exception('Cannot start Activity ' + activity)
                    if run_method(UIExerciser.screenshot, 180, args=[output_dir, activity, False]):
                        break
                    else:
                        logger.warnning("Timeout while dumping XML for " + activity)
                if traffic:
                    time.sleep(10)
                    # process.kill()  # takes more time
                    out_pcap = output_dir + package + cur_time + '.pcap'
                    UIExerciser.tcpdump_end(output_dir, package, cur_time)
                    if not os.path.exists(out_pcap):
                        logger.warning('The pcap does not exist.')
                        # raise Exception('The pcap does not exist.')
                    else:
                        UIExerciser.run_adb_cmd('shell rm /sdcard/' + package + cur_time + '.pcap')

                    taint_logs = []
                    run_method(OnlineTaintDroidLogHandler.collect_taint_log, 15, args=[taint_logs])
                    with open(output_dir + activity + '.json', 'w') as outfile:
                        json.dump(taint_logs, outfile)
            else:
                time.sleep(2)
                logger.error('Cannot start Activity: ' + activity)

    def __init__(self, series, aapt_loc, apk_dir, out_base_dir):
        self.series = series
        UIExerciser.series = series
        self.aapt_loc = aapt_loc

        self.apk_dir = apk_dir
        # self.monkeyrunner_loc = monkeyrunner_loc
        self.out_base_dir = out_base_dir

    @staticmethod
    def check_examined(out_dir):
        """
        Check existing dirs, each dir represents an examined app.
        :param out_dir:
        :return:
        """
        examined = []
        for root, dirs, files in os.walk(out_dir, topdown=False):
            for name in dirs:
                # print(os.path.join(root, name))
                examined.append(name)

        logger.info(len(examined))
        return examined

    @staticmethod
    def get_csv_path(base_dir, par_dir, apk_name):
        path = os.path.join(base_dir, par_dir)
        path = os.path.join(path, apk_name)
        path = os.path.join(path, apk_name + '.apk_tgtAct.csv')
        return path

    @staticmethod
    def open_emulator(emu_loc, emu_name):
        cmd = emu_loc + ' @' + emu_name
        pro = Popen(cmd, stdout=PIPE, shell=True)

        for i in range(1, 5):
            time.sleep(20)
            if UIExerciser.check_dev_online(UIExerciser.series):
                return pro

        return False

    @staticmethod
    def open_emu(emu_loc, emu_name):
        cmd = emu_loc + ' @' + emu_name
        emu_proc = Popen(cmd, stdout=PIPE, shell=True)
        while not UIExerciser.check_dev_online(UIExerciser.series):
            time.sleep(10)
            logger.info('waiting for the emulator ' + UIExerciser.series)
        logger.info(emu_name + ' found')
        return emu_proc

    @staticmethod
    def check_dev_online(series):
        try:
            output = check_output('adb devices', stderr=STDOUT, timeout=10)
            for line in output.split('\n'):
                if series in line:
                    if 'device' in line:
                        return True
        except Exception as e:
            logger.error(e)
            return False

    @staticmethod
    def kill(proc_pid):
        process = psutil.Process(proc_pid)
        for proc in process.children(recursive=True):
            proc.kill()
        try:
            process.kill()
        except Exception as e:
            logger.error(e)

    @staticmethod
    def close_emulator(emu_proc):
        UIExerciser.kill(emu_proc.pid)

    @staticmethod
    def install_apk(series, apk):
        cmd = 'install ' + apk
        if not UIExerciser.run_adb_cmd(cmd):
            sign_apk(apk)
            if not UIExerciser.run_adb_cmd(cmd):
                raise Exception('Cannot install ' + apk)

    @staticmethod
    def uninstall_pkg(series, pkg):
        cmd = 'uninstall ' + pkg
        UIExerciser.run_adb_cmd(cmd, series=series)

    @staticmethod
    def run_adb_cmd(cmd, series=None, seconds=60):
        if series:
            return UIExerciser.run_cmd('adb -s ' + UIExerciser.series + ' ' + cmd, seconds)
        else:
            return UIExerciser.run_cmd('adb ' + cmd, seconds)

    @staticmethod
    def run_cmd(cmd, seconds=60):
        logger.debug('Run cmd: ' + cmd)
        for i in range(1, 3):
            time.sleep(5)
            try:
                result = True
                output = check_output(cmd, stderr=STDOUT, timeout=seconds)
                for line in output.split('\n'):
                    if 'Failure' in line or 'Error' in line or 'unable' in line:
                        result = False
                    tmp = line.replace(' ', '')
                    tmp = tmp.replace('\n', '')
                    if tmp != '':
                        logger.debug(line)
                return result
            except Exception as exc:
                logger.error(exc)
                if not UIExerciser.check_dev_online(UIExerciser.series):
                    if UIExerciser.emu_proc:
                        UIExerciser.close_emulator(UIExerciser.emu_proc)
                        UIExerciser.emu_proc = UIExerciser.open_emu(UIExerciser.emu_loc, UIExerciser.emu_name)
                    else:
                        raise Exception(cmd)
        raise Exception(cmd)

    @staticmethod
    def start_taintdroid():
        UIExerciser.run_adb_cmd('shell am start -n fu.hao.uidroid/.TaintDroidNotifyController')

    @staticmethod
    def pass_first_page(dev):
        time.sleep(5)
        for i in range(8):
            time.sleep(1)
            xml_data = dev.dump()
            dom = parseString(xml_data.encode("utf-8"))
            nodes = dom.getElementsByTagName('node')
            # Iterate over all the uses-permission nodes
            stay = False
            for node in nodes:
                logger.info(node.getAttribute('scrollable'), node.getAttribute('class'))
                if node.getAttribute('scrollable') == 'true':
                    ui_object = dev(className=node.getAttribute('class'), scrollable='true')
                    if ui_object.exists:
                        ui_object.swipe.left()
                        stay = True
                        break
            if not stay:
                break

        xml_data = dev.dump()
        dom = parseString(xml_data.encode("utf-8"))
        nodes = dom.getElementsByTagName('node')
        # Iterate over all the uses-permission nodes
        clickables = []
        for node in nodes:
            logger.info(node.getAttribute('scrollable'), node.getAttribute('class'))
            if node.getAttribute('clickable') == 'true':
                clickables.append(node)
        logger.info(len(clickables))
        if len(clickables) == 1:
            node_bounds = clickables[0].getAttribute('bounds')
            UIExerciser.touch(dev, node_bounds)
            logger.info('click single')
        elif len(clickables) == 2:
            # if detect update info, if 取消， 否
            option_cancel = [u'否', u'取消', u'不升级', u'稍后再说', u'稍后', u'以后'
                                                                  u'稍后更新', u'不更新', u'以后再说',
                             u'Not now', u'Cancel', u'以后更新', u'取 消']
            for clickable in clickables:
                if clickable.getAttribute('text') in option_cancel:
                    UIExerciser.touch(dev, clickable.getAttribute('bounds'))
        time.sleep(5)

    def inspired_run(self, series, apk, examined, trigger_java_dir):
        # apk = 'F:\\Apps\\COMMUNICATION\\com.mobanyware.apk'
        logger.info('base name: ' + os.path.basename(apk))
        apk_name, apk_extension = os.path.splitext(apk)

        logger.info(apk_name)
        if '_modified' not in apk_name:
            return
            # apk_modified = apk_name + '_modified.apk'
        else:
            apk_modified = apk
            apk_name = apk_name.replace('_modified', '')

        apk_name = os.path.basename(apk_name)

        if apk_name in examined:
            logger.error('Already examined ' + apk_name)
            return

        cmd = 'adb devices'
        os.system(cmd)
        logger.info(apk_modified)

        # current_time = time.strftime(ISOTIMEFORMAT, time.localtime())
        par_dir = os.path.basename(os.path.abspath(os.path.join(apk, os.pardir)))  # the parent folder of the apk

        package = self.get_package_name(self.aapt_loc, apk_modified)

        if not package:
            logger.error('Not a valid pkg.')
            return

        csvpath = self.get_csv_path(trigger_java_dir, par_dir, apk_name)
        if not os.path.isfile(csvpath):
            logger.error('tgt_Act.csv does not exist:' + csvpath)
            return

        output_dir = self.out_base_dir + par_dir + '/' + apk_name + '/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_handler = set_file_log(logger, output_dir + 'COSMOS_TRIGGER_PY.log')
        logger.info('apk:' + apk_modified)
        logger.info('pkg:' + package)
        logger.info('csv: ' + csvpath)

        UIExerciser.uninstall_pkg(series, package)
        UIExerciser.install_apk(series, apk_modified)

        cur_time = time.strftime(ISO_TIME_FORMAT, time.localtime())
        UIExerciser.run_adb_cmd('shell monkey -p com.lexa.fakegps --ignore-crashes 1')
        d = Device()
        d(text='Set location').click()

        UIExerciser.run_adb_cmd('logcat -c')
        logger.info('clear logcat')  # self.screenshot(output_dir, activity)

        # UIExerciser.run_adb_cmd('shell "nohup /data/local/tcpdump -w /sdcard/' + package + current_time  + '.pcap &"')
        # UIExerciser.run_adb_cmd('shell "nohup logcat -v threadtime -s "UiDroid_Taint" > /sdcard/' + package + current_time +'.log &"')

        # cmd = 'adb -s ' + series + ' shell "nohup /data/local/tcpdump -w /sdcard/' + package + current_time + '.pcap &"'
        logger.info('tcpdump begins')
        cmd = 'adb -s ' + series + ' shell /data/local/tcpdump -w /sdcard/' + package + '_' + cur_time + '.pcap'
        # os.system(cmd)
        logger.info(cmd)
        process = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)

        UIExerciser.run_adb_cmd('shell monkey -p ' + package + ' --ignore-crashes 1')
        for i in range(1, 3):
            if not UIExerciser.check_dev_online(UIExerciser.series):
                if UIExerciser.emu_proc:
                    UIExerciser.close_emulator(UIExerciser.emu_proc)
                    UIExerciser.emu_proc = UIExerciser.open_emu(UIExerciser.emu_loc, UIExerciser.emu_name)
                else:
                    raise Exception('Cannot start the default Activity')
            if run_method(self.screenshot, 180, args=[output_dir, '', True, package]):
                break
            else:
                logger.warn("Time out while dumping XML for the default activity")

        # UIExerciser.adb_kill('logcat')
        # Utilities.adb_kill('tcpdump')
        # UIExerciser.run_adb_cmd('shell am force-stop fu.hao.uidroid')
        # os.system("TASKKILL /F /PID {pid} /T".format(pid=process.pid))
        time.sleep(10)
        process.kill()  # takes more time
        out_pcap = output_dir + package + '_' + cur_time + '.pcap'
        try:
            while not os.path.exists(out_pcap) or os.stat(out_pcap).st_size < 2:
                time.sleep(5)
                cmd = 'pull /sdcard/' + package + '_' + cur_time + '.pcap ' + out_pcap
                UIExerciser.run_adb_cmd(cmd)
                process.kill()  # takes more time
        except Exception as e:
            logger.warning(e)
            logger.info('wait..')
            # if not os.path.exists(out_pcap):
            # raise Exception('The pcap does not exist.')
        # UIExerciser.run_adb_cmd('shell rm /sdcard/' + package + current_time + '.pcap')

        # UIExerciser.run_adb_cmd('pull /sdcard/' + package + current_time + '.log ' + output_dir)
        # UIExerciser.run_adb_cmd('shell rm /sdcard/' + package + current_time + '.log')
        taint_logs = []
        run_method(OnlineTaintDroidLogHandler.collect_taint_log, 15, args=[taint_logs])
        with open(output_dir + package + '_' + cur_time + '.json', 'w') as outfile:
            json.dump(taint_logs, outfile)

        self.start_activities(package, csvpath, output_dir)

        self.uninstall_pkg(series, package)

        file_handler.close()
        logger.removeHandler(file_handler)
        kill_by_name('adb.exe')

    def inspired_run_lite(self, series, apk, examined, trigger_java_dir):
        # apk = 'F:\\Apps\\COMMUNICATION\\com.mobanyware.apk'
        logger.info('base name: ' + os.path.basename(apk))
        apk_name, apk_extension = os.path.splitext(apk)

        logger.info(apk_name)
        if '_modified' not in apk_name:
            return
            # apk_modified = apk_name + '_modified.apk'
        else:
            apk_modified = apk
            apk_name = apk_name.replace('_modified', '')

        apk_name = os.path.basename(apk_name)

        if apk_name in examined:
            logger.error('Already examined ' + apk_name)
            return

        cmd = 'adb devices'
        os.system(cmd)
        logger.info(apk_modified)

        # current_time = time.strftime(ISOTIMEFORMAT, time.localtime())
        par_dir = os.path.basename(os.path.abspath(os.path.join(apk, os.pardir)))  # the parent folder of the apk

        package = self.get_package_name(self.aapt_loc, apk_modified)

        if not package:
            logger.error('Not a valid pkg.')
            return

        csvpath = self.get_csv_path(trigger_java_dir, par_dir, apk_name)
        if not os.path.isfile(csvpath):
            logger.error('tgt_Act.csv does not exist:' + csvpath)
            return

        output_dir = self.out_base_dir + par_dir + '/' + apk_name + '/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_handler = set_file_log(logger, output_dir + 'COSMOS_TRIGGER_PY.log')
        logger.info('apk:' + apk_modified)
        logger.info('pkg:' + package)
        logger.info('csv: ' + csvpath)

        UIExerciser.uninstall_pkg(series, package)
        UIExerciser.install_apk(series, apk_modified)

        #current_time = time.strftime(ISOTIMEFORMAT, time.localtime())

        UIExerciser.run_adb_cmd('shell monkey -p ' + package + ' --ignore-crashes 1')
        for i in range(1, 3):
            if not UIExerciser.check_dev_online(UIExerciser.series):
                if UIExerciser.emu_proc:
                    UIExerciser.close_emulator(UIExerciser.emu_proc)
                    UIExerciser.emu_proc = UIExerciser.open_emu(UIExerciser.emu_loc, UIExerciser.emu_name)
                else:
                    raise Exception('Cannot start the default Activity')
            if run_method(self.screenshot, 180, args=[output_dir, '', True, package]):
                break
            else:
                logger.warn("Time out while dumping XML for the default activity")

        # UIExerciser.adb_kill('logcat')
        # Utilities.adb_kill('tcpdump')
        # UIExerciser.run_adb_cmd('shell am force-stop fu.hao.uidroid')
        # os.system("TASKKILL /F /PID {pid} /T".format(pid=process.pid))
            # if not os.path.exists(out_pcap):
            # raise Exception('The pcap does not exist.')
        # UIExerciser.run_adb_cmd('shell rm /sdcard/' + package + current_time + '.pcap')

        # UIExerciser.run_adb_cmd('pull /sdcard/' + package + current_time + '.log ' + output_dir)
        # UIExerciser.run_adb_cmd('shell rm /sdcard/' + package + current_time + '.log')

        self.start_activities(package, csvpath, output_dir, traffic=True)

        self.uninstall_pkg(series, package)

        file_handler.close()
        logger.removeHandler(file_handler)
        kill_by_name('adb.exe')
