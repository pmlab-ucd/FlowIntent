from subprocess import STDOUT, check_output
import logging
import psutil
import threading
import os
import time
import errno
import stat

ISO_TIME_FORMAT = '%m%d-%H-%M-%S'


def set_logger(tag, level='DEBUG'):
    log = logging.getLogger(tag)
    console_handler = logging.StreamHandler()

    if level == 'DEBUG':
        log.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
    elif level == 'INFO':
        log.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)
    elif level == 'WARN':
        log.setLevel(logging.WARN)
        console_handler.setLevel(logging.WARN)
    else:
        log.setLevel(logging.ERROR)
        console_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    log.addHandler(console_handler)
    return log


logger = set_logger('Utilities')


def run_cmd(cmd):
    logger.debug('Run cmd: ' + cmd)

    seconds = 60
    result = True
    for i in range(1, 3):
        try:
            result = True
            output = check_output(cmd, stderr=STDOUT, timeout=seconds)
            for line in output.split('\n'):
                if 'Failure' in line or 'Error' in line:
                    result = False
                tmp = line.replace(' ', '')
                tmp = tmp.replace('\n', '')
                if tmp != '':
                    logger.debug(line)
            break
        except Exception as exc:
            logger.warning(exc)
            result = False
            if i == 2:
                # close_emulator(emu_proc)
                # emu_proc = open_emu(emu_loc, emu_name)
                raise Exception(cmd)

    return result


def set_file_log(logger, file_path):
    file_handler = logging.FileHandler(file_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    return file_handler


def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def kill_proc_tree(pid, including_parent=True):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()
    psutil.wait_procs(children, timeout=5)
    if including_parent:
        parent.kill()
        parent.wait(5)


def run_method(target, timeout, args=None):
    p = threading.Thread(target=target, args=args)
    p.start()
    # Wait for 120 seconds or until process finishes
    p.join(timeout)
    # If thread is still active
    if p.is_alive():
        # Terminate
        # p.terminate()
        logger.warning('Timeout!!!')
        try:
            kill_proc_tree(p.ident, including_parent=False)
        except psutil.NoSuchProcess:
            return False
        p.join()
        return False
    else:
        return True


def adb_process2ids(name):
    seconds = 60
    output = check_output('adb shell ps', stderr=STDOUT, timeout=seconds)
    targets = []
    for line in output.split('\n'):
        # print line
        tmp = line.replace(' ', '')
        tmp = tmp.replace('\n', '')
        if tmp != '':
            # print line
            items = str(line).split(' ')
            items = filter(None, items)
            if name in items[len(items) - 1]:
                targets.append(items[1])
    return targets


def adb_id2process(pid):
    seconds = 60
    output = check_output('adb shell ps', stderr=STDOUT, timeout=seconds)
    for line in output.split('\n'):
        # print line
        tmp = line.replace(' ', '')
        tmp = tmp.replace('\n', '')
        if tmp != '':
            # print line
            items = str(line).split(' ')
            items = filter(None, items)
            if pid == items[1]:
                return items[len(items) - 1]
    else:
        return 'Unknown'


def adb_kill(name):
    for target in adb_process2ids(name):
        os.popen('adb shell kill ' + target)


def kill_by_name(name):
    for proc in psutil.process_iter():
        # check whether the process name matches
        if proc.name() == name:
            kill(proc.pid)


def current_time():
    return time.strftime(ISO_TIME_FORMAT, time.localtime())


def file_name_no_ext(path: str) -> str:
    return os.path.basename(os.path.splitext(path)[0])


def handle_remove_readonly(func, path, exc):
    """
    https://stackoverflow.com/questions/1213706/what-user-do-python-scripts-run-as-in-windows
    :param func:
    :param path:
    :param exc:
    :return:
    """
    excvalue = exc[1]
    if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
        func(path)
    else:
        raise RuntimeError
