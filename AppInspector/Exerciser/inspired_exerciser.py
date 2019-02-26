from AppInspector.Exerciser.ui_exerciser import UIExerciser
from utils import set_logger
import re
import os
import time

if __name__ == '__main__':
    logger = set_logger('COSMOS_TRIGGER_PY-Console')

    device = 'nexus4'
    pc = 'iai'
    lite = False

    if device == 'nexus4':
        series = '01b7006e13dd12a1'
    elif device == 'galaxy':
        series = '014E233C1300800B'
    elif device == 'nexuss':
        series = '39302E8CEA9B00EC'
    elif device == 'xiaoyao':
        series = '127.0.0.1:21503'
    else:
        series = 'emulator-5556'


    user = 'hfu'
    aapt_loc = 'C:\Users\\' + user + '\AppData\Local\Android\sdk/build-tools/19.1.0/aapt.exe'
    apk_dir = 'C:\Users\\' + user + '\Documents\FlowIntent\\apks\\Business\\'
    UIExerciser.emu_loc = 'C:\Users\hfu\AppData\Local\Android\sdk/tools/emulator.exe'
    UIExerciser.emu_name = 'Qvga'

    out_base_dir = os.path.abspath(os.pardir + '/output/') + '/'

    #UIExerciser.emu_proc = UIExerciser.open_emu(UIExerciser.emu_loc, UIExerciser.emu_name)
    examined = UIExerciser.check_examined(out_base_dir)
    for root, dirs, files in os.walk(apk_dir, topdown=False):
        for filename in files:
            if re.search('apk$', filename):
                # main_process = multiprocessing.Process(target=handle_apk, args=[os.path.join(root, filename), examined])
                # main_process.start()
                # main_process.join()
                for i in range(3):
                    try:
                        if device == 'xiaoyao':
                            UIExerciser.run_adb_cmd('connect 127.0.0.1:21503')
                        apk = os.path.join(root, filename)
                        exerciser = UIExerciser(series, aapt_loc, apk_dir, out_base_dir, logger)
                        if lite:
                            exerciser.inspired_run_lite(series, os.path.join(root, filename), examined, 'C:\\Users\\hfu\\Documents\\COSMOS\\output\\java\\')
                        else:
                            exerciser.inspired_run(series, os.path.join(root, filename), examined,
                                                   'C:\\Users\\hfu\\Documents\\COSMOS\\output\\java\\')
                        break
                    except Exception as e:
                        logger.warn(str(e))
                        if device == 'xiaoyao':
                            if not UIExerciser.run_adb_cmd('connect 127.0.0.1:21503'):
                                exit(1)
                        else:
                            UIExerciser.run_adb_cmd('reboot')
                        time.sleep(90)
