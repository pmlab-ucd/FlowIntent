from os.path import expanduser
import os, re
from subprocess32 import STDOUT, check_output

home = expanduser("~")


def run_cmd(cmd):
    print cmd
    seconds = 60
    result = True
    for i in range(1, 3):
        try:
            output = check_output(cmd, stderr=STDOUT, timeout=seconds)
            for line in output.split('\n'):
                if 'Failure' in line or 'Error' in line:
                    result = False
                tmp = line.replace(' ', '')
                tmp = tmp.replace('\n', '')
                if tmp != '':
                    print(line)
            break
        except Exception as exc:
            print(exc)
            result = False
            if i == 2:
                raise Exception(cmd)
    return result


def sign_apk(apk_file):
    cmd = "jarsigner -verbose -digestalg SHA1 -sigalg MD5withRSA -storepass android -keystore " + \
          home + "/.android/debug.keystore " + apk_file + " androiddebugkey"
    run_cmd(cmd)

if __name__ == '__main__':
    apk_dir = 'D:\COSMOS\\apks\FINANCE' #Weather'
    for root, dirs, files in os.walk(apk_dir, topdown=False):
        for filename in files:
            if '_modified' in filename and re.search('apk$', filename):
                sign_apk(os.path.join(root, filename))

