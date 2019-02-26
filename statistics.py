import math

from utils import set_logger

logger = set_logger('stats', 'INFO')


def median(x):
    sorted_x = sorted(x)
    if len(sorted_x) % 2 == 1:
        return sorted_x[int((len(sorted_x) + 1) / 2 - 1)]
    elif len(sorted_x) == 2:
        return float(sorted_x[0] + sorted_x[1]) / 2
    else:
        a1 = sorted_x[int(len(sorted_x) / 2 + 1)]
        a2 = sorted_x[int(len(sorted_x) / 2 - 1)]
        return float((a1 + a2) / 2)


def mean(x):
    return float(sum(x) / float(len(x)))


def skewness(x):
    cube_x = []
    a = len(x)
    m = mean(x)
    sd = s_dev(x)
    if sd == 0:
        logger.warn('sd is 0!')
        return 0.0
    else:
        for i in range(0, a):
            cube_x.append(((x[i] - m) / float(sd)) * ((x[i] - m) / float(sd)) * ((x[i] - m) / float(sd)))
        return float(sum(cube_x) / float(a))


def s_dev(x):
    square_x = []
    size = len(x)
    if size == 1:
        return 0
    else:
        m = mean(x)
        for i in range(0, size):
            square_x.append((x[i] - m) * (x[i] - m))
        variance = float(sum(square_x) / float(size))
        return math.sqrt(variance)


def kurtosis(x):
    quad_x = []
    size = len(x)
    m = mean(x)
    sd = s_dev(x)
    if sd == 0:
        return "?"
    else:
        for i in range(0, size):
            quad_x.append((x[i] - m) * (x[i] - m) * (x[i] - m) * (x[i] - m))
        k = float(sum(quad_x) / float(size))
        return k / (sd * sd * sd * sd)


def proto(protocol, dest_port):
    if protocol == 6:
        if dest_port == 80:
            return 1
        elif dest_port == 8080:
            return 1
        elif dest_port == 443:
            return 2
        else:
            return 3
    elif protocol == 17:
        return 4
