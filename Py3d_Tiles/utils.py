# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

from future import standard_library

standard_library.install_aliases()

from past.utils import old_div
import math
import gzip
import io
from struct import pack, unpack, calcsize

EPSILON6 = 0.000001


def packEntry(type, value):
    return pack('<%s' % type, value)


def unpackEntry(f, entry):
    return unpack('<%s' % entry, f.read(calcsize(entry)))[0]


def zigZagEncode(n):
    """
    ZigZag-Encodes a number:
       -1 = 1
       -2 = 3
        0 = 0
        1 = 2
        2 = 4
    """
    return (n << 1) ^ (n >> 31)


def zigZagDecode(z):
    """ Reverses ZigZag encoding """
    return (z >> 1) ^ (-(z & 1))


def clamp(val, minVal, maxVal):
    return max(min(val, maxVal), minVal)


def centroid(a, b, c):
    return [old_div(sum((a[0], b[0], c[0])), 3),
            old_div(sum((a[1], b[1], c[1])), 3),
            old_div(sum([a[2], b[2], c[2]]), 3)]

def gzipFileObject(data):
    compressed = io.BytesIO()
    gz = gzip.GzipFile(fileobj=compressed, mode='wb', compresslevel=5)
    gz.write(data.getvalue())
    gz.close()
    compressed.seek(0)
    return compressed

def ungzipFileObject(data):
    buff = io.BytesIO(data.read())
    f = gzip.GzipFile(fileobj=buff)
    return f



def utf8_byte_len(s):
    return len(s.encode('utf-8'))


def radian_to_degree(radian):
    return radian * 180 / math.pi


def degree_to_radian(degree):
    return degree * math.pi / 180
