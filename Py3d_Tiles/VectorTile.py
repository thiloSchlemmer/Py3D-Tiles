# -*- coding: utf-8 -*-
import json
from collections import OrderedDict
from .utils import unpackEntry, ungzipFileObject


class VectorTile(object):
    header_byte_length = 44
    vector_tile_header = OrderedDict([
        ['magic', '4s'],
        ['version', 'I'],  # 4bytes
        ['byteLength', 'I'],
        ['featureTableJsonByteLength', 'I'],
        ['featureTableBinaryByteLength', 'I'],
        ['batchTableJsonByteLength', 'I'],
        ['batchTableBinaryByteLength', 'I'],
        ['indicesByteLength', 'I'],
        ['polygonPositionsByteLength', 'I'],
        ['polylinePositionsByteLength', 'I'],
        ['pointPositionsByteLength', 'I'],
    ])

    def __init__(self):
        self.featureTable = ""
        self.header = OrderedDict()
        for k, v in VectorTile.vector_tile_header.items():
            self.header[k] = 0.0

    def fromBytesIO(self, f):
        # Header
        for k, v in VectorTile.vector_tile_header.items():
            self.header[k] = unpackEntry(f, v)

        # featureTable
        featureTableBytes = b''

        featureTableByteLength = self.header['featureTableJsonByteLength']
        for i in range(0, featureTableByteLength):
            featureTableBytes += unpackEntry(f, 's')
        self.featureTable = json.loads(featureTableBytes.decode('utf-8'))

    def fromFile(self, filePath, gzipped=False):
        """
        A method to read a vector tile file. It is assumed that the tile unzipped.

        Arguments:

        ``filePath``

            An absolute or relative path to a quantized-mesh terrain tile. (Required)

        ``gzipped``

            Indicate if the tile content is gzipped. Default is ``False``.
        """
        with open(filePath, 'rb') as f:
            if gzipped:
                f = ungzipFileObject(f)
            self.fromBytesIO(f, )
