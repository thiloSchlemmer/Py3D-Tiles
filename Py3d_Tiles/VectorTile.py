# -*- coding: utf-8 -*-
import gzip
import io
import json
import math
import os
from collections import OrderedDict

from Py3d_Tiles import earcut
from .utils import unpackEntry, ungzipFileObject, gzipFileObject, packEntry, clamp, zigZagEncode, utf8_byte_len, \
    zigZagDecode

MIN = 0.0
MAX = 32767.0
UV_RANGE = 32767


def unpack_and_decode_position(f, elementCount, structType, dimension=2):
    """
    A private method to iteratively unpack position-elements.
    :param f: The file-like object
    :param positions_length: the size in integer of the buffer,
    where the position-tuple must be read
    :param dimension: the dimension of the position-tuple, 2 for 2d-Position; 3 for 3d-Positions
    :return iterator over the position-elements as array
    """
    if elementCount == 0:
        return

    i = 0
    # Delta decoding
    delta = 0
    while i != elementCount:
        delta += zigZagDecode(unpackEntry(f, structType))
        yield delta
        i += 1


def get_string_padded(s, padding_size):
    byte_offset = 0
    boundary = padding_size

    byte_length = utf8_byte_len(s)
    remainder = (byte_offset + byte_length) % boundary
    padding = 0 if remainder == 0 else boundary - remainder

    return s + ' ' * padding


class VectorTile(object):
    VECTOR_TILE_VERSION = 1
    VECTOR_TILE_MAGIC = "vctr"
    geometry_types = {1: "Point",
                      2: "LineString",
                      3: "Polygon",
                      4: "MultiPoint",
                      5: "MultiLineString",
                      6: "Multipolygon"}
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
    default_feature_table = {'MINIMUM_HEIGHT': MAX,
                             'MAXIMUM_HEIGHT': MIN,
                             'POINTS_LENGTH': 0,
                             'POLYLINES_LENGTH': 0,
                             'POLYGONS_LENGTH': 0,
                             'RECTANGLE': [0, 0, 0, 0],
                             'REGION': [0, 0, 0, 0, 0, 0]}

    def __init__(self, tile_extent, property_names_to_publish=None):
        if property_names_to_publish is None:
            property_names_to_publish = []
        if tile_extent is None:
            tile_extent = {'west': None,
                           'south': None,
                           'east': None,
                           'north': None,
                           }
        self.featureTable = VectorTile.default_feature_table
        self.property_names_to_publish = property_names_to_publish

        self.extent = tile_extent

        self.features = []
        self.points = []
        self.polylines = []
        self.polygons = []
        self.header = OrderedDict()
        for k, v in self.vector_tile_header.items():
            self.header[k] = 0.0
        self.header['magic'] = bytes(VectorTile.VECTOR_TILE_MAGIC, 'utf-8')
        self.header['version'] = VectorTile.VECTOR_TILE_VERSION
        self.bounding_volume = {'west': self.extent['west'],
                                'south': self.extent['south'],
                                'east': self.extent['east'],
                                'north': self.extent['north'],
                                'minimum_height': None,
                                'maximum_height': None
                                }

    def from_bytes_io(self, f):
        # Header
        for k, v in VectorTile.vector_tile_header.items():
            self.header[k] = unpackEntry(f, v)

        # featureTable
        feature_table_bytes = b''

        feature_table_byte_length = self.header['featureTableJsonByteLength']
        print("reading FeatureTable...")
        for i in range(0, feature_table_byte_length):
            feature_table_bytes += unpackEntry(f, 's')
        self.featureTable = json.loads(feature_table_bytes.decode('utf-8'))
        print("done")

        feature_table_bytes = b''
        feature_table_binary_byte_length = self.header['featureTableBinaryByteLength']
        if feature_table_binary_byte_length == 0:
            print("FeatureTable BinaryBody is empty.")
        for i in range(0, feature_table_binary_byte_length):
            feature_table_bytes += unpackEntry(f, 's')
        self.featureTable_body = feature_table_bytes

        self.bounding_volume = {'west': self.featureTable['RECTANGLE'][0],
                                'south': self.featureTable['RECTANGLE'][1],
                                'east': self.featureTable['RECTANGLE'][2],
                                'north': self.featureTable['RECTANGLE'][3],
                                'minimum_height': self.featureTable['MINIMUM_HEIGHT'],
                                'maximum_height': self.featureTable['MAXIMUM_HEIGHT']
                                }
        batch_table_json = None
        batch_table_bytes = b''
        batch_table_json_byte_length = self.header['batchTableJsonByteLength']
        print("reading BatchTable...")
        for i in range(0, batch_table_json_byte_length):
            batch_table_bytes += unpackEntry(f, 's')
        batch_table_json = json.loads(batch_table_bytes.decode('utf-8'))
        print("...Found {} Properties".format(len(batch_table_json.keys())))
        print("done")

        indices_bytes_length = self.header['indicesByteLength']
        print("reading Indices...")
        indices = []

        for i in range(0, indices_bytes_length):
            indices.append(unpackEntry(f, 'I'))
        if indices_bytes_length == 0:
            print("...Found no indices...")
        else:
            print("...Found {} indices...".format(len(indices)))
        print("done")

        print("reading Positions...")
        polygon_u = []
        polygon_v = []
        polygon_positions_byte_length = self.header['polygonPositionsByteLength']

        for ud in unpack_and_decode_position(f, polygon_positions_byte_length / 8, 'H'):
            polygon_u.append(ud)
        for vd in unpack_and_decode_position(f, polygon_positions_byte_length / 8, 'H'):
            polygon_v.append(vd)

        polyline_u = []
        polyline_v = []
        polyline_h = []
        polyline_positions_byte_length = self.header['polylinePositionsByteLength']
        for ud in unpack_and_decode_position(f, polyline_positions_byte_length / 8, 'H'):
            polyline_u.append(ud)
        for vd in unpack_and_decode_position(f, polyline_positions_byte_length / 8, 'H'):
            polyline_v.append(vd)
        for hd in unpack_and_decode_position(f, polyline_positions_byte_length / 8, 'H'):
            polyline_h.append(hd)

        point_u = []
        point_v = []
        point_h = []
        point_positions_byte_length = self.header['pointPositionsByteLength']
        for ud in unpack_and_decode_position(f, point_positions_byte_length / 8, 'H'):
            point_u.append(ud)
        for vd in unpack_and_decode_position(f, point_positions_byte_length / 8, 'H'):
            point_v.append(vd)
        for hd in unpack_and_decode_position(f, point_positions_byte_length / 8, 'H'):
            point_h.append(hd)

        print(point_u)
        print(point_v)
        print(point_h)

    def from_file(self, file_path, gzipped=False):
        """
        A method to read a vector tile file. It is assumed that the tile is unzipped.

        Arguments:

        :param file_path: An absolute or relative path to a quantized-mesh terrain tile. (Required)
        :param gzipped: Indicate if the tile content is gzipped. Default is ``False``.
        """
        with open(file_path, 'rb') as f:
            if gzipped:
                f = ungzipFileObject(f)
            self.from_bytes_io(f, )

    def add_feature(self, feature: {}):
        """
        A method to add a feature to the vector tile. It is assumed that the feature is
        a geojson-node (with properties- and geometry-node)
        :param feature:  the (geojson) feature to add to the vector tile
        """
        assert ("properties" in feature)
        assert ("geometry" in feature)

        geometry = feature["geometry"]

        assert ("type" in geometry)
        assert ("coordinates" in geometry)

        geometry_type = geometry["type"]
        coordinates = geometry["coordinates"]

        self._update_extent(coordinates)

        if geometry_type == "Point":
            self.points.append(feature)
            self.featureTable['POINTS_LENGTH'] += 1
        elif geometry_type == "LineString":
            self.polylines.append(feature)
            self.featureTable['POLYLINES_LENGTH'] += 1
        elif geometry_type == "Polygon":
            self.polygons.append(feature)
            self.featureTable['POLYGONS_LENGTH'] += 1
        else:
            raise ValueError("Geometry-Type {} is not supported.".format(geometry_type))

    def to_file(self, filePath, gzipped=False):
        """
        A method to write the terrain tile data to a physical file.

        Argument:

        ``file_path``

            An absolute or relative path to write the terrain tile. (Required)

        ``gzipped``

            Indicate if the content should be gzipped. Default is ``False``.
        """
        if os.path.isfile(filePath):
            raise IOError('File %s already exists' % filePath)

        if not gzipped:
            with open(filePath, 'wb') as f:
                self._write_to(f)
        else:
            with gzip.open(filePath, 'wb') as f:
                self._write_to(f)

    def to_bytes_io(self, gzipped=False):
        """
        A method to write the vector-tile data to a file-like object (a string buffer).

        :param gzipped: Whether the content should be gzipped or not. Default is ``False``.
        """
        f = io.BytesIO()
        self._write_to(f)
        if gzipped:
            f = gzipFileObject(f)
        return f

    def _write_to(self, f):
        """
        A private method to write the terrain tile to a file or file-like object.

        :param f: The file-like object
        """
        self.featureTable['MINIMUM_HEIGHT'] = self.bounding_volume['minimum_height']
        self.featureTable['MAXIMUM_HEIGHT'] = self.bounding_volume['maximum_height']
        self.featureTable['RECTANGLE'] = [self.bounding_volume['west'],
                                          self.bounding_volume['south'],
                                          self.bounding_volume['east'],
                                          self.bounding_volume['north']]
        self.featureTable['REGION'] = [self.bounding_volume['west'],
                                       self.bounding_volume['south'],
                                       self.bounding_volume['east'],
                                       self.bounding_volume['north'],
                                       self.bounding_volume['minimum_height'],
                                       self.bounding_volume['maximum_height']]
        polygon_positions_container = self.prepare_polygons()
        polyline_positions_container = self.prepare_polylines()
        point_positions_container = self.prepare_points()

        indices_binary = polygon_positions_container['indices_buffer']
        polygon_positions = polygon_positions_container['positions_buffer']
        polyline_positions = polyline_positions_container['positions_buffer']
        point_positions = point_positions_container['positions_buffer']

        if polygon_positions is not None:
            self.featureTable['POLYGON_COUNTS'] = polygon_positions_container['counts']
            self.featureTable['POLYGON_INDEX_COUNTS'] = polygon_positions_container['indexCounts']

        if polyline_positions is not None:
            self.featureTable['POLYLINE_COUNTS'] = polyline_positions_container['counts']

        feature_table_json = json.dumps(self.featureTable)
        feature_table_binary = packEntry("I", 0)
        batch_table_json = json.dumps(self.create_batch_table())
        batch_table_binary = packEntry("I", 0)

        indices_binary_byte_length = 0 if indices_binary is None else len(indices_binary)
        self.header['byteLength'] = VectorTile.header_byte_length + \
                                    utf8_byte_len(feature_table_json) + \
                                    len(feature_table_binary) + \
                                    utf8_byte_len(batch_table_json) + \
                                    len(batch_table_binary) + \
                                    indices_binary_byte_length

        self.header['featureTableJsonByteLength'] = utf8_byte_len(feature_table_json)
        self.header['featureTableBinaryByteLength'] = len(feature_table_binary)
        self.header['batchTableJsonByteLength'] = utf8_byte_len(batch_table_json)
        self.header['batchTableBinaryByteLength'] = len(batch_table_binary)
        self.header['indicesByteLength'] = indices_binary_byte_length
        self.header['polygonPositionsByteLength'] = 0 if polygon_positions is None else int(len(polygon_positions) / 2)
        self.header['polylinePositionsByteLength'] = 0 if polyline_positions is None else int(
            len(polyline_positions) / 2)
        self.header['pointPositionsByteLength'] = 0 if point_positions is None else int(len(point_positions) / 3)

        # begin write
        # Header
        for k, v in self.vector_tile_header.items():
            type_format = v
            entry = self.header[k]
            f.write(packEntry(type_format, entry))

        f.write(bytes(feature_table_json, 'utf-8'))
        f.write(feature_table_binary)
        f.write(bytes(batch_table_json, 'utf-8'))
        f.write(batch_table_binary)
        if indices_binary is not None:
            f.write(indices_binary)
        if polygon_positions is not None:
            f.write(polygon_positions)
        if polyline_positions is not None:
            f.write(polyline_positions)
        if point_positions is not None:
            f.write(point_positions)

    def _update_extent(self, coordinates: []):
        if type(coordinates[0]) in [list, tuple]:
            west = min([c[0] for c in coordinates])
            east = max([c[0] for c in coordinates])
            north = max([c[1] for c in coordinates])
            south = min([c[1] for c in coordinates])
            min_height = min([c[2] for c in coordinates])
            max_height = max([c[2] for c in coordinates])
        else:
            west = coordinates[0]
            east = coordinates[0]
            north = coordinates[1]
            south = coordinates[1]
            min_height = coordinates[2]
            max_height = coordinates[2]

        self.bounding_volume['west'] = west if self.bounding_volume['west'] is None else min(
            self.bounding_volume['west'], west)
        self.bounding_volume['east'] = east if self.bounding_volume['east'] is None else max(
            self.bounding_volume['east'], east)
        self.bounding_volume['north'] = north if self.bounding_volume['north'] is None else max(
            self.bounding_volume['north'], north)
        self.bounding_volume['south'] = south if self.bounding_volume['south'] is None else min(
            self.bounding_volume['south'], south)
        self.bounding_volume['minimum_height'] = min_height if self.bounding_volume['minimum_height'] is None else min(
            self.bounding_volume['minimum_height'], min_height)
        self.bounding_volume['maximum_height'] = max_height if self.bounding_volume['maximum_height'] is None else max(
            self.bounding_volume['maximum_height'], max_height)

    def create_batch_table(self) -> {}:
        batch_table = {}

        for property_name in self.property_names_to_publish:
            values = [f["properties"][property_name] for f in self.points] + \
                     [f["properties"][property_name] for f in self.polylines] + \
                     [f["properties"][property_name] for f in self.polygons]

            batch_table[property_name] = values

        return batch_table

    def prepare_polygons(self):
        polygon_coordinates = []
        polygon_indices = []
        polygon_coordinates_count = []
        polygon_index_counts = []
        prepared_polygons = {'positions_buffer': None,
                             'indices_buffer': None,
                             'counts': [],
                             'indexCounts': []}

        if len(self.polygons) == 0:
            return prepared_polygons

        for feature in self.polygons:
            coordinates = feature["geometry"]["coordinates"]
            coordinates2d = [[c[0], c[1]] for c in coordinates]
            data = earcut.flatten(coordinates2d)
            indices = earcut.earcut(data['vertices'], data['holes'], data['dimensions'])
            polygon_coordinates.extend(data['vertices'])
            polygon_indices.extend(indices)
            polygon_coordinates_count.append(len(data['vertices']))
            polygon_index_counts.append(len(indices))

        prepared_polygons['positions_buffer'] = self._encode_polygon_positions(polygon_coordinates)
        prepared_polygons['indices_buffer'] = [packEntry('B', i) for i in polygon_indices]
        prepared_polygons['counts'] = polygon_coordinates_count
        prepared_polygons['index_counts'] = polygon_index_counts

        return prepared_polygons

    def prepare_polylines(self):
        polyline_coordinates = []
        polyline_coordinates_count = []

        prepared_polyline = {'positions_buffer': None,
                             'counts': []}

        if len(self.polylines) == 0:
            return prepared_polyline

        for feature in self.polylines:
            coordinates = feature["geometry"]["coordinates"]
            data = earcut.flatten(coordinates)
            polyline_coordinates.extend(data['vertices'])
            polyline_coordinates_count.append(len(data['vertices']))

        prepared_polyline['positions_buffer'] = self._encode_polyline_positions(polyline_coordinates)
        prepared_polyline['counts'] = polyline_coordinates_count

        return prepared_polyline

    def prepare_points(self):
        prepared_points = {'positions_buffer': None,
                           'property_name': 'POSITION_QUANTIZED'}

        u_buffer = packEntry('I', 0);
        v_buffer = packEntry('I', 0);
        h_buffer = packEntry('I', 0);
        last_v = 0
        last_u = 0
        last_h = 0

        for feature in self.points:
            coordinate = feature["geometry"]["coordinates"]
            u, v, h = self._encode_point_position(coordinate)

            zig_zag_u = zigZagEncode(u - last_u)
            zig_zag_v = zigZagEncode(v - last_v)
            zig_zag_h = zigZagEncode(h - last_h)
            u_buffer += packEntry('I', zig_zag_u)
            v_buffer += packEntry('I', zig_zag_v)
            h_buffer += packEntry('I', zig_zag_h)

            last_u = u
            last_v = v
            last_h = h

        prepared_points['positions_buffer'] = u_buffer + v_buffer + h_buffer

        return prepared_points

    def _encode_polygon_positions(self, polygon_coordinates):
        u_buffer = ""
        v_buffer = ""
        last_v = 0
        last_u = 0
        for c in polygon_coordinates:
            longitude = c[0]
            latitude = c[1]
            u = (longitude - self.bounding_volume['west']) / self.bounding_volume['west']
            v = (latitude - self.bounding_volume['south']) / self.bounding_volume['south']

            u = clamp(u, 0.0, 1.0)
            v = clamp(v, 0.0, 1.0)

            zig_zag_u = zigZagEncode(u - last_u)
            zig_zag_v = zigZagEncode(v - last_v)
            u_buffer += packEntry('B', zig_zag_u)
            v_buffer += packEntry('B', zig_zag_v)

            last_u = u
            last_v = v

        return u_buffer + v_buffer

    def _encode_polyline_positions(self, polyline_coordinates):
        u_buffer = ""
        v_buffer = ""
        h_buffer = ""
        last_v = 0
        last_u = 0
        last_h = 0

        for c in polyline_coordinates:
            longitude = c[0]
            latitude = c[1]
            height = c[2]
            u = (longitude - self.bounding_volume['west']) / self.bounding_volume['west']
            v = (latitude - self.bounding_volume['south']) / self.bounding_volume['south']
            h = (height - self.bounding_volume['minimum_height']) / \
                (self.bounding_volume['maximum_height'] - self.bounding_volume['minimum_height'])

            u = clamp(u, 0.0, 1.0)
            v = clamp(v, 0.0, 1.0)
            h = clamp(h, 0.0, 1.0)

            u = math.floor(u * UV_RANGE)
            v = math.floor(v * UV_RANGE)
            h = math.floor(h * UV_RANGE)

            zig_zag_u = zigZagEncode(u - last_u)
            zig_zag_v = zigZagEncode(v - last_v)
            zig_zag_h = zigZagEncode(h - last_h)
            u_buffer += packEntry('B', zig_zag_u)
            v_buffer += packEntry('B', zig_zag_v)
            h_buffer += packEntry('B', zig_zag_h)

            last_u = u
            last_v = v
            last_h = h

        return u_buffer + v_buffer + h_buffer

    def _encode_point_position(self, coordinate):
        longitude = coordinate[0]
        latitude = coordinate[1]
        height = coordinate[2]

        bounds_and_origins = self._get_bounds_and_origin(self.bounding_volume)

        delta_longitude = longitude - bounds_and_origins['origin_longitude']
        delta_latitude = latitude - bounds_and_origins['origin_latitude']
        delta_height = height - bounds_and_origins['origin_height']
        v = math.floor((delta_latitude * UV_RANGE) / bounds_and_origins['bounding_latitude'])
        if math.isnan(v):
            v = 0

        u = math.floor((delta_longitude * UV_RANGE) / bounds_and_origins['bounding_longitude'])
        if math.isnan(u):
            u = 0

        if bounds_and_origins['bounding_height'] == 0:
            return u, v, 0
        else:
            h = math.floor((delta_height * UV_RANGE) / bounds_and_origins['bounding_height'])
            return u, v, h

    @staticmethod
    def _get_bounds_and_origin(bounding_volume):

        return {'bounding_longitude': bounding_volume['east'] - bounding_volume['west'],
                'bounding_latitude': bounding_volume['north'] - bounding_volume['south'],
                'bounding_height': bounding_volume['maximum_height'] - bounding_volume['minimum_height'],
                'origin_longitude': bounding_volume['west'],
                'origin_latitude': bounding_volume['south'],
                'origin_height': bounding_volume['minimum_height']
                }
