# -*- coding: utf-8 -*-
import gzip
import io
import json
import math
import os
from collections import OrderedDict

from Py3d_Tiles import earcut
from .utils import unpackEntry, ungzipFileObject, gzipFileObject, packEntry, clamp, zigZagEncode, utf8_byte_len, \
    zigZagDecode, radian_to_degree, degree_to_radian

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


def get_string_padded(s, padding_size=8):
    byte_offset = 0
    boundary = padding_size

    byte_length = utf8_byte_len(s)
    remainder = (byte_offset + byte_length) % boundary
    padding = 0 if remainder == 0 else boundary - remainder

    return s + ' ' * padding


def _add_region(region, parent):
    if parent:
        return [min(region[0], parent[0]),
                min(region[1], parent[1]),
                max(region[2], parent[2]),
                max(region[3], parent[3]),
                min(region[4], parent[4]),
                max(region[5], parent[5]),
                ]
    else:
        return region


def read_geojson(geojson_path):
    with open(geojson_path, mode='r') as f:
        features = json.load(f)["features"]

    print("Reading {} features from geojson '{}':".format(len(features), geojson_path))
    return features


class VectorTileFactory(object):
    GEOMETRIC_ERRORS = {
        4: 77067.3397799586,
        5: 38533.66989,
        6: 19266.83494,
        7: 9633.417472,
        8: 4816.708736247414,
        9: 2408.354368123707,
        10: 1204.1771840618535,
        11: 602.0885920309267,
        12: 301.0442960154634,
        13: 150.5221480077317,
        14: 75.26107400386584,
        15: 37.63053700193292,
        16: 18.81526850096646,
        17: 9.40763425048323}

    def __init__(self, metadata_path, source_path, destination_path, property_names_to_publish=None):
        self.tileset_count = 0
        self.metadata = None
        self.source_path = source_path
        self.destination_path = destination_path
        self.property_names_to_publish = property_names_to_publish

        os.makedirs(self.destination_path, exist_ok=True)

        assert os.path.exists(metadata_path) & os.path.isfile(metadata_path)
        assert os.path.exists(self.source_path) & os.path.isdir(self.source_path)

        with open(metadata_path, mode='r') as f:
            self.metadata = json.load(f)

    def _create_tileset_path(self):
        tileset_name = "tileset.json"
        if 0 < self.tileset_count:
            tileset_name = "tileset_{}.json".format(self.tileset_count)

        self.tileset_count += 1

        return os.path.join(self.destination_path, tileset_name)

    def create_tileset(self, node_limit=400):
        self._create_tileset(self.metadata, node_limit)

    def _create_tileset(self, data, node_limit, node_count=0):
        # build tileset json
        children = data["children"]
        tileset_json_path = self._create_tileset_path()
        tileset_node, child_node_count = self._add_tiles(data, node_limit, node_count + len(children))
        root = tileset_node

        tileset_data = {
            "asset": {
                "version": '1.0',
                "tilesetVersion": '1.0'
            },
            "geometricError": root["geometricError"],
            "root": root
        }

        tileset_name = self._save_tileset(tileset_data, tileset_json_path)
        nodes = {"boundingVolume": {"region": root["boundingVolume"]["region"]},
                 "geometricError": root["geometricError"],
                 "refine": "ADD",
                 "content": {"uri": tileset_name}}
        return nodes

    def _add_tiles(self, data, node_limit, node_count):
        # build nodes from data
        tile = data['tile']
        region = [degree_to_radian(data['bounds'][0]),
                  degree_to_radian(data['bounds'][1]),
                  degree_to_radian(data['bounds'][2]),
                  degree_to_radian(data['bounds'][3]),
                  data['bounds'][4],
                  data['bounds'][5]]
        z, x, y = tile[0], tile[1], tile[2]

        children = data["children"]
        current_node_count = node_count + len(children)

        aggregated_region = None
        child_nodes = []

        if current_node_count <= node_limit:
            for child in children:
                if current_node_count <= node_limit:
                    child_node, child_node_count = self._add_tiles(child, node_limit=node_limit,
                                                                   node_count=current_node_count)
                    child_region = child_node["boundingVolume"]["region"]
                    if child_region:
                        aggregated_region = _add_region(child_region, aggregated_region)
                        child_nodes.append(child_node)
                        current_node_count += child_node_count
                else:
                    child_node = self._create_tileset(child, node_limit, current_node_count)
                    child_region = child_node["boundingVolume"]["region"]
                    if child_region:
                        aggregated_region = _add_region(child_region, aggregated_region)
                        child_nodes.append(child_node)
        else:
            for child in children:
                child_node = self._create_tileset(child, node_limit, current_node_count)
                child_region = child_node["boundingVolume"]["region"]
                if child_region:
                    aggregated_region = _add_region(child_region, aggregated_region)
                    child_nodes.append(child_node)

        nodes = None
        if tile != 'root':
            tile_extent = {'west': region[0],
                           'south': region[1],
                           'east': region[2],
                           'north': region[3],
                           }
            tile_region = [region[0],
                           region[1],
                           region[2],
                           region[3],
                           region[4],
                           region[5]
                           ]
            nodes = {"boundingVolume": {"region": tile_region},
                     "geometricError": VectorTileFactory.GEOMETRIC_ERRORS[z],
                     "refine": "ADD",
                     "children": child_nodes}

            output_filename = "{}_{}_{}.vctr".format(z, x, y)
            geojson_path = os.path.join(self.source_path, "{}_{}_{}.json".format(z, x, y))
            vctr_path = os.path.join(self.destination_path, output_filename)
            if os.path.exists(geojson_path) & os.path.isfile(geojson_path):
                vctr_tile = VectorTile(tile_extent, self.property_names_to_publish)
                features = read_geojson(geojson_path)
                for feature in features:
                    vctr_tile.add_feature(feature)

                if os.path.exists(vctr_path):
                    os.remove(vctr_path)
                vctr_tile.to_file(vctr_path)
                nodes["content"] = {"uri": output_filename}

        else:
            nodes = {"boundingVolume": {
                "region": aggregated_region
            },
                "geometricError": 4816.708736247414,
                "refine": "ADD",
                "children": child_nodes}

        return nodes, current_node_count

    def _save_tileset(self, tileset_data, tileset_json_path):
        with open(tileset_json_path, mode='w') as f:
            json.dump(tileset_data, f)
        return os.path.basename(tileset_json_path)


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
        self.existing_property_names = set()
        if tile_extent is None:
            tile_extent = {'west': None,
                           'south': None,
                           'east': None,
                           'north': None,
                           }
        self.featureTable = dict(VectorTile.default_feature_table)
        self.property_names_to_publish = [] if property_names_to_publish is None else property_names_to_publish

        self.extent = tile_extent

        self.point_features = []
        self.polyline_features = []
        self.polygon_features = []

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
        self.batch_table = {}

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

        batch_table_bytes = b''
        batch_table_json_byte_length = self.header['batchTableJsonByteLength']
        print("reading BatchTable...")
        for i in range(0, batch_table_json_byte_length):
            batch_table_bytes += unpackEntry(f, 's')
        batch_table_json = json.loads(batch_table_bytes.decode('utf-8'))
        print("...Found {} Properties".format(len(batch_table_json.keys())))
        self.batch_table = batch_table_json
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

        polygon_positions_byte_length = self.header['polygonPositionsByteLength']
        polygon_count = polygon_positions_byte_length / 8
        if polygon_count != self.featureTable['POLYGONS_LENGTH']:
            polygon_count = self.featureTable['POLYGONS_LENGTH']
        if polygon_positions_byte_length != 0:
            polygon_u = []
            polygon_v = []
            for ud in unpack_and_decode_position(f, polygon_count, 'H'):
                polygon_u.append(ud)
            for vd in unpack_and_decode_position(f, polygon_count, 'H'):
                polygon_v.append(vd)

            for i in range(len(polygon_u)):
                u = polygon_u[i]
                v = polygon_v[i]
                rad_longitude = (u * self.bounding_volume['west']) + self.bounding_volume['west']
                rad_latitude = (v * self.bounding_volume['south']) + self.bounding_volume['south']

                self.polygons.append([rad_longitude, rad_latitude])

        polyline_positions_byte_length = self.header['polylinePositionsByteLength']
        polyline_count = polyline_positions_byte_length / 8
        if polyline_count != self.featureTable['POLYLINES_LENGTH']:
            polyline_count = self.featureTable['POLYLINES_LENGTH']
        if polyline_positions_byte_length != 0:
            polyline_u = []
            polyline_v = []
            polyline_h = []
            for ud in unpack_and_decode_position(f, polyline_count, 'H'):
                polyline_u.append(ud)
            for vd in unpack_and_decode_position(f, polyline_count, 'H'):
                polyline_v.append(vd)
            for hd in unpack_and_decode_position(f, polyline_count, 'H'):
                polyline_h.append(hd)

            for i in range(len(polyline_u)):
                u = polyline_u[i]
                v = polyline_v[i]
                h = polyline_h[i]
                rad_longitude = (u * self.bounding_volume['west']) + self.bounding_volume['west']
                rad_latitude = (v * self.bounding_volume['south']) + self.bounding_volume['south']
                rad_height = (h * self.bounding_volume['minimum_height']) + self.bounding_volume['minimum_height']

                self.polylines.append([rad_longitude, rad_latitude, rad_height])

        point_positions_byte_length = self.header['pointPositionsByteLength']
        point_count = point_positions_byte_length / 8
        if point_count != self.featureTable['POINTS_LENGTH']:
            point_count = self.featureTable['POINTS_LENGTH']
        if point_positions_byte_length:
            point_u = []
            point_v = []
            point_h = []
            for ud in unpack_and_decode_position(f, point_count, 'H'):
                point_u.append(ud)
            for vd in unpack_and_decode_position(f, point_count, 'H'):
                point_v.append(vd)
            for hd in unpack_and_decode_position(f, point_count, 'H'):
                point_h.append(hd)

            for i in range(len(point_u)):
                uvh = point_u[i], point_v[i], point_h[i]
                rad_longitude, rad_latitude, height = self._decode_point_position(uvh)

                longitude = radian_to_degree(rad_longitude)
                latitude = radian_to_degree(rad_latitude)
                self.points.append([longitude, latitude, height])

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
        self._register_property_name(feature)
        if geometry_type == "Point":
            self.point_features.append(feature)
            self.featureTable['POINTS_LENGTH'] += 1
        elif geometry_type == "LineString":
            self.polyline_features.append(feature)
            self.featureTable['POLYLINES_LENGTH'] += 1
        elif geometry_type == "Polygon":
            self.polygon_features.append(feature)
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
        polygon_positions_length = 0 if polygon_positions is None else len(polygon_positions)
        polyline_positions_length = 0 if polyline_positions is None else len(polyline_positions)
        point_positions_length = 0 if point_positions is None else len(point_positions)

        print("Content (byte): {} for polygons, {} for polylines, {} for points".format(polygon_positions_length,
                                                                                        polyline_positions_length,
                                                                                        point_positions_length))
        if polygon_positions is not None:
            self.featureTable['POLYGON_COUNTS'] = polygon_positions_container['counts']
            self.featureTable['POLYGON_INDEX_COUNTS'] = polygon_positions_container['indexCounts']

        if polyline_positions is not None:
            self.featureTable['POLYLINE_COUNTS'] = polyline_positions_container['counts']

        feature_table_json = get_string_padded(json.dumps(self.featureTable))
        print("Write FeatureTable-Content:{}".format(feature_table_json))
        feature_table_binary = None

        self.batch_table = self.create_batch_table()
        batch_table_json =  get_string_padded(json.dumps(self.batch_table))
        batch_table_binary = None

        indices_binary_byte_length = 0 if indices_binary is None else len(indices_binary)
        feature_table_binary_length = 0 if feature_table_binary is None else len(feature_table_binary)
        batch_table_binary_length = 0 if batch_table_binary is None else len(batch_table_binary)
        self.header['byteLength'] = VectorTile.header_byte_length + \
                                    utf8_byte_len(feature_table_json) + \
                                    feature_table_binary_length + \
                                    utf8_byte_len(batch_table_json) + \
                                    batch_table_binary_length + \
                                    indices_binary_byte_length

        self.header['featureTableJsonByteLength'] = utf8_byte_len(feature_table_json)
        self.header['featureTableBinaryByteLength'] = feature_table_binary_length
        self.header['batchTableJsonByteLength'] = utf8_byte_len(batch_table_json)
        self.header['batchTableBinaryByteLength'] = batch_table_binary_length
        self.header['indicesByteLength'] = indices_binary_byte_length
        self.header['polygonPositionsByteLength'] = 0 if polygon_positions is None else polygon_positions_length
        self.header['polylinePositionsByteLength'] = 0 if polyline_positions is None else polyline_positions_length
        self.header['pointPositionsByteLength'] = 0 if point_positions is None else point_positions_length

        # begin write
        # Header
        for k, v in self.vector_tile_header.items():
            type_format = v
            entry = self.header[k]
            f.write(packEntry(type_format, entry))

        f.write(bytes(feature_table_json, 'utf-8'))
        if feature_table_binary is not None:
            f.write(feature_table_binary)
        f.write(bytes(batch_table_json, 'utf-8'))
        if batch_table_binary is not None:
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
            west = degree_to_radian(min([c[0] for c in coordinates]))
            east = degree_to_radian(max([c[0] for c in coordinates]))
            north = degree_to_radian(max([c[1] for c in coordinates]))
            south = degree_to_radian(min([c[1] for c in coordinates]))
            min_height = min([c[2] for c in coordinates])
            max_height = max([c[2] for c in coordinates])
        else:
            west = degree_to_radian(coordinates[0])
            east = degree_to_radian(coordinates[0])
            north = degree_to_radian(coordinates[1])
            south = degree_to_radian(coordinates[1])
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
            if property_name in self.existing_property_names:
                values = [f["properties"][property_name] for f in self.polygon_features] + \
                         [f["properties"][property_name] for f in self.polyline_features] + \
                         [f["properties"][property_name] for f in self.point_features]

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

        if len(self.polygon_features) == 0:
            return prepared_polygons

        for feature in self.polygon_features:
            coordinates = feature["geometry"]["coordinates"]
            coordinates2d = [[degree_to_radian(c[0]), degree_to_radian(c[1])] for c in coordinates]
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

        prepared_polylines = {'positions_buffer': None,
                              'counts': []}

        if len(self.polyline_features) == 0:
            return prepared_polylines

        for feature in self.polyline_features:
            coordinates = feature["geometry"]["coordinates"]
            rad_coordinates = [[degree_to_radian(c[0]), degree_to_radian(c[1]), degree_to_radian(c[2])] for c in
                               coordinates]
            data = earcut.flatten(rad_coordinates)
            polyline_coordinates.extend(data['vertices'])
            polyline_coordinates_count.append(len(data['vertices']))

        prepared_polylines['positions_buffer'] = self._encode_polyline_positions(polyline_coordinates)
        prepared_polylines['counts'] = polyline_coordinates_count

        return prepared_polylines

    def prepare_points(self):
        prepared_points = {'positions_buffer': None,
                           'property_name': 'POSITION_QUANTIZED'}

        u_buffer = None
        v_buffer = None
        h_buffer = None
        last_v = 0
        last_u = 0
        last_h = 0

        for feature in self.point_features:
            coordinate = feature["geometry"]["coordinates"]
            rad_coordinate = [degree_to_radian(coordinate[0]),
                              degree_to_radian(coordinate[1]),
                              coordinate[2]]
            u, v, h = self._encode_point_position(rad_coordinate)

            zig_zag_u = zigZagEncode(u - last_u)
            zig_zag_v = zigZagEncode(v - last_v)
            zig_zag_h = zigZagEncode(h - last_h)
            u_buffer = packEntry('I', zig_zag_u) if u_buffer is None else u_buffer + packEntry('I', zig_zag_u)
            v_buffer = packEntry('I', zig_zag_v) if v_buffer is None else v_buffer + packEntry('I', zig_zag_v)
            h_buffer = packEntry('I', zig_zag_h) if h_buffer is None else h_buffer + packEntry('I', zig_zag_h)

            last_u = u
            last_v = v
            last_h = h
            u_buffer_length = len(u_buffer)

        positions_buffer = u_buffer + v_buffer + h_buffer
        prepared_points['positions_buffer'] = positions_buffer

        return prepared_points

    def _encode_polygon_positions(self, polygon_coordinates):
        u_buffer = None
        v_buffer = None
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
            u_buffer = packEntry('I', zig_zag_u) if u_buffer is None else u_buffer + packEntry('I', zig_zag_u)
            v_buffer = packEntry('I', zig_zag_v) if v_buffer is None else v_buffer + packEntry('I', zig_zag_v)

            last_u = u
            last_v = v

        return u_buffer + v_buffer

    def _encode_polyline_positions(self, polyline_coordinates):
        u_buffer = None
        v_buffer = None
        h_buffer = None
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
            u_buffer = packEntry('I', zig_zag_u) if u_buffer is None else u_buffer + packEntry('I', zig_zag_u)
            v_buffer = packEntry('I', zig_zag_v) if v_buffer is None else v_buffer + packEntry('I', zig_zag_v)
            h_buffer = packEntry('I', zig_zag_h) if h_buffer is None else h_buffer + packEntry('I', zig_zag_h)

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

        v = 0 if bounds_and_origins['bounding_latitude'] == 0 else math.floor(
            (delta_latitude * UV_RANGE) / bounds_and_origins['bounding_latitude'])
        if math.isnan(v):
            v = 0

        u = 0 if bounds_and_origins['bounding_longitude'] == 0 else math.floor(
            (delta_longitude * UV_RANGE) / bounds_and_origins['bounding_longitude'])
        if math.isnan(u):
            u = 0

        if bounds_and_origins['bounding_height'] == 0:
            return u, v, 0
        else:
            h = math.floor((delta_height * UV_RANGE) / bounds_and_origins['bounding_height'])
            return u, v, h

    def _decode_point_position(self, uvh):
        u, v, h = uvh

        bounds_and_origins = self._get_bounds_and_origin(self.bounding_volume)
        longitude = (u / UV_RANGE) * bounds_and_origins['bounding_longitude'] + bounds_and_origins['origin_longitude']
        latitude = (v / UV_RANGE) * bounds_and_origins['bounding_latitude'] + bounds_and_origins['origin_latitude']
        height = h * bounds_and_origins['bounding_height'] + bounds_and_origins['origin_height']
        return longitude, latitude, height

    @staticmethod
    def _get_bounds_and_origin(bounding_volume):

        return {'bounding_longitude': bounding_volume['east'] - bounding_volume['west'],
                'bounding_latitude': bounding_volume['north'] - bounding_volume['south'],
                'bounding_height': bounding_volume['maximum_height'] - bounding_volume['minimum_height'],
                'origin_longitude': bounding_volume['west'],
                'origin_latitude': bounding_volume['south'],
                'origin_height': bounding_volume['minimum_height']
                }

    def _register_property_name(self, feature):
        properties = feature["properties"]
        for property_name in properties.keys():
            self.existing_property_names.add(property_name)
