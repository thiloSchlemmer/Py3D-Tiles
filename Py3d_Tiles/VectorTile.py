# -*- coding: utf-8 -*-
import gzip
import io
import json
from collections import OrderedDict

from Py3d_Tiles import earcut
from .utils import unpackEntry, ungzipFileObject, gzipFileObject, packEntry

MIN = 0.0
MAX = 32767.0


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

    def __init__(self, property_names_to_publish=None):
        if property_names_to_publish is None:
            property_names_to_publish = []
        self.featureTable = VectorTile.default_feature_table
        self.property_names_to_publish = property_names_to_publish
        self.extent = None
        self.features = []
        self.points = []
        self.polylines = []
        self.polygons = []
        self.header = OrderedDict()
        for k, v in self.vector_tile_header.items():
            self.header[k] = 0.0
        self.header['magic'] = VectorTile.VECTOR_TILE_MAGIC
        self.header['version'] = VectorTile.VECTOR_TILE_VERSION

    def from_bytes_io(self, f):
        # Header
        for k, v in VectorTile.vector_tile_header.items():
            self.header[k] = unpackEntry(f, v)

        # featureTable
        feature_table_bytes = b''

        feature_table_byte_length = self.header['featureTableJsonByteLength']
        for i in range(0, feature_table_byte_length):
            feature_table_bytes += unpackEntry(f, 's')
        self.featureTable = json.loads(feature_table_bytes.decode('utf-8'))

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
        A method to add a feature to the vector tile. It is assumed that the featue is
        a geojson-node (with properties- and geometry-node)
        :param feature:  the (geojson) feature to add to the vectortile
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

    def to_bytes_io(self, gzipped=False):
        """
        A method to write the terrain tile data to a file-like object (a string buffer).

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
        self.featureTable['MINIMUM_HEIGHT'] = self.extent['minimum_height']
        self.featureTable['MAXIMUM_HEIGHT'] = self.extent['maximum_height']
        self.featureTable['RECTANGLE'] = [self.extent['west'],
                                          self.extent['south'],
                                          self.extent['east'],
                                          self.extent['north']]
        self.featureTable['REGION'] = [self.extent['west'],
                                       self.extent['south'],
                                       self.extent['east'],
                                       self.extent['north'],
                                       self.extent['minimum_height'],
                                       self.extent['maximum_height']]

        # Header
        for k, v in self.vector_tile_header.items():
            f.write(packEntry(v, self.header[k]))

        batch_table_json = json.dumps(self.create_batch_table())

        polygon_positions_container = self.prepare_polygons()
        polyline_positions_container = self.prepare_polylines()

    def _update_extent(self, coordinates: []):
        west = min([c[0] for c in coordinates])
        south = min([c[1] for c in coordinates])
        east = max([c[0] for c in coordinates])
        north = max([c[1] for c in coordinates])
        min_height = min([c[2] for c in coordinates])
        max_height = max([c[2] for c in coordinates])

        if self.extent is None:
            self.extent = {'west': west,
                           'south': south,
                           'east': east,
                           'north': north,
                           'minimum_height': min_height,
                           'maximum_height': max_height
                           }
        else:
            self.extent = {'west': min(self.extent['west'], west),
                           'south': min(self.extent['south'], south),
                           'east': max(self.extent['east'], east),
                           'north': max(self.extent['north'], north),
                           'minimum_height': min(self.extent['minimum_height'], min_height),
                           'maximum_height': max(self.extent['maximum_height'], max_height)
                           }

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
        prepared_polygons = {'positionsBuffer': packEntry('B', 0),
                             'indicesBuffer': packEntry('B', 0),
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

        prepared_polygons['positionsBuffer'] = self._encode_polygon_positions(polygon_coordinates)
        prepared_polygons['indicesBuffer'] = [packEntry('B', i) for i in polygon_indices]
        prepared_polygons['counts'] = polygon_coordinates_count
        prepared_polygons['indexCounts'] = polygon_index_counts

        return prepared_polygons

    def _encode_polygon_positions(self, polygon_coordinates):

        return []
