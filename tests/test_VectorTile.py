# -*- coding: utf-8 -*-
import json
import os
import unittest
import tempfile
from Py3d_Tiles.VectorTile import VectorTile, VectorTileFactory


class TestVectorTile(unittest.TestCase):
    def setUp(self):
        self.vector_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'data/test.vctr')
        self.temp_directory_path = tempfile.mkdtemp()
        self.vector_temp_file_path = os.path.join(self.temp_directory_path,
                                                  'temp.vctr')
        self.meta_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'data/geojson/metadata_tileset.json')
        self.geojson_directory_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   'data/geojson')
        if os.path.isfile(self.vector_temp_file_path):
            os.remove(self.vector_temp_file_path)

    def tearDown(self):
        if os.path.isfile(self.vector_temp_file_path):
            os.remove(self.vector_temp_file_path)

    def test_fromFile_expect_valid_Header(self):
        # arrange
        expected_magic = b'vctr'
        expected_byte_length = 484
        expected_feature_table_json_byte_length = 352
        expected_feature_table_binary_byte_length = 0
        expected_batch_table_json_byte_length = 88
        expected_batch_table_binary_byte_length = 0
        expected_indices_byte_length = 0
        expected_polygon_positions_byte_length = 0
        expected_polyline_positions_byte_length = 0
        expected_point_positions_byte_length = 8

        # act
        vector_tile = VectorTile(None)
        vector_tile.from_file(self.vector_file_path, gzipped=False)
        header = vector_tile.header
        actual_magic = header['magic']
        actual_byte_length = header['byteLength']
        actual_feature_table_json_byte_length = header['featureTableJsonByteLength']
        actual_feature_table_binary_byte_length = header['featureTableBinaryByteLength']
        actual_batch_table_json_byte_length = header['batchTableJsonByteLength']
        actual_batch_table_binary_byte_length = header['batchTableBinaryByteLength']
        actual_indices_byte_length = header['indicesByteLength']
        actual_polygon_positions_byte_length = header['polygonPositionsByteLength']
        actual_polyline_positions_byte_length = header['polylinePositionsByteLength']
        actual_point_positions_byte_length = header['pointPositionsByteLength']

        print(vector_tile.featureTable)
        print(vector_tile.batch_table)
        # assert
        self.assertEquals(actual_magic, expected_magic)
        self.assertEquals(actual_byte_length, expected_byte_length)
        self.assertEquals(actual_feature_table_json_byte_length,
                          expected_feature_table_json_byte_length)
        self.assertEquals(actual_feature_table_binary_byte_length,
                          expected_feature_table_binary_byte_length)
        self.assertEquals(actual_batch_table_json_byte_length,
                          expected_batch_table_json_byte_length)
        self.assertEquals(actual_batch_table_binary_byte_length,
                          expected_batch_table_binary_byte_length)
        self.assertEquals(actual_indices_byte_length, expected_indices_byte_length)
        self.assertEquals(actual_polygon_positions_byte_length,
                          expected_polygon_positions_byte_length)
        self.assertEquals(actual_polyline_positions_byte_length,
                          expected_polyline_positions_byte_length)
        self.assertEquals(actual_point_positions_byte_length,
                          expected_point_positions_byte_length)

    def test_toFile_expectOnePointFeature(self):
        # arrange
        expected_magic = b'vctr'

        one_feature = {"type": "Feature",
                       "geometry": {"type": "Point",
                                    "coordinates": [9.6718497312, 50.4135642177, -9999.0000353837]},
                       "properties": {"foo": "baz",
                                      "bar": "37.5"},
                       }

        # act
        vector_tile = VectorTile(None, property_names_to_publish=["foo", "bar"])
        vector_tile.add_feature(one_feature)
        vector_tile.to_file(self.vector_temp_file_path, gzipped=False)

        vector_tile = VectorTile(None)
        vector_tile.from_file(self.vector_temp_file_path)

        # assert
        self.assertEquals(1, len(vector_tile.points))

    def test_toFile_expectTwoPointFeatures(self):
        # arrange
        features = [{"type": "Feature",
                     "geometry": {"type": "Point",
                                  "coordinates": [9.6718497312, 50.4135642177, -9999.0000353837]},
                     "properties": {"foo": "baz",
                                    "bar": "37.5"},
                     },
                    {"type": "Feature",
                     "geometry": {"type": "Point",
                                  "coordinates": [12.9135709195, 49.0548060332, 449.3999660118]},
                     "properties": {"foo": "baz",
                                    "bar": "8"},
                     }
                    ]

        # act
        vector_tile = VectorTile(None, property_names_to_publish=["foo", "bar"])
        for feature in features:
            vector_tile.add_feature(feature)
        vector_tile.to_file(self.vector_temp_file_path, gzipped=False)

        vector_tile = VectorTile(None)
        vector_tile.from_file(self.vector_temp_file_path)

        print(vector_tile.featureTable)
        print(vector_tile.batch_table)
        # assert
        self.assertEquals(2, len(vector_tile.points))

    def test_create_Tileset(self):
        # arrange
        metadata_path = self.meta_file_path
        source_path = self.geojson_directory_path
        destination_path = self.temp_directory_path
        # act
        property_names_to_publish = ["name"]
        factory = VectorTileFactory(metadata_path=metadata_path,
                                    source_path=source_path,
                                    destination_path=destination_path,
                                    property_names_to_publish=property_names_to_publish)
        factory.create_tileset(node_limit=1000)

        # assert
        self.assertIsNotNone(factory)
