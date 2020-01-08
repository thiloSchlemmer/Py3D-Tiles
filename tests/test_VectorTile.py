# -*- coding: utf-8 -*-
import os
import unittest
from Py3d_Tiles.VectorTile import VectorTile


class TestVectorTile(unittest.TestCase):
    def setUp(self):
        self.vector_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'data/test.vctr')
        self.vector_temp_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'data/temp.vctr')

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
        expected_byte_length = 324
        expected_feature_table_json_byte_length = 352
        expected_feature_table_binary_byte_length = 0
        expected_batch_table_json_byte_length = 88
        expected_batch_table_binary_byte_length = 0
        expected_indices_byte_length = 0
        expected_polygon_positions_byte_length = 0
        expected_polyline_positions_byte_length = 0
        expected_point_positions_byte_length = 8
        one_feature = {"type": "Feature",
                       "geometry": {"type": "Point",
                                    "coordinates": [9.6718497312, 50.4135642177, -9999.0000353837]},
                       "properties": {"srs": "EPSG:25832",
                                      "objektart": "ortsname",
                                      "class": "37.5",
                                      "name": "Hühnerfarm",
                                      "align": "Left"},
                       }

        # act
        vector_tile = VectorTile(None)
        vector_tile.add_feature(one_feature)
        vector_tile.to_file(self.vector_temp_file_path, gzipped=False)

        vector_tile = VectorTile(None)
        vector_tile.from_file(self.vector_temp_file_path)
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

        # assert
        self.assertTrue(False)
