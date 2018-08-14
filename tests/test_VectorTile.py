# -*- coding: utf-8 -*-
import os
import unittest
from Cesium_3d_tiles_py.VectorTile import VectorTile


class TestVectorTile(unittest.TestCase):
    def setUp(self):
        self.vector_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/test.vctr')

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
        vector_tile = VectorTile()
        vector_tile.fromFile(self.vector_file_path, gzipped=False)
        actual_header = vector_tile.header
        actual_magic = actual_header['magic']
        actual_byte_length = actual_header['byteLength']
        actual_feature_table_json_byte_length = actual_header['featureTableJsonByteLength']
        actual_feature_table_binary_byte_length = actual_header['featureTableBinaryByteLength']
        actual_batch_table_json_byte_length = actual_header['batchTableJsonByteLength']
        actual_batch_table_binary_byte_length = actual_header['batchTableBinaryByteLength']
        actual_indices_byte_length  = actual_header['indicesByteLength']
        actual_polygon_positions_byte_length = actual_header['polygonPositionsByteLength']
        actual_polyline_positions_byte_length = actual_header['polylinePositionsByteLength']
        actual_point_positions_byte_length = actual_header['pointPositionsByteLength']

        # assert
        self.assertEquals(actual_magic, expected_magic)
        self.assertEquals(actual_byte_length, expected_byte_length)
        self.assertEquals(actual_feature_table_json_byte_length, expected_feature_table_json_byte_length)
        self.assertEquals(actual_feature_table_binary_byte_length, expected_feature_table_binary_byte_length)
        self.assertEquals(actual_batch_table_json_byte_length, expected_batch_table_json_byte_length)
        self.assertEquals(actual_batch_table_binary_byte_length, expected_batch_table_binary_byte_length)
        self.assertEquals(actual_indices_byte_length, expected_indices_byte_length)
        self.assertEquals(actual_polygon_positions_byte_length,expected_polygon_positions_byte_length)
        self.assertEquals(actual_polyline_positions_byte_length, expected_polyline_positions_byte_length)
        self.assertEquals(actual_point_positions_byte_length, expected_point_positions_byte_length)