"""Synthetic overlap and request-boundary checks; no CUDA device is required."""
import sqlite3
import unittest

from nsys_window import summarize, union_ns


class WindowTests(unittest.TestCase):
    def test_interval_union(self):
        self.assertEqual(union_ns([(8, 10), (0, 4), (2, 6), (2, 3)]), 8)
        self.assertEqual(union_ns([(0, 10), (2, 3)]), 10)
        self.assertEqual(union_ns([]), 0)

    def test_window_clips_kernels_and_unions_copies(self):
        db = sqlite3.connect(':memory:')
        db.executescript('''
            CREATE TABLE TARGET_INFO_SESSION_START_TIME(utcEpochNs INTEGER);
            INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (1000);
            CREATE TABLE StringIds(id INTEGER,value TEXT);
            INSERT INTO StringIds VALUES(1,'kernel'),(2,'cuLaunchKernel');
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,demangledName INTEGER,correlationId INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(5,15,1,1),(12,18,1,2),(30,40,1,3);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(17,19);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(9,11,2);
        ''')
        result = summarize(db, 1010, 1020)
        self.assertEqual(result['kernel_count'], 2)
        self.assertAlmostEqual(result['kernel_sum_ms'], 11 / 1e6)
        self.assertAlmostEqual(result['kernel_busy_ms'], 8 / 1e6)
        self.assertAlmostEqual(result['device_busy_ms'], 9 / 1e6)
        self.assertAlmostEqual(result['device_idle_ms'], 1 / 1e6)
        self.assertAlmostEqual(result['apis'][0]['ms'], 1 / 1e6)
        with self.assertRaises(ValueError):
            summarize(db, 1020, 1010)


if __name__ == '__main__':
    unittest.main()
