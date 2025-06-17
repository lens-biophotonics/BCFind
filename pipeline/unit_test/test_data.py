import unittest
from ..data import get_start_end_indicies

class TestgetStartEndIndicies(unittest.TestCase):
    # def test_full_range(self):
    #     # Basic full slice sanity check
    #     # n -> 10
    #     # 0% → 0, 100% → 10 (n)
    #     self.assertEqual(get_start_end_indicies(0, 100, 10), (0, 10))
    
    # def test_empty_slice_start(self):
    #     # Zero-length range
    #     # n -> 10
    #     # 0% -> 0, 0% -> 0
    #     self.assertEqual(get_start_end_indicies(0,0, 10), (0, 0))

    # def test_empty_slice_middle(self):
    #     # when start == end
    #     # n -> 10
    #     # 50% -> 5, 50% -> 5
    #     self.assertEqual(get_start_end_indicies(50, 50, 10), (5, 5))

    # def test_empty_slice_end(self):
    #     # Zero-length range at the end
    #     # n -> 10
    #     # 100% -> 10 (n), 100% -> 10 (n)
    #     self.assertEqual(get_start_end_indicies(100, 100, 10), (10, 10))


    # def test_exact_integer_boundaries(self):
    #     # No floor/ciel ambiguity (solid numbers)
    #     # n -> 8
    #     # 25% -> 2, 75% -> 6
    #     self.assertEqual(get_start_end_indicies(25, 75, 8), (2, 6))

    # def test_fraction_floor_ceil(self):
    #     # Verify floor and ceil logic
    #     # n -> 7
    #     # 30% -> floor(2.1) -> 2, 70% -> ceil(4.9) -> 5
    #     self.assertEqual(get_start_end_indicies(30, 70, 7), (2, 5))

    # def test_fraction_floor_ceil_middle(self):
    #     # Verify floor and ceil logic when both the pct are half
    #     # n -> 7
    #     # 50% -> floor(3.5) -> 3, 50% -> ceil(3.5) -> 4
    #     self.assertEqual(get_start_end_indicies(50, 50, 7), (3, 4))

    # def test_start_clamp(self):
    #     # negative values clamps to 0
    #     # n -> 20
    #     # -20% -> 0, 50% -> 10
    #     self.assertEqual(get_start_end_indicies(-20, 50, 20), (0, 10))

    # def test_end_clamp(self):
    #     # overshoot value calmps to n
    #     # n -> 20
    #     # 50% -> 10, 200% -> 20
    #     self.assertEqual(get_start_end_indicies(50, 200, 20), (10, 20))

    # def test_end_less_than_start_raises(self):\
    #     # raise ValueError number of patches is less than 1
    #     # n -> 0
    #     # 0% -> ValueError, 100% -> ValueError
    #     with self.assertRaises(ValueError):
    #         get_start_end_indicies(0, 100, 0)

    # def test_invalid_n_of_patches_raises(self):\
    #     # raise ValueError when end < start
    #     # n -> 10
    #     # 60% -> ValueError, 40% -> ValueError
    #     with self.assertRaises(ValueError):
    #         get_start_end_indicies(60, 40, 10)

    # def test_single_patch_full_range(self):
    #     # single patch full slice sanity check
    #     # n -> 1
    #     # 0% -> floor(0) -> 0, 100% -> ceil(1) -> 1
    #     self.assertEqual(get_start_end_indicies(0,100, 1), (0, 1))

    # def test_single_patch_full_range_middle(self):
    #     # single patch full slice when start == end (middle value)
    #     # n -> 1
    #     # 50% -> floor(0.5) -> 0, 50% -> ceil(0.5) -> 1
    #     self.assertEqual(get_start_end_indicies(0,100, 1), (0, 1))

    # def test_single_patch_empty_slice(self):
    #     # single patch zero_length range (end)
    #     # n -> 1
    #     # 100% -> floor(1) -> 1, 100% -> ceil(1) -> 1
    #     self.assertEqual(get_start_end_indicies(100, 100, 1), (1, 1))

    # def test_single_patch_clamp(self):
    #     # single patch undershoot and overshoot clamp to start and end bound
    #     # n -> 1
    #     # -10% -> 0, 110% -> 1
    #     self.assertEqual(get_start_end_indicies(-10, 110, 1), (0, 1))

    def test_single_patch_clamp(self):
        # single patch undershoot and overshoot clamp to start and end bound
        # n -> 1
        # -10% -> 0, 110% -> 1
        print(get_start_end_indicies(0, 10, 143647))
        self.assertEqual(get_start_end_indicies(10, 110, 143647), (0, 1))

if __name__ == '__main__':
    unittest.main()