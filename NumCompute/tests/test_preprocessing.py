# pyright: reportArgumentType=false
import numpy as np
import pytest

from numcompute import preprocessing
from numcompute.pipeline import Pipeline


# ---------------------------------------------------------------------
# Contract tests (must hold for every transformer)
# ---------------------------------------------------------------------

def impute_scale_pipeline():
    """Build a fresh transformer-only Pipeline for the contract tests."""
    return Pipeline([
        ("impute", preprocessing.SimpleImputer()),
        ("scale", preprocessing.StandardScaler()),
    ])


ALL_TRANSFORMERS = [
    preprocessing.StandardScaler,
    preprocessing.MinMaxScaler,
    preprocessing.OneHotEncoder,
    preprocessing.SimpleImputer,
    impute_scale_pipeline,
]


@pytest.mark.parametrize("TransformerClass", ALL_TRANSFORMERS, ids=lambda cls: cls.__name__)
def test_fit_transform_equivalence(TransformerClass):
    X = np.array([[1, 10], [3, 30]])
    transformer = TransformerClass()
    X_fit_transform = transformer.fit_transform(X)
    transformer2 = TransformerClass()
    transformer2.fit(X)
    X_transform = transformer2.transform(X)

    np.testing.assert_array_almost_equal(X_fit_transform, X_transform, decimal=6)


@pytest.mark.parametrize("TransformerClass", ALL_TRANSFORMERS, ids=lambda cls: cls.__name__)
def test_1d_array(TransformerClass):
    X = np.array([1, 2, 3])  # 1D array

    preprocessor = TransformerClass()
    with pytest.raises(ValueError, match="Input data must have 2 dimensions. Got shape:"):
        preprocessor.fit(X)


@pytest.mark.parametrize("TransformerClass", ALL_TRANSFORMERS, ids=lambda cls: cls.__name__)
def test_column_count_mismatch(TransformerClass):
    X = np.array([[1, 10], [3, 30]])
    Y = np.array([[1, 20, 5], [1, 10, 6]])  # Mismatched column count

    preprocessor = TransformerClass()
    preprocessor.fit(X)
    with pytest.raises(ValueError, match="Incompatible dimensionality"):
        preprocessor.transform(Y)


@pytest.mark.parametrize("TransformerClass", ALL_TRANSFORMERS, ids=lambda cls: cls.__name__)
def test_input_ndim_mismatch(TransformerClass):
    X = np.array([[1, 10], [3, 30]])
    Y = np.array([[[1, 20], [1, 10]]])  # 3D array

    preprocessor = TransformerClass()
    preprocessor.fit(X)
    with pytest.raises(ValueError, match="Input data must have 2 dimensions. Got shape:"):
        preprocessor.transform(Y)


@pytest.mark.parametrize("TransformerClass", ALL_TRANSFORMERS, ids=lambda cls: cls.__name__)
def test_empty_fit(TransformerClass):
    X = np.array([]).reshape(0, 2)

    preprocessor = TransformerClass()
    with pytest.raises(ValueError, match="Cannot fit .* on an empty array."):
        preprocessor.fit(X)


@pytest.mark.parametrize("TransformerClass", ALL_TRANSFORMERS, ids=lambda cls: cls.__name__)
def test_empty_transform_general(TransformerClass):
    X = np.array([[1, 10], [3, 30]])
    Y = np.array([]).reshape(0, 2)

    preprocessor = TransformerClass()
    preprocessor.fit(X)

    assert preprocessor.transform(Y).shape[0] == 0


@pytest.mark.parametrize("TransformerClass", ALL_TRANSFORMERS, ids=lambda cls: cls.__name__)
def test_transform_without_fit(TransformerClass):
    Y = np.array([[1, 20], [1, 10]])

    preprocessor = TransformerClass()
    with pytest.raises(RuntimeError, match="fit must be called before transform."):
        preprocessor.transform(Y)


@pytest.mark.parametrize("TransformerClass", ALL_TRANSFORMERS, ids=lambda cls: cls.__name__)
def test_return_type(TransformerClass):
    X = np.array([[0, 10], [1, 20]])
    Y = np.array([[1, 20], [0, 10]])

    preprocessor = TransformerClass()
    preprocessor.fit(X)
    Y_transformed = preprocessor.transform(Y)

    assert isinstance(Y_transformed, np.ndarray), "Output should be a numpy array"
    assert Y_transformed.dtype == np.float64, "Output dtype should be float64"


@pytest.mark.parametrize("TransformerClass", ALL_TRANSFORMERS, ids=lambda cls: cls.__name__)
def test_transform_does_not_mutate_input(TransformerClass):
    X = np.array([[1.0, 10.0], [3.0, np.nan]])
    Y = np.array([[1.0, np.nan], [1.0, 10.0]])

    preprocessor = TransformerClass()
    preprocessor.fit(X)
    Y_copy = Y.copy()
    preprocessor.transform(Y)

    np.testing.assert_array_equal(Y, Y_copy, err_msg="Input array was mutated during transform.")


@pytest.mark.parametrize("TransformerClass", ALL_TRANSFORMERS, ids=lambda cls: cls.__name__)
def test_fit_transform_with_y_parameter(TransformerClass):
    X = np.array([[1, 10], [3, 30]])
    y = np.array([0, 1])

    preprocessor = TransformerClass()
    X_fit_transform = preprocessor.fit_transform(X, y=y)
    preprocessor2 = TransformerClass()
    preprocessor2.fit(X, y=y)
    X_transform = preprocessor2.transform(X)

    np.testing.assert_array_almost_equal(X_fit_transform, X_transform, decimal=6)


@pytest.mark.parametrize("TransformerClass", ALL_TRANSFORMERS, ids=lambda cls: cls.__name__)
def test_y_ignored_in_fit(TransformerClass):
    X = np.array([[1, 10], [3, 30]])
    y = np.array([0, 1])

    preprocessor = TransformerClass()
    X_fit_transform = preprocessor.fit_transform(X, y)

    preprocessor2 = TransformerClass()
    preprocessor2.fit(X)
    X_transform = preprocessor2.transform(X)

    np.testing.assert_array_almost_equal(X_fit_transform, X_transform, decimal=6)


class TestStandardScaler:

    # ---------------------------------------------------------------------
    # Basic / expected behaviour
    # ---------------------------------------------------------------------

    def test_basic(self):

        X = np.array([[1, 10], [3, 30]])
        Y = np.array([[2, 20], [1, 10]])

        preprocessor = preprocessing.StandardScaler()
        preprocessor.fit(X)
        Y_transformed = preprocessor.transform(Y)

        np.testing.assert_array_almost_equal(Y_transformed, np.array([[0, 0], [-1, -1]]), decimal=6)

    # ---------------------------------------------------------------------
    # Edge cases
    # ---------------------------------------------------------------------

    def test_constant_column(self):
        X = np.array([[1, 10], [1, 30]])
        Y = np.array([[1, 20], [1, 10]])

        preprocessor = preprocessing.StandardScaler()
        preprocessor.fit(X)
        Y_transformed = preprocessor.transform(Y)

        assert np.all(Y_transformed[:, 0] == 0)  # All values in the constant column should be 0

    def test_small_values_constant_column(self):
        X = np.array([[0.1], [0.1], [0.1]])
        Y = np.array([[0.1], [0.2]])

        preprocessor = preprocessing.StandardScaler()
        preprocessor.fit(X)
        Y_transformed = preprocessor.transform(Y)

        # The first column should be close to [0, 0.1]
        np.testing.assert_allclose(Y_transformed[:, 0], np.array([0, 0.1]), atol=1e-8)

    def test_nanometre_scale_column_is_standardized(self):
        X = np.array([[1e-9], [2e-9], [3e-9]])

        preprocessor = preprocessing.StandardScaler()
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)

        expected = np.array([-np.sqrt(1.5), 0.0, np.sqrt(1.5)])  # ≈ [-1.2247, 0, 1.2247]
        np.testing.assert_allclose(X_transformed[:, 0], expected, rtol=1e-7, atol=1e-12)

    def test_nan_in_fit(self):
        X = np.array([[1, 10], [3, np.nan]])
        Y = np.array([[1, 20], [1, 10]])

        preprocessor = preprocessing.StandardScaler()
        preprocessor.fit(X)
        Y_transformed = preprocessor.transform(Y)

        assert not np.isnan(Y_transformed).any()  # Ensure no NaN values in the output

    def test_nan_in_transform(self):
        X = np.array([[1, 10], [3, 30]])
        Y = np.array([[1, np.nan], [1, 10]])

        preprocessor = preprocessing.StandardScaler()
        preprocessor.fit(X)
        Y_transformed = preprocessor.transform(Y)

        # Input NaN in transform should be output as NaN in the corresponding position
        assert np.isnan(Y_transformed[0, 1])  # The first row, second column should be NaN
        assert not np.isnan(Y_transformed[1, 1])  # The second row, second column should not be NaN

    def test_nan_column(self):
        X = np.array([[np.nan, 10], [np.nan, 30]])

        preprocessor = preprocessing.StandardScaler()

        with pytest.raises(ValueError, match="Cannot fit StandardScaler on a column with all "
                                             "NaN values."):
            preprocessor.fit(X)

    def test_empty_transform(self):
        X = np.array([[1, 10], [3, 30]])
        Y = np.array([]).reshape(0, 2)

        preprocessor = preprocessing.StandardScaler()
        preprocessor.fit(X)

        # Should return an empty array with the correct shape
        assert preprocessor.transform(Y).shape == (0, 2)


class TestMinMaxScaler:

    def test_basic(self):
        X = np.array([[1, 10], [3, 30]])
        Y = np.array([[2, 20], [1, 10]])

        preprocessor = preprocessing.MinMaxScaler()
        preprocessor.fit(X)
        Y_transformed = preprocessor.transform(Y)

        np.testing.assert_array_almost_equal(Y_transformed, np.array([[0.5, 0.5], [0.0, 0.0]]),
                                             decimal=6)

    def test_basic_negative_feature_range(self):
        X = np.array([[1, 10], [3, 30]])
        Y = np.array([[2, 20], [1, 10]])

        preprocessor = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        preprocessor.fit(X)
        Y_transformed = preprocessor.transform(Y)

        np.testing.assert_array_almost_equal(Y_transformed, np.array([[0.0, 0.0], [-1.0, -1.0]]),
                                             decimal=6)

    def test_basic_outside_range(self):
        X = np.array([[1, 10], [3, 30]])
        Y = np.array([[5, 50]])

        preprocessor = preprocessing.MinMaxScaler()
        preprocessor.fit(X)
        Y_transformed = preprocessor.transform(Y)

        np.testing.assert_array_almost_equal(Y_transformed, np.array([[2, 2]]), decimal=6)

    def test_single_column_with_nan(self):
        X = np.array([[10], [30], [np.nan]])
        Y = np.array([[20], [40]])

        preprocessor = preprocessing.MinMaxScaler()
        preprocessor.fit(X)
        Y_transformed = preprocessor.transform(Y)

        np.testing.assert_array_almost_equal(Y_transformed, np.array([[0.5], [1.5]]), decimal=6)

    def test_constant_column(self):
        X = np.array([[1, 10], [1, 30]])
        Y = np.array([[1, 20], [1, 10]])

        preprocessor = preprocessing.MinMaxScaler()
        preprocessor.fit(X)
        Y_transformed = preprocessor.transform(Y)

        np.testing.assert_array_almost_equal(Y_transformed, np.array([[0, 0.5], [0, 0]]), decimal=6)

    def test_constant_column_negative_feature_range(self):
        X = np.array([[1, 10], [1, 30]])
        Y = np.array([[1, 20], [1, 10]])

        preprocessor = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        preprocessor.fit(X)
        Y_transformed = preprocessor.transform(Y)

        np.testing.assert_array_almost_equal(Y_transformed, np.array([[-1, 0], [-1, -1]]),
                                             decimal=6)

    def test_invalid_feature_range(self):

        with pytest.raises(ValueError, match="feature range must follow low < hi"):
            preprocessing.MinMaxScaler(feature_range=(2, 1))

    def test_empty_transform(self):
        X = np.array([[1, 10], [3, 30]])
        Y = np.array([]).reshape(0, 2)

        preprocessor = preprocessing.MinMaxScaler()
        preprocessor.fit(X)

        # Should return an empty array with the correct shape
        assert preprocessor.transform(Y).shape == (0, 2)

    def test_nan_column(self):
        X = np.array([[np.nan, 10], [np.nan, 30]])

        preprocessor = preprocessing.MinMaxScaler()

        with pytest.raises(ValueError, match="Cannot fit MinMaxScaler on a column with all "
                                             "NaN values."):
            preprocessor.fit(X)


class TestOneHotEncoder:

    def test_basic(self):
        X = np.array([[0, 10], [1, 20], [2, 10]])
        Y = np.array([[1, 20], [0, 10]])

        preprocessor = preprocessing.OneHotEncoder()
        preprocessor.fit(X)
        Y_transformed = preprocessor.transform(Y)

        expected = np.array([[0, 1, 0, 0, 1], [1, 0, 0, 1, 0]])  # One-hot encoded representation
        np.testing.assert_array_equal(Y_transformed, expected)

    # Made the decision to treat NaN as a separate category for one-hot encoding.
    def test_NaN_as_category(self):
        X = np.array([[0], [1], [np.nan]])
        Y = np.array([[np.nan], [1]])

        preprocessor = preprocessing.OneHotEncoder()
        preprocessor.fit(X)
        Y_transformed = preprocessor.transform(Y)

        expected = np.array([[0, 0, 1], [0, 1, 0]])
        np.testing.assert_array_equal(Y_transformed, expected)

    # Made the decision to raise an error if an unseen category is encountered during transform
    # rather than output all zeros.
    def test_unseen_category(self):
        X = np.array([[0], [1]])
        Y = np.array([[2]])  # Category '2' is unseen

        preprocessor = preprocessing.OneHotEncoder()
        preprocessor.fit(X)
        with pytest.raises(ValueError, match="Found unknown categories during transform"):
            preprocessor.transform(Y)

    def test_unseen_category_is_nan(self):
        X = np.array([[0], [1]])
        Y = np.array([[np.nan]])  # NaN is unseen

        preprocessor = preprocessing.OneHotEncoder()
        preprocessor.fit(X)
        with pytest.raises(ValueError, match="Found unknown categories during transform"):
            preprocessor.transform(Y)

    def test_mixed_row_with_unseen_category(self):
        X = np.array([[0], [1]])
        Y = np.array([[2], [0]])  # Category '2' is unseen

        preprocessor = preprocessing.OneHotEncoder()
        preprocessor.fit(X)
        with pytest.raises(ValueError, match="Found unknown categories during transform"):
            preprocessor.transform(Y)

    def test_string_category(self):
        X = np.array([["cat"], ["dog"], ["cat"]])
        Y = np.array([["dog"], ["cat"]])

        preprocessor = preprocessing.OneHotEncoder()
        preprocessor.fit(X)
        Y_transformed = preprocessor.transform(Y)

        expected = np.array([[0, 1], [1, 0]])  # One-hot encoded representation
        np.testing.assert_array_equal(Y_transformed, expected)

    def test_string_category_with_nan(self):
        X = np.array([["cat"], ["dog"], [np.nan]], dtype=object)
        Y = np.array([[np.nan], ["dog"]], dtype=object)

        preprocessor = preprocessing.OneHotEncoder()
        preprocessor.fit(X)
        Y_transformed = preprocessor.transform(Y)

        expected = np.array([[0, 0, 1], [0, 1, 0]])  # One-hot encoded representation
        np.testing.assert_array_equal(Y_transformed, expected)

    def test_string_category_with_unseen(self):
        X = np.array([["cat"], ["dog"]])
        Y = np.array([["mouse"]])  # Unseen category

        preprocessor = preprocessing.OneHotEncoder()
        preprocessor.fit(X)
        with pytest.raises(ValueError, match="Found unknown categories during transform"):
            preprocessor.transform(Y)

    def test_string_category_with_unseen_and_nan(self):
        X = np.array([["cat"], ["dog"]])
        Y = np.array([["cat"], [np.nan]], dtype=object)  # Unseen category and NaN

        preprocessor = preprocessing.OneHotEncoder()
        preprocessor.fit(X)
        with pytest.raises(ValueError, match="Found unknown categories during transform"):
            preprocessor.transform(Y)

    def test_string_category_with_number(self):
        X = np.array([["cat"], ["dog"]])
        Y = np.array([[1]], dtype=object)  # Unseen category (number)

        preprocessor = preprocessing.OneHotEncoder()
        preprocessor.fit(X)
        with pytest.raises(ValueError, match="Found unknown categories during transform"):
            preprocessor.transform(Y)

    def test_mixed_string_and_number_category(self):
        X = np.array([["cat"], ["dog"]])
        Y = np.array([["cat"], [1]], dtype=object)  # Mixed: seen string and unseen number

        preprocessor = preprocessing.OneHotEncoder()
        preprocessor.fit(X)
        with pytest.raises(ValueError, match="Found unknown categories during transform"):
            preprocessor.transform(Y)

    def test_load_csv_output_shape(self):
        # Simulate loading a CSV with mixed types
        X = np.array([["cat", 1.0], ["dog", 2.0], ["cat", 3.0]], dtype=object)
        Y = np.array([["dog", 1.0], ["cat", 3.0]], dtype=object)

        preprocessor = preprocessing.OneHotEncoder()
        preprocessor.fit(X)
        Y_transformed = preprocessor.transform(Y)

        expected = np.array([[0, 1, 1, 0, 0], [1, 0, 0, 0, 1]])
        np.testing.assert_array_equal(Y_transformed, expected)

    def test_mixed_column_raises_type_error(self):
        X = np.array([[1, "cat"], ["two", "dog"]], dtype=object)

        preprocessor = preprocessing.OneHotEncoder()

        with pytest.raises(TypeError, match="Mixed types in columns"):
            preprocessor.fit(X)


class TestSimpleImputer:

    def test_basic(self):
        X = np.array([[1, np.nan], [3, 30]])
        Y = np.array([[np.nan, 20], [1, np.nan]])

        preprocessor = preprocessing.SimpleImputer(fill_value=0)
        preprocessor.fit(X)
        Y_transformed = preprocessor.transform(Y)

        expected = np.array([[0, 20], [1, 0]])
        np.testing.assert_array_equal(Y_transformed, expected)

    def test_negative_fill_value(self):
        X = np.array([[1, np.nan], [3, 30]])
        Y = np.array([[np.nan, 20], [1, np.nan]])

        preprocessor = preprocessing.SimpleImputer(fill_value=-1)
        preprocessor.fit(X)
        Y_transformed = preprocessor.transform(Y)

        expected = np.array([[-1, 20], [1, -1]])
        np.testing.assert_array_equal(Y_transformed, expected)

    def test_all_nan_column(self):
        X = np.array([[np.nan, 10], [np.nan, 30]])
        Y = np.array([[np.nan, 20], [1, np.nan]])

        preprocessor = preprocessing.SimpleImputer(fill_value=0)
        preprocessor.fit(X)
        Y_transformed = preprocessor.transform(Y)

        # This shouldnt raise an error, and the NaN in the first column should be replaced with 0
        expected = np.array([[0, 20], [1, 0]])
        np.testing.assert_array_equal(Y_transformed, expected)

    def test_string_fill_value_type(self):
        with pytest.raises(ValueError, match="fill_value must be a numeric type"):
            preprocessing.SimpleImputer(fill_value="invalid")

    def test_nan_fill_value_type(self):
        with pytest.raises(ValueError, match="fill_value must be a numeric type"):
            preprocessing.SimpleImputer(fill_value=np.nan)

    def test_infinity_fill_value_type(self):
        with pytest.raises(ValueError, match="fill_value must be a numeric type"):
            preprocessing.SimpleImputer(fill_value=np.inf)

    def test_large_numeric_fill_value(self):
        X = np.array([[1, np.nan], [3, 30]])
        Y = np.array([[np.nan, 20], [1, np.nan]])

        large_value = np.int64(1e10)
        preprocessor = preprocessing.SimpleImputer(fill_value=large_value)
        preprocessor.fit(X)
        Y_transformed = preprocessor.transform(Y)

        expected = np.array([[large_value, 20], [1, large_value]])
        np.testing.assert_array_equal(Y_transformed, expected)
