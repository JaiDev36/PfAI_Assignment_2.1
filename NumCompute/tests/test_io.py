# pyright: reportArgumentType=false
import numpy as np
import pytest

from numcompute import io


# ---------------------------------------------------------------------
# Basic / expected behaviour
# ---------------------------------------------------------------------

def test_blanks_become_fill_values(tmp_path):
    """ missing='fill' must replace blanks with the fill_value."""
    file = tmp_path / "test.csv"
    file.write_text("1,2\n"
                    "3, \n"
                    ",4\n"
                    "5,6")
    result = io.load_csv(file, delimiter=',', missing='fill', fill_value=0)

    np.testing.assert_array_equal(result, np.array([[1, 2], [3, 0], [0, 4], [5, 6]]))


def test_tab_delimiter(tmp_path):
    """Test that tab delimiter works correctly."""
    file = tmp_path / "test.csv"
    file.write_text("1\t2\n"
                    "3\t4\n"
                    "5\t6")
    result = io.load_csv(file, delimiter='\t', missing='fill', fill_value=0)

    np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4], [5, 6]]))


def test_single_row(tmp_path):
    """Test that a single row is loaded correctly."""
    file = tmp_path / "test.csv"
    file.write_text("1,2,3")
    result = io.load_csv(file, delimiter=',', missing='fill', fill_value=0)

    assert result.shape == (1, 3)


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------

def test_skip_ignores_fill_value(tmp_path):
    """missing='skip' must drop incomplete rows even when fill_value is not NaN."""

    file = tmp_path / "test.csv"
    file.write_text("1,2\n"
                    "3, \n"
                    ",4\n"
                    "5,6")
    result = io.load_csv(file, delimiter=',', missing='skip', fill_value=-1)

    np.testing.assert_array_equal(result, np.array([[1, 2], [5, 6]]))
    assert result.shape == (2, 2)  # Expecting 2 rows and 2 columns


def test_all_rows_skipped(tmp_path):
    """Test that all rows are skipped when missing='skip' and all rows have missing values."""
    file = tmp_path / "test.csv"
    file.write_text("1, \n"
                    " ,2\n"
                    " , ")
    result = io.load_csv(file, delimiter=',', missing='skip', fill_value=0)

    assert result.shape == (0, 2)  # Expecting an empty array with shape (0, 2)


def test_empty_file(tmp_path):
    file = tmp_path / "test.csv"
    file.write_text("")

    result = io.load_csv(file, delimiter=',', missing='fill', fill_value=0)
    assert result.shape == (0, 0)  # Expecting an empty array with shape (0, 0)


# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------

def test_invalid_missing_input_raises(tmp_path):
    """Test that an invalid 'missing' parameter raises a ValueError."""

    file = tmp_path / "test.csv"
    file.write_text("1,2\n3,4")

    with pytest.raises(ValueError, match="missing parameter must be 'fill' or 'skip'"):
        io.load_csv(file, missing="invalid")


# ---------------------------------------------------------------------
# String / mixed-type input (dtype=object)
# ---------------------------------------------------------------------

def test_literal_nan_is_treated_as_missing_not_text(tmp_path):
    """Test that the literal string 'NaN' is treated as a missing value, not text."""
    file = tmp_path / "test.csv"
    file.write_text("1,NaN\n"
                    "3,4\n"
                    "5,6")
    result = io.load_csv(file, delimiter=',', missing='fill', fill_value=0, dtype=float)

    np.testing.assert_array_equal(result, np.array([[1, 0], [3, 4], [5, 6]]))


def test_nan_case_and_whitespace_are_missing(tmp_path):
    """Test that 'nan', 'NaN', 'NAN' and whitespace are treated as missing values."""
    file = tmp_path / "test.csv"
    file.write_text("1,nan\n"
                    "3,NaN\n"
                    "5,NAN\n"
                    "7, \n"
                    "9,10")
    result = io.load_csv(file, delimiter=',', missing='fill', fill_value=0, dtype=float)

    np.testing.assert_array_equal(result, np.array([[1, 0], [3, 0], [5, 0], [7, 0], [9, 10]]))


def test_mixed_type_columns_raises_error_under_object_dtype(tmp_path):
    """Test that mixed-type columns raise a TypeError when dtype=object."""
    file = tmp_path / "test.csv"
    file.write_text("1,hello\n"
                    "3,4\n"
                    "5,world")
    with pytest.raises(TypeError):
        io.load_csv(file, delimiter=',', missing='fill', fill_value=np.nan, dtype=object)


def test_purely_text_columns_are_loaded_correctly(tmp_path):
    """Test that purely text columns are loaded correctly when dtype=object."""
    file = tmp_path / "test.csv"
    file.write_text("hello,world\n"
                    "foo,bar\n"
                    "baz,qux")
    result = io.load_csv(file, delimiter=',', missing='fill', fill_value=np.nan, dtype=object)

    expected = np.array([["hello", "world"],
                         ["foo", "bar"],
                         ["baz", "qux"]], dtype=object)
    np.testing.assert_array_equal(result, expected)


def test_fill_value_type_mismatch_raises(tmp_path):
    """Test that a TypeError is raised when fill_value type does not match dtype."""
    file = tmp_path / "test.csv"
    file.write_text("1,2\n"
                    "3, \n"
                    " ,4\n"
                    "5,6")
    with pytest.raises(TypeError, match="Cannot fill missing values with text when dtype is float"):
        io.load_csv(file, delimiter=',', missing='fill', fill_value="text", dtype=float)


def test_dtype_must_be_float_or_object(tmp_path):
    """Test that a TypeError is raised when dtype is not float or object."""
    file = tmp_path / "test.csv"
    file.write_text("1,2\n3,4")
    with pytest.raises(TypeError, match="dtype must be float or object"):
        io.load_csv(file, delimiter=',', missing='fill', fill_value=np.nan, dtype=int)


def test_numeric_columns_with_text_raises_type_error(tmp_path):
    """Test that numeric columns with text raise a TypeError when dtype=float."""
    file = tmp_path / "test.csv"
    file.write_text("1,hello\n"
                    "5,world")
    with pytest.raises(TypeError, match="Text in columns: column 1"):
        io.load_csv(file, delimiter=',', missing='fill', fill_value=np.nan, dtype=float)


def test_fill_value_clash_with_text_column_raises(tmp_path):
    """Test that a TypeError is raised when trying to fill missing values in a column
    that has text."""
    file = tmp_path / "test.csv"
    file.write_text("1,hello\n"
                    "3, \n"
                    "5,world")
    with pytest.raises(TypeError, match="Cannot fill missing values in columns: column 1"):
        io.load_csv(file, delimiter=',', missing='fill', fill_value=0, dtype=object)


def test_fill_value_clash_with_numeric_column_raises(tmp_path):
    """Test that a TypeError is raised when trying to fill missing values in a column
    that has numbers."""
    file = tmp_path / "test.csv"
    file.write_text("1,2\n"
                    "3, \n"
                    "5,6")
    with pytest.raises(TypeError, match="Cannot fill missing values in columns: column 1"):
        io.load_csv(file, delimiter=',', missing='fill', fill_value="text", dtype=object)


def test_fill_value_not_number_or_string_raises(tmp_path):
    """Test that a TypeError is raised when fill_value is neither a number nor a string."""
    file = tmp_path / "test.csv"
    file.write_text("1,2\n3,4")
    with pytest.raises(TypeError, match="fill_value must be a number or a string, got"):
        io.load_csv(file, delimiter=',', missing='fill', fill_value=[1, 2], dtype=float)


def test_mixed_type_columns_with_skip(tmp_path):
    """Test that mixed-type columns raise a TypeError when missing='skip'."""
    file = tmp_path / "test.csv"
    file.write_text("1,hello\n"
                    "3,4\n"
                    "5,world")
    with pytest.raises(TypeError, match="Mixed types in columns: column 1"):
        io.load_csv(file, delimiter=',', missing='skip', fill_value=np.nan, dtype=object)


def test_missing_skip_with_text_column(tmp_path):
    """Test that missing='skip' works correctly with a text column."""
    file = tmp_path / "test.csv"
    file.write_text("1,hello\n"
                    "3, \n"
                    "5,world")
    result = io.load_csv(file, delimiter=',', missing='skip', fill_value=np.nan, dtype=object)

    expected = np.array([[1, "hello"],
                         [5, "world"]], dtype=object)
    np.testing.assert_array_equal(result, expected)


def test_skip_ignores_fill_value_with_text_column(tmp_path):
    """Test that missing='skip' ignores fill_value even with a text column."""
    file = tmp_path / "test.csv"
    file.write_text("1,hello\n"
                    "3, \n"
                    "5,world")
    result = io.load_csv(file, delimiter=',', missing='skip', fill_value="fill", dtype=object)

    expected = np.array([[1, "hello"],
                         [5, "world"]], dtype=object)
    np.testing.assert_array_equal(result, expected)


def test_skip_ignores_fill_value_with_numeric_column(tmp_path):
    """Test that missing='skip' ignores fill_value even with a numeric column."""
    file = tmp_path / "test.csv"
    file.write_text("1,2\n"
                    "3, \n"
                    "5,6")
    result = io.load_csv(file, delimiter=',', missing='skip', fill_value=0, dtype=float)

    expected = np.array([[1, 2],
                         [5, 6]], dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_default_fill_value_is_nan(tmp_path):
    """Test that the default fill_value is np.nan when missing='fill'."""
    file = tmp_path / "test.csv"
    file.write_text("1,2\n"
                    "3, \n"
                    " ,4\n"
                    "5,6")
    result = io.load_csv(file, delimiter=',', missing='fill', dtype=object)

    expected = np.array([[1.0, 2.0],
                         [3.0, np.nan],
                         [np.nan, 4.0],
                         [5.0, 6.0]], dtype=float)
    assert result.dtype == object
    np.testing.assert_array_equal(result.astype(float), expected)  # cast result


def test_default_nan_fill_allowed_in_text_column(tmp_path):
    """Test that the default fill_value (np.nan) is allowed in a text column when dtype=object."""
    file = tmp_path / "test.csv"
    file.write_text("1,hello\n"
                    "3, \n"
                    "5,world")
    result = io.load_csv(file, delimiter=',', missing='fill', dtype=object)

    expected = np.array([[1.0, "hello"],
                         [3.0, np.nan],
                         [5.0, "world"]], dtype=object)
    assert result.dtype == object
    # assert_array_equal treats nan != nan in object arrays; assert_equal on lists handles it
    np.testing.assert_equal(result.tolist(), expected.tolist())
