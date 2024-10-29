import re

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal
from plotly import graph_objects as go

from src.data_exploration.dimensionality_reduction import *


########################################################################################################################
# MARK: Fixtures
########################################################################################################################
@pytest.fixture
def example_components() -> np.array:
    return np.array([[1.1, 2.1, 3.1], [1.2, 2.2, 3.2], [1.3, 2.3, 3.3]])


@pytest.fixture
def example_labels() -> list:
    return ["Category_1", "Category_2"]


@pytest.fixture
def example_input_df() -> pd.DataFrame:
    """
    example_input_df is meant to mimic an input dataframe with both categorical and feature columns.
    the categorical columns are meant as labels(s) for the datapoints,
    the features will be input for the dimensionality reduction
    """
    data = {
        "Category_1": ["cat_1.1", "cat_1.2", "cat_1.3"],
        "Category_2": ["cat_2.1", "cat_2.2", "cat_2.3"],
        "Feature_1": [1.1, 1.2, 1.3],
        "Feature_2": [2.1, 2.2, 2.3],
        "Feature_3": [3.1, 3.2, 3.3],
    }
    df = pd.DataFrame(data=data)
    return df


@pytest.fixture
def example_umap_input_df() -> pd.DataFrame:
    """
    example_input_df is meant to mimic an input dataframe with both categorical and feature columns.
    the categorical columns are meant as labels(s) for the datapoints,
    the features will be input for the dimensionality reduction
    """
    data = {
        "Category_1": ["cat_1.1", "cat_1.2", "cat_1.3", "cat_1.4"],
        "Category_2": ["cat_2.1", "cat_2.2", "cat_2.3", "cat_2.4"],
        "Feature_1": [1.1, 1.2, 1.3, 1.4],
        "Feature_2": [2.1, 2.2, 2.3, 2.4],
        "Feature_3": [3.1, 3.2, 3.3, 3.4],
    }
    df = pd.DataFrame(data=data)
    return df


@pytest.fixture
def expected_component_df() -> pd.DataFrame:
    """
    expected_component_df is meant to mimic a dimensional reduced dataframe with both labels as component columns.
    """
    data = {
        "Category_1": ["cat_1.1", "cat_1.2", "cat_1.3"],
        "Category_2": ["cat_2.1", "cat_2.2", "cat_2.3"],
        "Component_1": [1.1, 1.2, 1.3],
        "Component_2": [2.1, 2.2, 2.3],
        "Component_3": [3.1, 3.2, 3.3],
    }
    df = pd.DataFrame(data=data)
    return df


@pytest.fixture
def expected_umap_component_df() -> pd.DataFrame:
    """
    expected_component_df is meant to mimic a dimensional reduced dataframe with both labels as component columns.
    """
    data = {
        "Category_1": ["cat_1.1", "cat_1.2", "cat_1.3", "cat_1.4"],
        "Category_2": ["cat_2.1", "cat_2.2", "cat_2.3", "cat_2.4"],
        "Component_1": [1.1, 1.2, 1.3, 1.4],
        "Component_2": [2.1, 2.2, 2.3, 2.4],
        "Component_3": [3.1, 3.2, 3.3, 3.4],
    }
    df = pd.DataFrame(data=data)
    return df


@pytest.fixture
def expected_tsne_df() -> pd.DataFrame:
    """
    expected_tsne_df is the output expected from running expected_component_df into get_tsne_df()
    """
    data = {
        "Category_1": ["cat_1.1", "cat_1.2", "cat_1.3"],
        "Category_2": ["cat_2.1", "cat_2.2", "cat_2.3"],
        "Component_1": [
            -13.507349592754748,
            -7.622942589888864e-08,
            13.507349668984174,
        ],
        "Component_2": [
            0.7270614694603017,
            1.6944748303560152e-06,
            -0.7270631639351319,
        ],
    }
    df = pd.DataFrame(data=data)
    return df


@pytest.fixture
def expected_pca_df() -> pd.DataFrame:
    """
    expected_pca_df is the output expected from running expected_component_df into get_pca_df()
    """
    data = {
        "Category_1": ["cat_1.1", "cat_1.2", "cat_1.3"],
        "Category_2": ["cat_2.1", "cat_2.2", "cat_2.3"],
        "Component_1": [
            0.17320508075688804,
            2.563950248511418e-16,
            -0.17320508075688715,
        ],
        "Component_2": [
            5.623678430006907e-16,
            3.374425194124418e-16,
            5.623678430006941e-16,
        ],
    }
    df = pd.DataFrame(data=data)
    return df


@pytest.fixture
def expected_umap_df() -> pd.DataFrame:
    """
    expected_umap_df is the output expected from running expected_component_df into get_umap_df()
    """
    data = {
        "Category_1": ["cat_1.1", "cat_1.2", "cat_1.3", "cat_1.4"],
        "Category_2": ["cat_2.1", "cat_2.2", "cat_2.3", "cat_2.4"],
        "Component_1": [
            24.47628,
            24.158213,
            23.217754,
            22.571693,
            #     12.117031,
            #     11.648557,
            #     10.659976,
            #     10.146545,
        ],
        "Component_2": [
            10.975056648254395,
            10.268843650817871,
            9.946699142456055,
            10.470194816589355,
            #     14.640894,
            #     14.024240,
            #     13.920915,
            #     14.574826,
        ],
    }
    df = pd.DataFrame(data=data)
    return df


########################################################################################################################
# MARK: Unascorted
########################################################################################################################


# bundle_components_and_labels_into_df
######################################
def test_bundle_components_and_labels_into_df_returns_correct_df_format(
    example_components,
    example_labels,
    example_input_df,
    expected_component_df,
):
    """Test if bundle_components_and_labels_into_df returns a pd DataFrame"""
    result = bundle_components_and_labels_into_df(
        labels=example_input_df[example_labels], components=example_components
    )
    assert_frame_equal(result, expected_component_df)


def test_bundle_components_and_labels_into_df_outputs_correct_shape(
    example_components, example_labels, example_input_df, expected_component_df
):
    """Test if bundle_components_and_labels_into_df returns the right number of rows and columns"""
    df = bundle_components_and_labels_into_df(
        labels=example_input_df[example_labels], components=example_components
    )
    assert df.shape == expected_component_df.shape


######################################
# MARK: tSNE
######################################


@pytest.mark.parametrize("n_components, expected_shape", [(2, (3, 4)), (3, (3, 5))])
def test_get_tsne_df_returns_correct_number_of_components(
    expected_component_df, example_labels, n_components, expected_shape
):
    """Test if the output of get_tsne_df() contains the right number of components"""
    df_result, embedding = get_tsne_df(
        df=expected_component_df,
        labels=example_labels,
        n_components=n_components,
        random_state=1,
    )
    assert df_result.shape == expected_shape


@pytest.mark.parametrize(
    "n_components, error_msg",
    [
        (-1, "n_components must >= 1"),
        (0, "n_components must >= 1"),
        (2.5, "n_components must be round number"),
    ],
)
def test_get_tsne_df_returns_errors_with_invalid_component_nr(
    n_components, error_msg, expected_component_df, example_labels
):
    """Test if invalid nr of components in get_tsne_df() returns the right error message"""
    with pytest.raises(ValueError, match=error_msg):
        df_result, embedding = get_tsne_df(
            df=expected_component_df,
            labels=example_labels,
            n_components=n_components,
            random_state=1,
        )


def test_get_tsne_df_returns_correct_output(
    example_input_df, example_labels, expected_tsne_df
):
    """Test if tSNE is properly calculated."""
    df_result, embedding = get_tsne_df(
        df=example_input_df,
        labels=example_labels,
        n_components=2,
        random_state=1,
    )
    assert_frame_equal(df_result, expected_tsne_df, check_exact=False)


def test_tsne_transform_returns_correct_output(example_input_df):
    """tsne_transform should transform a new data point into the tsne space"""
    df_point = pd.DataFrame(
        data={
            "Category_1": ["cat_1"],
            "Category_2": ["cat_2"],
            "Feature_1": [1.5],
            "Feature_2": [1.5],
            "Feature_3": [1.5],
        }
    )

    df_expected = pd.DataFrame(
        data={
            "Category_1": ["cat_1"],
            "Category_2": ["cat_2"],
            "Component_1": [
                -13.505010939385105,
            ],
            "Component_2": [
                0.7269364320272106,
            ],
        }
    )

    df_tsne, embedding = get_tsne_df(
        df=example_input_df,
        labels=["Category_1", "Category_2"],
        n_components=2,
        random_state=1,
    )
    df_point_transformed = tsne_transform(
        df=df_point, labels=["Category_1", "Category_2"], embedding=embedding
    )

    assert_frame_equal(df_point_transformed, df_expected, check_exact=False)


######################################
# MARK: PCA
######################################


@pytest.mark.parametrize("n_components, expected_shape", [(2, (3, 4)), (3, (3, 5))])
def test_get_pca_df_returns_correct_number_of_components(
    example_input_df, example_labels, n_components, expected_shape
):
    """Test if the output of get_pca_df() contains the right number of components"""
    df_result, embedding = get_pca_df(
        df=example_input_df,
        labels=example_labels,
        n_components=n_components,
    )
    assert df_result.shape == expected_shape


@pytest.mark.parametrize(
    "n_components, error_msg",
    [
        (-1, "n_components must >0"),
        (0, "n_components must >0"),
        (2.5, "If n_components > 1, it must be round number"),
    ],
)
def test_get_pca_df_returns_errors_with_invalid_component_nr(
    n_components, error_msg, expected_component_df, example_labels
):
    """Test if invalid nr of components in get_pca_df() returns the right error message"""
    with pytest.raises(ValueError, match=error_msg):
        df_result, embedding = get_pca_df(
            df=expected_component_df,
            labels=example_labels,
            n_components=n_components,
        )


def test_get_pca_accepts_n_components_between_zero_and_one(
    expected_component_df, example_labels
):
    """Test if n_components argument in get_pca_df() is a fraction between 0-1 does NOT raise an error"""
    try:
        get_pca_df(
            df=expected_component_df,
            labels=example_labels,
            n_components=0.7,
        )
    except Exception as exc:
        assert False, f"'get_pca_df()' with n_component=0.95 raised the exception {exc}"


def test_get_pca_df_returns_correct_output(
    example_input_df, example_labels, expected_pca_df
):
    """Test if PCA is properly calculated."""

    df_result, embedding = get_pca_df(
        df=example_input_df,
        labels=example_labels,
        n_components=2,
    )
    assert_frame_equal(df_result, expected_pca_df, check_exact=False)


def test_pca_transform_returns_correct_output(example_input_df):
    """pca_transform should transform a new data point into the PCA space"""
    df_point = pd.DataFrame(
        data={
            "Category_1": ["cat_1"],
            "Category_2": ["cat_2"],
            "Feature_1": [1.5],
            "Feature_2": [2.5],
            "Feature_3": [3.5],
        }
    )

    df_expected = pd.DataFrame(
        data={
            "Category_1": ["cat_1"],
            "Category_2": ["cat_2"],
            "Component_1": [
                -0.5196152422706628,
            ],
            "Component_2": [
                12.7755575615628914e-16,
            ],
        }
    )

    df_pca, embedding = get_pca_df(
        df=example_input_df,
        labels=["Category_1", "Category_2"],
        n_components=2,
    )
    df_point_transformed = pca_transform(
        df=df_point, labels=["Category_1", "Category_2"], embedding=embedding
    )

    assert_frame_equal(df_point_transformed, df_expected, check_exact=False)


######################################
# MARK: UMAP
######################################


@pytest.mark.parametrize("n_components, expected_shape", [(1, (4, 3)), (2, (4, 4))])
def test_get_umap_df_returns_correct_number_of_components(
    example_umap_input_df, example_labels, n_components, expected_shape
):
    """Test if the output of get_umap_df() contains the right number of components"""
    df_result, embedding = get_umap_df(
        df=example_umap_input_df,
        labels=example_labels,
        n_components=n_components,
    )
    assert df_result.shape == expected_shape


@pytest.mark.parametrize(
    "n_components, error_msg",
    [
        (-1, "n_components must >= 1"),
        (0, "n_components must >= 1"),
        (
            3,
            "n_components can maximimally be n_samples-2. n_samples==4, so n_components needs to be <= 2",
        ),
        (
            10,
            "n_components can maximimally be n_samples-2. n_samples==4, so n_components needs to be <= 2",
        ),
        (1.5, "n_components must be round number"),
    ],
)
def test_get_umap_df_returns_errors_with_invalid_component_nr(
    n_components, error_msg, expected_umap_component_df, example_labels
):
    """Test if invalid nr of components in get_umap_df() returns the right error message"""
    with pytest.raises(ValueError, match=error_msg):
        df_result, embedding = get_umap_df(
            df=expected_umap_component_df,
            labels=example_labels,
            n_components=n_components,
        )


# Disabled this test. Due to the non-deterministic nature of UMAP, these test fail when they are run in batches. No solution found yet to ensure the kernel is in exactly the same state.
# def test_get_umap_df_returns_correct_output(
#     example_umap_input_df, example_labels, expected_umap_df
# ):
#     """Test if UMAP is properly calculated."""

#     df_result_umap, embedding = get_umap_df(
#         df=example_umap_input_df,
#         labels=example_labels,
#         n_components=2,
#     )
#     assert_frame_equal(
#         df_result_umap, expected_umap_df, check_exact=False, check_dtype=False
#     )

# Disabled this test. Due to the non-deterministic nature of UMAP, these test fail when they are run in batches. No solution found yet to ensure the kernel is in exactly the same state.
# def test_umap_transform_returns_correct_output(example_input_df):
#     """pca_transform should transform a new data point into the UMAP space"""
#     df_point = pd.DataFrame(
#         data={
#             "Category_1": ["cat_1"],
#             "Category_2": ["cat_2"],
#             "Feature_1": [1.5],
#             "Feature_2": [2.5],
#             "Feature_3": [3.5],
#         }
#     )

#     df_expected = pd.DataFrame(
#         data={
#             "Category_1": ["cat_1"],
#             "Category_2": ["cat_2"],
#             "Component_1": [
#                 # -21.608802795410156,
#                 -19.530899047851562,
#             ],
#         }
#     )

#     df_umap, embedding = get_umap_df(
#         df=example_input_df,
#         labels=["Category_1", "Category_2"],
#         n_components=1,
#     )
#     df_point_transformed = umap_transform(
#         df=df_point, labels=["Category_1", "Category_2"], embedding=embedding
#     )

#     assert_frame_equal(
#         df_point_transformed, df_expected, check_exact=False, check_dtype=False
#     )


######################################
# MARK: Plotting
######################################


def test_confidence_ellipse_returns_correct_path():
    """Test if confidence ellipse returns the correct path"""
    x = np.array([1, 2, 3, 4])
    y = x / 2

    expected = "M 5.03034911952218, 2.51517455976109L1.234825440238911, 0.6174127201194555L1.2348254402389092, 0.6174127201194546L5.03034911952218, 2.51517455976109 Z"
    result = confidence_ellipse(x=x, y=y, n_std=1.96, size=4)

    # You can't compare strings directly as the precision set on the device can differ.
    # Instead, extract all numbers, put them in an array and test assert_array_almost_equal()
    expected_array = np.array(
        [float(nr) for nr in re.sub(r"[A-Z,]", " ", expected).strip().split()]
    )
    result_array = np.array(
        [float(nr) for nr in re.sub(r"[A-Z,]", " ", result).strip().split()]
    )

    assert_array_almost_equal(result_array, expected_array, decimal=10)


@pytest.mark.parametrize(
    "size, expected",
    [
        (100, 99),
        (1234, 1233),
        (3, 2),
    ],
)
def test_confidence_ellipse_returns_ellipse_with_correct_size(size, expected):
    """Test if setting the size returns the correct number of data points in the ellipse path"""
    # The number of data points in the ellipse string path cam be found by counting the number of 'L' characters.
    # size = count('L') - 1
    x = np.array([1, 2, 3, 4])
    y = x / 2
    ellipse_path = confidence_ellipse(x=x, y=y, n_std=1.96, size=size)
    result = ellipse_path.count("L")
    assert result == expected


def test_plot_dim_reduction_returns_figure(expected_tsne_df):
    """Test if plot_dim_reduction returns a go figure object"""
    fig = plot_dim_reduction(
        df=expected_tsne_df,
        labels=["Category_1", "Category_2"],
        hue_col="Category_1",
        plot_ellips=True,
        title="Test_Plot",
    )
    assert type(fig) is go.Figure
