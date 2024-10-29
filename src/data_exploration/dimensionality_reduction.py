import numpy as np
import pandas as pd
import plotly.express as px
from openTSNE import TSNE
from plotly import graph_objects as go
from sklearn.decomposition import PCA
from umap import UMAP

# from sklearn.preprocessing import StandardScaler, MinMaxScaler


def bundle_components_and_labels_into_df(
    labels: pd.DataFrame | pd.Series, components
) -> pd.DataFrame:
    """Bundles numerical (component/feature) df and df with labels into single df.

    Dimensional reduction tools take numerical df's as input and require you to keep the labels in a separate variable.
    With this function you can easily combine them again, which helps with visualising the output e.g. with seaborn.
    Output is structure to put the labels columns first and then the (numerical) component columns.
    Components are renamed to 'Component_#' with # corresponding to the number of that component.

    Args:
        labels (df): containing minimal 1 column with labels
        components (df): containing minimal 1 column of components

    Returns:
        df with labels and components

    """
    # First add labels
    if isinstance(labels, pd.Series):
        df = pd.DataFrame(labels)
    else:
        df = labels.copy()

    # Count components
    n_components = components.shape[1]

    # Then add feature date to df
    for iComponent in range(n_components):
        df[f"Component_{iComponent+1}"] = components[:, iComponent]

    return df


######################################
# tSNE
######################################
# MARK: tSNE


def get_tsne_df(
    df: pd.DataFrame,
    labels: str | list,
    n_components: int = 2,
    random_state: int = 42,
    perplexity: int = 30,
) -> tuple[pd.DataFrame, TSNE]:
    """Calculates tSNE and returns the data in a easy to plot dataframe

    Args:
        df (pd.DataFrame): input dataframe that contains at least labels column listed in 'labels' and one or more numerical data columns
        labels (str | list): labels(s) of the data points, these will be added to the tSNE space output.
        n_components (int, optional): Number of components the tSNE should calculate+return. Defaults to 2.
        random_state (int, optional): Random state of the tSNE calculation, fix this to generate consistent plots. Defaults to 42.
        perplexity (int|float, optional): The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity (typical: 5-50).

    Returns:
        pd.DataFrame: tSNE components + additional labels(s) in a df that is seaborn/plotly plot proof
        embedding: tsne embedding object <class 'openTSNE.tsne.TSNEEmbedding'>, can be used to transform new data points
    """
    if n_components > 1:  # must be round number
        if n_components % 1 != 0:
            raise ValueError("n_components must be round number")
    elif n_components < 1:  # must be greater than or equal to 1
        raise ValueError("n_components must >= 1")
    else:
        pass
    # Get numerical data from input df
    x = np.array(df.loc[:, ~df.columns.isin(labels)])

    # Calc t-SNE
    tsne = TSNE(
        perplexity=perplexity,
        n_components=n_components,
        metric="euclidean",
        n_jobs=-1,
        random_state=random_state,
    )
    embedding = tsne.fit(x)

    # Bundle everything in a df
    df_tsne = bundle_components_and_labels_into_df(
        labels=df[labels], components=embedding
    )

    return df_tsne, embedding


def tsne_transform(
    df: pd.DataFrame, labels: str | list, embedding: TSNE
) -> pd.DataFrame:
    """Transforms new data points in the same tSNE space as that of the datapoints the embedding object was fitted with. Returns an easy to plot dataframe

    Args:
        df (pd.DataFrame): input dataframe with the same format as that is used to train the 'embedding' variable. (i.e. contains at least labels column listed in 'labels' and one or more numerical data columns)
        labels (str | list): labels(s) of the data points, these will be added to the tSNE space output.
        embedding (TSNE): already fitted openTSNE.tsne.TSNEEmbedding you wish to use to transform the new datapoints in df

    Returns:
        pd.DataFrame: tSNE components + additional labels(s) in a df that is seaborn/plotly plot proof
        TSNE: tsne object fitted to the input data
    """
    # Get numerical data from input df
    x = df.loc[:, ~df.columns.isin(labels)]

    # transform data points
    embedded_x = embedding.transform(x)

    # Bundle everything in a df
    df_tsne = bundle_components_and_labels_into_df(
        labels=df[labels], components=embedded_x
    )

    return df_tsne


######################################
# PCA
######################################
# MARK: PCA


def get_pca_df(
    df: pd.DataFrame,
    labels: str | list,
    n_components: int | float = 2,
) -> tuple[pd.DataFrame, PCA]:
    """Calculates PCA and returns the data in a easy to plot dataframe

    Args:
        df (pd.DataFrame): input dataframe that contains at least labels column listed in 'labels' and one or more numerical data columns
        labels (str | list): labels(s) of the data points, these will be added to the pca space output.
        n_components (int, optional): Number of components the PCA should calculate+return. Defaults to 2.

    Returns:
        pd.DataFrame: PCA components + additional labels(s) in a df that is seaborn/plotly plot proof
        PCA: PCA scikitlearn object that can be used to transform new data points
    """
    if n_components > 1:  # if bigger than zero, n_components must be round number
        if n_components % 1 != 0:
            raise ValueError("If n_components > 1, it must be round number")
    elif n_components <= 0:  # must be positive
        raise ValueError("n_components must >0")
    else:  # note n_components is allowed to be between 0-1 => it is then the fraction of variance that needs to be explained
        pass

    # Get numerical data from input df
    x = df.loc[:, ~df.columns.isin(labels)]

    # Calc PCA
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(x)

    # Bundle everything in a df
    df_pca = bundle_components_and_labels_into_df(
        labels=df[labels], components=embedding
    )

    return df_pca, pca


def pca_transform(df: pd.DataFrame, labels: str | list, embedding: PCA) -> pd.DataFrame:
    """Transforms new data points in the same tSNE space as that of the datapoints the embedding object was fitted with. Returns an easy to plot dataframe.

    Args:
        df (pd.DataFrame): input dataframe with the same format as that is used to train the 'embedding' variable. (i.e. contains at least labels column listed in 'labels' and one or more numerical data columns)
        labels (str | list): labels(s) of the data points, these will be added to the PCA space output.
        embedding (PCA): already fitted PCA embedding you wish to use to transform the new datapoints in df

    Returns:
        pd.DataFrame: PCA components + additional labels(s) in a df that is seaborn/plotly plot proof
    """
    # Get numerical data from input df
    x = df.loc[:, ~df.columns.isin(labels)]

    # transform data points
    embedded_x = embedding.transform(x)

    # Bundle everything in a df
    df_pca = bundle_components_and_labels_into_df(
        labels=df[labels], components=embedded_x
    )

    return df_pca


######################################
# UMAP
######################################
# MARK: UMAP


def get_umap_df(
    df: pd.DataFrame,
    labels: str | list,
    n_components: int = 2,
    random_state: int = 42,
    n_neighbours: int = 2,
    min_dist: float = 0.1,
    metric: str = "euclidean",
) -> tuple[pd.DataFrame, UMAP]:
    """Calculates UMAP and returns the data in a easy to plot dataframe

    Args:
        df (pd.DataFrame): input dataframe that contains at least labels column listed in 'labels' and one or more numerical data columns
        labels (str | list): labels(s) of the data points, these will be added to the UMAP space output.
        n_components (int, optional): Number of components the UMAP should calculate+return. Defaults to 2.
        random_state (int, optional): Random state of the tSNE calculation, fix this to generate consistent plots. Defaults to 42.
        n_neighbours (int, optional): Controls how UMAP balances local versus global structure in the data. Defaults to 15
        min_dist (float, optional): Controls how tightly UMAP is allowed to pack points together. Defaults to 0.1
        metric (string, optional): Controls how distance is computed in the ambient space of the input data. Defaults to 'euclidean'

    Returns:
        pd.DataFrame: UMAP components + additional labels(s) in a df that is seaborn/plotly plot proof
        reducer: UMAP scikitlearn-like object that can be used to transform new data points
    """
    if n_components > 1:
        if n_components % 1 != 0:  # must be round number
            raise ValueError("n_components must be round number")
        if n_components >= df.shape[0] - 1:
            raise ValueError(
                f"n_components can maximimally be n_samples-2. n_samples=={df.shape[0]}, so n_components needs to be <= {df.shape[0]-2}"
            )
    elif n_components < 1:  # must be greater than or equal to 1
        raise ValueError("n_components must >= 1")
    else:
        pass

    # Get numerical data from input df
    x = df.loc[:, ~df.columns.isin(labels)]

    # Calc PCA
    reducer = UMAP(
        n_components=n_components,
        random_state=random_state,
        n_neighbors=n_neighbours,
        min_dist=min_dist,
        metric=metric,
    )
    embedding = reducer.fit_transform(x)

    # Bundle everything in a df
    df_umap = bundle_components_and_labels_into_df(
        labels=df[labels], components=embedding
    )

    return df_umap, reducer


def umap_transform(
    df: pd.DataFrame, labels: str | list, embedding: UMAP
) -> pd.DataFrame:
    """Transforms new data points in the same UMAP space as that of the datapoints the embedding object was fitted with. Returns an easy to plot dataframe.

    Args:
        df (pd.DataFrame): input dataframe with the same format as that is used to train the 'embedding' variable. (i.e. contains at least labels column listed in 'labels' and one or more numerical data columns)
        labels (str | list): labels(s) of the data points, these will be added to the UMAP space output.
        embedding (UMAP): already fitted UMAP embedding you wish to use to transform the new datapoints in df

    Returns:
        pd.DataFrame: UMAP components + additional labels(s) in a df that is seaborn/plotly plot proof
    """
    # Get numerical data from input df
    x = df.loc[:, ~df.columns.isin(labels)]

    # transform data points
    embedded_x = embedding.transform(x)

    # Bundle everything in a df
    df_umap = bundle_components_and_labels_into_df(
        labels=df[labels], components=embedded_x
    )

    return df_umap


######################################
# Plotting
######################################
# MARK: Plotting


def confidence_ellipse(x, y, n_std: float = 1.96, size: int = 100) -> str:
    """Get the covariance confidence ellipse for data set x, y

    Args:
        x (array with shape (n, )): x-axis data points
        y (array with shape (n, )): y-axis data points
        n_std (float): The radius of the ellipse in standard deviations.
        size (int):  Number of points defining the ellipse

    Returns:
        str containing an SVG path for the ellipse

    References (H/T)
    ----------------
    https://matplotlib.org/3.1.1/gallery/statistics/confidence_ellipse.html
    https://community.plotly.com/t/arc-shape-with-path/7205/5
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    if size < 3:
        raise ValueError(
            "size must be >=3 to at least encircle some data points (recommended: 100) "
        )

    # Calc ellipse coordinates
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    theta = np.linspace(0, 2 * np.pi, size)
    ellipse_coords = np.column_stack(
        [ell_radius_x * np.cos(theta), ell_radius_y * np.sin(theta)]
    )

    # Calculate the scale (=size) and mean (offset) from the x and y coordinates.
    # the std. dev. of x from the sqrt of the variance and multiplying this with the n_std argument.
    x_scale = np.sqrt(cov[0, 0]) * n_std
    y_scale = np.sqrt(cov[1, 1]) * n_std
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # translate and rotate the elipse, so it encapsulates the input data points
    translation_matrix = np.tile([x_mean, y_mean], (ellipse_coords.shape[0], 1))
    rotation_matrix = np.array(
        [
            [np.cos(np.pi / 4), np.sin(np.pi / 4)],
            [-np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ]
    )
    scale_matrix = np.array([[x_scale, 0], [0, y_scale]])
    ellipse_coords = (
        ellipse_coords.dot(rotation_matrix).dot(scale_matrix) + translation_matrix
    )

    # Calc ellipse path which can be plotted with fig.add_shape(path=ellipse_path)
    ellipse_path = f"M {ellipse_coords[0, 0]}, {ellipse_coords[0, 1]}"
    for k in range(1, len(ellipse_coords)):
        ellipse_path += f"L{ellipse_coords[k, 0]}, {ellipse_coords[k, 1]}"
    ellipse_path += " Z"

    return ellipse_path


def plot_dim_reduction(
    df: pd.DataFrame,
    labels: str | list,
    hue_col: str,
    plot_ellips: bool = True,
    title: str = "tSNE plot",
) -> go.Figure:
    """Helper function to plot dimensional reduced data with optional confidence interval ellipse

    Args:
        df (pd.DataFrame): input dataframe with the same format as the output from the t-SNE functions (i.e. contains at least label column listed in 'labels' and one or more numerical data columns)
        labels (str | list): label(s) of the data points, needed for separating numerical columns
        hue_col (str): df column that determines the hue of the plot
        plot_ellips (bool): True to plot confidence interval ellipse
        title (str): title of the plot

    Returns:
        fig object of type go.Figure

    """
    # Generate unique colours for each unique entry in df[hue_col]:
    n_colours = len(df[hue_col].unique())
    colour_list = px.colors.sample_colorscale(
        "turbo", [n / (n_colours - 1) for n in range(n_colours)]
    )

    fig = go.Figure()

    # Loop over each 'hue' category
    for target_value, target_name in enumerate(df[hue_col].unique()):
        color = colour_list[target_value]
        x = df.loc[df[hue_col] == target_name, "Component_1"]
        y = df.loc[df[hue_col] == target_name, "Component_2"]
        text = df.loc[df[hue_col] == target_name, labels]
        # Scatterplot the data points
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=target_name,
                customdata=df.loc[df[hue_col] == target_name, labels],
                mode="markers",
                marker={"color": color, "size": 10},
                text=text,
                hovertemplate="%{customdata}",
            )
        )
        # Make title bold
        str_title = f"<b>{title}</b>"

        # Plot Ellipse (optional)
        if plot_ellips:
            # Calculate elipsoid path
            ellips_path = confidence_ellipse(x, y, n_std=1.960)
            # Dashed ellipsoid line
            fig.add_shape(
                type="path",
                path=ellips_path,
                line={"dash": "dot"},
                line_color=color,
                line_width=3,
                opacity=0.5,
            )
            # Fill in ellipsoid
            fig.add_shape(
                type="path",
                path=ellips_path,
                line={"dash": "dot"},
                line_color=color,
                fillcolor=color,
                opacity=0.03,
            )
            str_title = str_title + " + 90% confidence interval"

    # Set layout
    fig.update_layout(
        width=1200,
        height=700,
        plot_bgcolor="white",
        xaxis={"showticklabels": False},
        yaxis={"showticklabels": False},
        showlegend=True,
        font_size=17,
        title=str_title,
    )

    return fig


def plot_dim_reduction_v2(
    df: pd.DataFrame,
    labels: str | list,
    hue_col: str,
    plot_ellips: bool = True,
    title: str = "tSNE plot",
) -> go.Figure:
    """Helper function to plot dimensional reduced data with optional confidence interval ellipse

    Args:
        df (pd.DataFrame): input dataframe with the same format as the output from the t-SNE functions (i.e. contains at least label column listed in 'labels' and one or more numerical data columns)
        labels (str | list): label(s) of the data points, needed for separating numerical columns
        hue_col (str): df column that determines the hue of the plot
        plot_ellips (bool): True to plot confidence interval ellipse
        title (str): title of the plot

    Returns:
        fig object of type go.Figure

    """
    # Generate unique colours for each unique entry in df[hue_col]:
    n_colours = len(df[hue_col].unique())
    colour_list = px.colors.sample_colorscale(
        "turbo", [n / (n_colours - 1) for n in range(n_colours)]
    )

    fig = go.FigureWidget()

    for target_value, target_name in enumerate(df[hue_col].unique()):
        color = colour_list[target_value]
        x = df.loc[df[hue_col] == target_name, "Component_1"]
        y = df.loc[df[hue_col] == target_name, "Component_2"]
        text = df.loc[df[hue_col] == target_name, labels]

        # Scatterplot the data points
        fig.add_scatter(
            x=x,
            y=y,
            name=target_name,
            customdata=df.loc[df[hue_col] == target_name, labels],
            mode="markers",
            marker={"color": color, "size": 7},
            text=text,
            hovertemplate="%{customdata}",
        )
        # Scatter line plot for fermentation path arrows
        fig.add_scatter(
            x=[],
            y=[],
            mode="lines+markers",
            marker=dict(size=10, symbol="arrow-bar-up", angleref="previous"),
            customdata=df.loc[df[hue_col] == target_name, labels],
            hovertemplate="%{customdata}",
        )
        # Make title bold
        str_title = f"<b>{title}</b>"

        # Plot Ellipse (optional)
        if plot_ellips:
            # Calculate elipsoid path
            ellips_path = confidence_ellipse(x, y, n_std=1.960)
            fig.add_shape(
                type="path",
                path=ellips_path,
                line={"dash": "dot"},
                line_color=color,
                line_width=3,
                opacity=0.5,
            )
            fig.add_shape(
                type="line",
                path=ellips_path,
                line={"dash": "dot"},
                line_color=color,
                fillcolor=color,
                opacity=0.03,
            )
            str_title = str_title + " + 90% confidence interval"

    fig.update_layout(
        width=1000,
        height=700,
        plot_bgcolor="white",
        xaxis={"showticklabels": False},
        yaxis={"showticklabels": False},
        showlegend=True,
        font_size=17,
        title=str_title,
    )

    line = fig.data[1]
    return fig
