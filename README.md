
<h1 align="center"> Landmark sampling </h1> <br>

Creating a representative training subset can be difficult, particularly for weighted datasets. To address this issue, a program has been developed that reduces the dataset size by using the roulette wheel selection method in combination with k-means clustering to create a training dataset that can be used for further analysis.

## Documentation
The program reads an input file with a dataset and constructs a training set of a specified size, returning it in the form of a CSV file. It includes an optimized method for roulette wheel selection, as well as modified k-means and DBSCAN clustering algorithms that enable the grouping of periodic data by creating a special distance matrix. The program allows the user to modify necessary parameters and provides the ability to display and save additional visualizations of the iterative selection process.

## Input parameters:
- **_dataset_** - name of the dataset; possible values **_1_** - one-dimensional dataset, **_2_** - two-dimensional dataset (default value 2),
- **_size_** - size of the training set,
- **_n_clusters_** - number of clusters,
- **_n_samples_** - maximum number of samples selected at once from a cluster,
- **_figures_** - the value "y" (yes) to display additional plots during sampling after each iteration,
- **_path_** - path to visualizations.


## Output parameters:
- **_results.csv_** - file containing the training set,
- **_elbow_method.pdf_** - optimal number of clusters - plot,
- **_kmeans.pdf_** - plot visualizing the data division,
- **_selection_progress_1.pdf_**, **_selection_progress_2.pdf_**, ..., set of plots showing the selection progress,
- **_results_scatter.pdf**_ - data distribution plot,
- **_results_histogram.pdf_** - empirical distribution of a feature plot.

## Results in iterations:
<p align="center">
  <img alt="results" title="results" src="/images/selection_results.jpg" width="450">
</p>
