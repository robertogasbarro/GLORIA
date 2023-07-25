# A Graph Convolutional Network-based Approach for Review Spam Detection (GLORIA)

The repository contains code refered to the work:

_Giuseppina Andresini, Annalisa Appice, Roberto Gasbarro, Donato Malerba_

**GLORIA: A Graph Convolutional Network-based Approach for Review Spam Detection**   _(accepted for publication)_


Please cite our work if you find it useful for your research and work.

## Code requirements
* [PyTorch Geometric 2.3.0](https://pytorch-geometric.readthedocs.io/en/latest/)
* [Hyperopt 0.2.7](https://github.com/hyperopt)
* [NetworkX 3.1](https://networkx.org/)
* [Numpy 1.24.0](https://numpy.org/)
* [Pandas 1.3.1](https://pandas.pydata.org/)
* [Scikiu-learn 1.2.2](https://scikit-learn.org/stable/#)
* [PyTorch 1.13.1](https://pytorch.org/)

## Data
The datasets used for experiments are requestable form [__DATASETS__](https://www.cs.uic.edu/~liub/FBS/fake-reviews.html). 
These datasets contain reviews about Hotel and Restaurant from Yelp.
Data were preprocessed and features have been extracted from review texts, reviewers and products metadata. Details about preprocessing phase and feature extraction is described in the work [EUPHORIA:  A neural multi-view approach to combine content and behavioral features in review spam detection](https://doi.org/10.1016/j.jcmds.2022.100036)

## How to use
Repository contains scripts of all experiments included in the paper. All scripts are contained in src folder:
* __main.py__: script to run GLORIA
* __centrality.py__: script to compute centrality of nodes
* __scatter_plot.py__: script to plot scatter plots from centralities
* __graph_plot.py__: script to plot subgraphs

Code contains models used for experiments and an example dataset.

## Replicate the experiments
To replicate experiments reported in the work, you can use models stored in models folder. You need to download dataset from dataset author site and preprocess them as explained in the work [EUPHORIA:  A neural multi-view approach to combine content and behavioral features in review spam detection](https://doi.org/10.1016/j.jcmds.2022.100036)
Tu used serialized models, you need to edit global variables __config.conf__ and set 'test' mode and model name.

```python
    mode = test
    model_name = ModelCGmod2
    max_epochs = 1000
    patience = 30
    num_fake_user_feature = -1
    num_fake_item_feature = -1
    batch_size = 64
    model = path_to_serialized_model
```


## Download datasets
Original datasets are requestable from the page [DATASETS](https://www.cs.uic.edu/~liub/FBS/fake-reviews.html).
