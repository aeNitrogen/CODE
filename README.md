# Thesis code

## Configuration
In the configs directory, several configurations can be found.
A lot of them are outdated. Usually configs ending with an underscore and a large letter are up to date.
Most of the values are self-explanatory or have a comment, otherwise hints are in patchAdapter.py.

## Adding new Datasets
If they are in the same format as the ones used in the thesis, only a few lines of code on data_loader.py are necessary.
These will include batch size for optional hyperparameter tuning on reduced batches and optional normalization for datasets that come with a normalizer but not normalized, like SinData.

## Adding new Models
New models can be added in training_iterator.py. For most models, one of the iterators already existing should be fine.
Otherwise, new iterators can be added easily by looking at the structure of the existing ones.

## Adding new Hyperparameters
patchAdapter.py translates the configuration dict to a namespace that most implementations use.
Add the new hyperparameters in that translation or as an optional update at the bottom of it.


