# Aide-memoir
This project represents the backend and model code for A-m. 
Please use the provided `setup.py` to install the package and dependencies.

This can be used to train a model used for PR-Issue link prediction as:
```
filename = 'data/google_guava.json'
with open(filename) as f:
    repo = jsonpickle.decode(f.read())
with open(filename[:-len('.json')] + '_truth.json') as f:
    truth = jsonpickle.decode(f.read())

print('Loaded repository %s' % repo.name)
Linker = Linker(net_size_in_days=14, min_tok_len=2, undersample_multiplicity=100)
Linker.fit(repo, truth)
print('Trained Random Forest classifier')
out_path = os.path.join(os.getcwd(), 'models', repo.name[1:].translate({ord(c): '_' for c in '\\/'}))
os.makedirs(out_path, exist_ok=True)
Linker.persist_to_disk(out_path)
print('Recorded model to disk')
```

Or make use of the `generate_models.py` script in `backend`.
```
python backend/generate_models.py <path_to_crawled_repository>
```

To start the backend, edit `config.json` to point to the desired models and local gits and start it with:
```
python backend/backend.py
```
This should be done before the chrome plug-in may be used.

To use a remote backend, edit the `SERVER_ADDR` variable from `chrome_entry/__main__.py` from:
```
SERVER_ADDR = 'http://localhost:8000'
```
To the desired server.

The corpora and truth data can be found at and downloaded from [here](https://figshare.com/s/83c448eb518b3d04651f)

The crawling code is provided in `Prediction/gitScraper.py`.

The backend can be used for replication as a standalone or it can be used together with a chrome plug-in.
For the latter more details at: https://github.com/PPPI/am-chrome

This work was done for the purpose of the following paper: [link](https://partachi.com/papers/am_just_accepted.pdf)

## Dependencies
This project assumes Python 3.5 or newer.

```
jsonpickle       # For data storage
scikit-learn     # For classifier utility functions, recursive-feature elimination etc.
gensim           # For TF.IDF implementation
numpy            # For numerical operations
pygithub         # For interaction with github
pytz             # To handle timestamps and time data
pandas           # For results interpretation and output
scikit-garden    # For Mondrian Forest implementation
```
