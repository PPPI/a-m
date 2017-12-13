# Aide-memoir
This project represents the backend and model code for A-m. This can be used to train a model used for PR-Issue link prediction as:
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

The corpora and truth data cand be found at and downloaded from: https://1drv.ms/f/s!AnfFX0y_EVFMmoxKP8NchZzn53RcHw

The backend can be used for replication as a standalone or it can be used together with a chrome plug-in.
For the latter more details at: https://github.com/PPPI/tlinker-chrome

This work was done for the purpose of the following paper: https://github.com/PPPI/tlinker-tex