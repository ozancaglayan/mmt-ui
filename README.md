# mmt-ui

This simple web interface allows you to parse and visualize the `ranksys` folder produced by the [multeval](https://github.com/jhclark/multeval) utility.

## Requirements

- A [modified version](https://github.com/ozancaglayan/multeval) of the `multeval` utility is required to produce the results folders. Specifically, a [python wrapper](https://github.com/ozancaglayan/multeval/blob/master/multeval.py) around the `multeval` Java toolkit constructs a folder hierarchy similar to below for each experiment:

```
<results folder>
├── <src_language>-<trg_langage>_<experiment name> (ex: en-fr_multimodal_att_v1)
│   ├── <test set> (ex: test_2016_flickr)
│   │   ├── ranksys
│   │   │   ├── snmt-rnn-multimodal.sortedby.bleu
│   │   │   ├── snmt-rnn-multimodal.sortedby.meteor
│   │   │   ├── snmt-rnn-multimodal.sortedby.ter
│   │   │   ├── wait1-rnn-multimodal.sortedby.bleu
│   │   │   ├── wait1-rnn-multimodal.sortedby.meteor
│   │   │   ├── wait1-rnn-multimodal.sortedby.ter
│   │   │   ├── wait1-rnn-unimodal.sortedby.bleu
│   │   │   ├── wait1-rnn-unimodal.sortedby.meteor
│   │   │   └── wait1-rnn-unimodal.sortedby.ter
│   │   ├── results.tex
│   │   ├── results.txt
│   │   └── srcs.pkl.bz2
│   └── <another test set>
│       ├── ranksys
│       │   ├── ...
│       ├── results.tex
│       ├── results.txt
│       └── srcs.pkl.bz2
├── <another experiment folder>
│   ├── ...
```

- Every test set folder contains a `srcs.pkl.bz2` file which is a dictionary packing the list of pre-processed source sentences as they are provided to the MT model during training. If the file is missing, source sentences will not be shown in the web interface. The file is a bz2-compressed `pickle` file, which has the following structure:

```
{
  'snmt-rnn-unimodal': [src_sent_1, ..., src_sent_N],
  'snmt-rnn-multimodal': [src_sent_1, ..., src_sent_N],
  'wait1-rnn-unimodal': [src_sent_1, ..., src_sent_N],
  'wait1-rnn-multimodal': [src_sent_1, ..., src_sent_N],
}
```

- Current code relies on Multi30k test sets and embeds the images for `test_2016_flickr`, `test_2017_flickr` and `test_2017_mscoco` test sets into a `pickle` file under the `data/` folder. If your `ranksys` folders are named to include these identifiers for test sets, everything should work out-of-the-box.

## Running
After installing the runtime dependencies with `pip install -r requirements.txt`, run the following command and fire up your browser to point at `localhost:8080`.

```
$ python server.py -r <results folder>
```

## Example
![Example](screenshot.jpg?raw=true)
