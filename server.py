#!/usr/bin/env python
import bz2
import pickle
import argparse
from collections import defaultdict
from pathlib import Path

from flask import render_template, Flask, url_for
from flask_caching import Cache

from src.lib import parse_multeval_results_table, parse_ranksys
from src.utils import natural_sort


CONFIG = {
    "CACHE_TYPE": "simple",     # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 36000,
}

app = Flask('mmt-ui')
app.config.from_mapping(CONFIG)
cache = Cache(app)


def get_tree_dict(folder):
    """Parses a folder hierarchy where each subfolder is a multeval
    output folder that contains experiment results into a dict."""

    def read_sources(fname):
        """Reads srcs.pkl.bz2 files to get source sentences used in MT
        training for visualization purposes."""
        try:
            with bz2.BZ2File(fname, 'rb') as f:
                d = pickle.load(f)
        except Exception:
            return None

        return {k: d[v] if isinstance(v, str) else v for k, v in d.items()}

    # Final dictionary has tasks as keys, test_sets as inner keys
    # and a tuple of (path_to_folder, URL for results page, source sentences)
    # as value
    d = defaultdict(lambda: defaultdict(dict))

    # The folder with experiment results
    tasks = [exp.name for exp in Path(folder).iterdir() if exp.is_dir()]
    tasks = natural_sort(tasks)

    for task in tasks:
        # Each subfolder is an experiment's multeval results
        # srclang-trglang_<task description>
        slang, tlang = task.split('_', 1)[0].split('-')
        for test_set in Path(f'{folder}/{task}').iterdir():
            source_dict = read_sources(test_set / 'srcs.pkl.bz2')

            d[task][test_set.name] = (
                Path(f'{folder}/{task}/{test_set.name}'),
                url_for('results', task=task, testset=test_set.name),
                source_dict)
    return d


@app.route("/")
def index():
    return render_template('index.html', tasks=get_tree_dict(app.config['results']))


@app.route("/<task>/<testset>")
@app.route("/<task>/<testset>/<system>")
@cache.memoize(timeout=36000)
def results(task, testset, system=None):

    result_db = get_tree_dict(app.config['results'])
    folder, _, source_dict = result_db[task][testset]

    # Parse multeval table
    results_table, baseline = parse_multeval_results_table(
        folder / 'results.txt', task, testset)

    kwargs = {'task': task, 'testset': testset, 'results_table': results_table}

    if system is not None:
        srcs = source_dict[system] if source_dict else None
        kwargs['system'] = system
        kwargs['baseline'] = baseline
        kwargs['systems_table'] = parse_ranksys(
            folder / 'ranksys', system, testset, srcs)

    return render_template('view.html', **kwargs)


def main():
    parser = argparse.ArgumentParser(prog='mmt-ui')

    parser.add_argument('-r', '--results', help='Results folder',
                        required=True, type=str)
    parser.add_argument('-p', '--port', help='Server port', default=8086)
    parser.add_argument('-n', '--host', help='Host server IP', default='0.0.0.0')
    parser.add_argument('-d', '--debug', help='Debug mode for Flask',
                        action='store_true')

    args = parser.parse_args()

    app.config['results'] = args.results
    app.config['DEBUG'] = args.debug
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == '__main__':
    main()
