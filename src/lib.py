#!/usr/bin/env python

import html
import pickle as pkl
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from flask import url_for

from .utils import fopen, get_image_db

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
CWD = Path(__file__).parent
DATA_PATH = f"{CWD}/../data"
IMAGE_DB = get_image_db(f'{DATA_PATH}')
THUMBS = pkl.load(open(f'{DATA_PATH}/flickr+coco.images.pkl', 'rb'))


def parse_ranksys(ranksys_root, sys_name, test_set, src_sents):
    def parse_sent(line):
        return line.split(':', 1)[-1].strip()

    def format_sent(line, label):
        line = html.escape(parse_sent(line))
        return f'<span class="{label.lower()}"><b>[{label}] </b></span>{line}<br/>'

    def format_delta(x):
        if x < 0:
            return f'<span class="delta" style="color:red">{x}</span>'
        elif x > 0:
            return f'<span class="delta" style="color:green">{x}</span>'
        else:
            return f'<span class="delta">{x}</span>'

    result = []
    ranksys = list(ranksys_root.glob('{}.sortedby.*'.format(sys_name)))[0]
    img_names = IMAGE_DB[test_set]

    with fopen(str(ranksys)) as f:
        sent_idx = None
        for line in f:
            line = line.strip()

            if line.startswith('SENT'):
                out_str = ""
                parts = line.replace('\t', ' ').split()
                sent_idx = int(parts[1])

                info_dict = {'ID': sent_idx, 'IMAGE': img_names[sent_idx]}

                # parse delta metrics
                # Order is: baseline, variant, delta
                order = [name[:-1].upper() for name in parts[2::6]]
                deltas = [float(d) for d in parts[7::6]]
                delta_dict = {}
                for metric, delta in zip(order, deltas):
                    delta_dict[f'\u0394{metric}'] = format_delta(delta)

                if src_sents is not None:
                    out_str += format_sent(src_sents[sent_idx], 'SRC')

            elif 'median-baseline' in line:
                # baseline's median hypothesis
                out_str += format_sent(line, 'BSL')
            elif 'median-variant' in line:
                # variant's median hypothesis
                out_str += format_sent(line, 'SYS')
            elif line.startswith('ref1'):
                # reference
                out_str += format_sent(line, 'REF')

                # Parsing completed for this sentence
                info_dict['SENTENCES'] = out_str
                info_dict.update(delta_dict)
                result.append(info_dict)

#                 result.append(OrderedDict({
                        # 'ID': sent_idx,
                        # 'IMAGE': img_names[sent_idx],
                        # 'SENTENCES': out_str,
                        # '\u0394METEOR': format_delta(meteors[2]),
                        # '\u0394BLEU': format_delta(bleus[2]),
                        # '\u0394TER': format_delta(ters[2]),
                    # }))

    # Sort the test set
    result = sorted(result, key=lambda x: x['ID'])

    classes = "display compact row-border cell-border stripe systems"
    return pd.DataFrame.from_dict(result).to_html(
        escape=False, index=False, classes=classes, border=0,
        formatters={
            'ID': lambda s: f'<span style="font-size: 14px;"><b>{s}</b></span>',
            'IMAGE': lambda s:
                '<img width="150" src="data:image/jpeg;base64,{}">'.format(THUMBS[s])})


def parse_multeval_results_table(fname, task, testset):
    """Parses the results.txt produced by multeval python wrapper to detect
    the available systems."""
    result = []

    def parse_line(line, headers):
        values = list(filter(
            lambda s: s, map(lambda s: s.strip(), line.split('  '))))
        return OrderedDict({k: v for k, v in zip(headers, values)})

    def convert_to_link(sys_name):
        if sys_name.startswith('baseline:'):
            return sys_name
        url = url_for('results', task=task, testset=testset,
                      system=sys_name)
        return f'<a href="{url}">{sys_name}</a>'

    with open(fname) as f:
        header = f.readline().strip()
        n_runs = int(header.split()[4][-1])
        metrics = header.split()[5:]
        # add description of parentheses fields
        metrics = [f'{m} <em>(\u03C3-sel/\u03C3-opt/p)</em>' for m in metrics]
        headers = [f'SYSTEM ({n_runs} runs)'] + metrics
        # Skip empty line
        f.readline()
        # Get baseline system
        result.append(parse_line(f.readline().strip(), headers))
        baseline_name = result[0][headers[0]].split()[-1]
        for line in f:
            line = line.strip()
            if line:
                system = parse_line(line, headers)
                # Skip double baseline
                if system[headers[0]] != baseline_name:
                    result.append(system)

    return pd.DataFrame.from_dict(result).to_html(
        index=False, border=0, justify='left', escape=False,
        classes="display compact row-border multeval",
        formatters={
            # Convert system names to links
            headers[0]: lambda s: convert_to_link(s)}), baseline_name
