import os
from collections import defaultdict 
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib   
# matplotlib.use('TkAgg')

SPLITS = ['train','val','test','all',['val','test']]

def plot(ours, theirs,queries,title=None):
    nsamples=len(ours)
    x = np.arange(nsamples)  # the label locations
    width = 0.33  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    offset = 0
    recs = ax.bar(x+offset, np.array(ours)*100, width, label='POP-3D (ours)')
    #ax.bar_label(recs, padding=3)
    
    offset = width
    recs = ax.bar(x+offset, np.array(theirs)*100, width, label='MaskCLIP+')
    #ax.bar_label(recs, padding=3)

    ax.set_xticks(x + width, queries, rotation=90)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 100)
    ax.set_ylabel('AP [%]')

    title='{}ours: {:.2f}, MaskCLIP+:{:.2f}, {} samples'.format('' if title is None else f'{title}, ', np.mean(ours)*100, np.mean(theirs)*100,nsamples)
    ax.set_title(title)
    print(title)

    plt.show()


def split2splits(parsed, split_name):
    splits, specs, queries, names, mAPs, mAPs_visible, mAPs_visible_fts = parsed
    _specs, _queries, _names, _mAPs, _mAPs_visible, _mAPs_visible_fts = [], [], [], [], [], []
    for i, split in enumerate(splits):
        if (type(split_name)==list and split in split_name) or (split==split_name) or (split_name=='all'):
            _specs.append(specs[i])
            _queries.append(queries[i])
            _names.append(names[i])
            _mAPs.append(mAPs[i])
            _mAPs_visible.append(mAPs_visible[i])
            _mAPs_visible_fts.append(mAPs_visible_fts[i])

    return _specs, _queries, _names, _mAPs, _mAPs_visible, _mAPs_visible_fts

def results2latex(queries, names, mAPs_visible, mAPs_visible_fts, show_queries=False):
    
    latex = []
    for query, name, mAP, mAP_fts in zip(queries, names, mAPs_visible, mAPs_visible_fts):
        cur = "{} & {:.2f} & {:.2f} \\\\".format(name, mAP*100, mAP_fts*100)
        if show_queries:
            cur = f"{query} & " + cur
        print(cur)



def parse_lines(lines):
    splits, specs, queries, names, mAPs, mAPs_visible, mAPs_visible_fts = [], [], [], [], [], [], []
    counts = defaultdict(int)
    counts_by_split = {'train':defaultdict(int),'val':defaultdict(int),'test':defaultdict(int)}
    for i,line in enumerate(lines):
        # print(line)
        split, spec, query, mAP, mAP_visible, _, mAP_visible_fts = line.split(';')
        counts[query]+=1
        counts_by_split[split][query]+=1
        splits.append(split)
        specs.append(spec)
        queries.append(query)
        names.append(f'{query} {counts[query]}')
        mAPs.append(float(mAP))
        try:
            mAPs_visible.append(float(mAP_visible))
        except:
            mAPs_visible.append(0.)
            print(f'Couldnt parse mAP_visible for spec {spec}')
        try:
            mAPs_visible_fts.append(float(mAP_visible_fts))
        except:
            mAPs_visible_fts.append(0.)
            print(f'Couldnt parse mAP_visible_fts for spec {spec}')

    print('query & total & val & test & train\\\\')
    for query, count in counts.items():
        print(f'{query} & {count} & {counts_by_split["val"][query]} & {counts_by_split["test"][query]} & {counts_by_split["train"][query]} \\\\')

    for query,_ in counts.items():
        print('\\rotatebox[origin=l]{90}{'+query+'} &')

    for split,split_counts in counts_by_split.items():
        row = '\\texttt{{{}}} &'.format(split) + ' & '.join([str(split_counts[query]) for query in counts.keys()]) + ' & {:d}'.format(sum(split_counts.values()))  + '\\\\'
        print(row)
    

    _sums = []
    for query in counts.keys():
        _sum = 0
        for split,split_counts in counts_by_split.items():
            _sum += split_counts[query]
        _sums.append(_sum)
    row = 'total & ' + ' & '.join([str(_sum) for _sum in _sums]) + '\\\\'
    print(row)


    print('\n***************************\n')

    split_order = ['val','test','train']
    header = 'query & ' + ' & '.join([split for split in split_order]) + '\\\\\n\\hline\\hline'
    print(header)

    split_sum = {split:0 for split in split_order}
    total = 0
    for query in counts.keys():
        _sum = 0
        _c=[]
        for split in split_order:
            split_counts = counts_by_split[split]
            _sum += split_counts[query]
            split_sum[split] += split_counts[query]
            _c.append(split_counts[query])
        row = f'{query} & ' + ' & '.join([str(c) for c in _c]) + f' & {_sum}\\\\'
        total += _sum
        print(row)
    sum_str = ' & '.join([str(split_sum[split]) for split in split_order]) + f' & {total}'
    print(f'\\hline\ntotal & {sum_str}')

    print('\n***************************\n')

    return splits, specs, queries, names, mAPs, mAPs_visible, mAPs_visible_fts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--results-file', type=str,
    #  default='/home/vobecant/PhD/TPVFormer-OpenSet/results/results_123554.txt',
     default='/home/vobecant/PhD/TPVFormer-OpenSet/results/results_130239.txt',
     )
    args = parser.parse_args()


    with open(args.results_file,'r') as f:
        lines = [l.strip() for l in f.readlines()]

    parsed = parse_lines(lines)


    for split in SPLITS:
        _specs, _queries,_names, _mAPs, _mAPs_visible, _mAPs_visible_fts = split2splits(parsed, split)
        if type(split)==list:
            split = ', '.join(split)
        plot(_mAPs_visible, _mAPs_visible_fts,_queries,title=split)
        print(f'***{split}***')
        results2latex(_queries,_names,_mAPs_visible, _mAPs_visible_fts)
        print()
