import argparse
import copy
import os
import pickle

DATA_FOLDER = './data'


def main(args):
    skip_ratio = args.skip_ratio
    assert skip_ratio is not None and skip_ratio < 100 and skip_ratio > 0

    name = args.name
    if name is None:
        kept_perc = 100 // skip_ratio
        name_train = f'nuscenes_infos_{kept_perc}_train.pkl'
        name_val = f'nuscenes_infos_{kept_perc}_val.pkl'
    else:
        name_train = f'{name}_train.pkl'
        name_val = f'{name}_val.pkl'
    name_train = os.path.join(DATA_FOLDER, name_train)
    name_val = os.path.join(DATA_FOLDER, name_val)

    with open(os.path.join(DATA_FOLDER, 'nuscenes_infos_train.pkl'), 'rb') as f:
        train = pickle.load(f)
    train_new = copy.deepcopy(train)
    train_new['infos'] = train_new['infos'][::skip_ratio]
    with open(name_train, 'wb') as f:
        pickle.dump(train_new, f)
    print(f'New training split with skip_ratio={skip_ratio} saved to {name_train}')
    print(f'It contains {len(train_new["infos"])} samples (full set has {len(train["infos"])} samples).\n')

    with open(os.path.join(DATA_FOLDER, 'nuscenes_infos_val.pkl'), 'rb') as f:
        val = pickle.load(f)
    val_new = copy.deepcopy(val)
    val_new['infos'] = val_new['infos'][::skip_ratio]
    with open(name_val, 'wb') as f:
        pickle.dump(val_new, f)
    print(f'New validation split with skip_ratio={skip_ratio} saved to {name_val}')
    print(f'It contains {len(val_new["infos"])} samples (full set has {len(val["infos"])} samples).\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--skip-ratio', type=int, default=None)
    args = parser.parse_args()
    main(args)
