import argparse


def keep_xyz(xyz, xmin, xmax, ymin, ymax, zmin, zmax):
    x, y, z = xyz

    if xmin is not None:
        if x <= xmin or x >= xmax:
            return False

    if ymin is not None:
        if y <= ymin or y >= ymax:
            return False

    if zmin is not None:
        if z <= zmin or z >= zmax:
            return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--txt-path', default=None)
    parser.add_argument('--xlim', default=None, type=float, nargs='+')
    parser.add_argument('--ylim', default=None, type=float, nargs='+')
    parser.add_argument('--zlim', default=None, type=float, nargs='+')
    args = parser.parse_args()

    xmax = xmin = ymax = ymin = zmax = zmin = None
    if args.xlim is not None:
        xmin, xmax = args.xlim
    if args.ylim is not None:
        ymin, ymax = args.ylim
    if args.zlim is not None:
        zmin, zmax = args.zlim

    with open(args.txt_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    kept = []
    for line in lines:
        splitted = line.split(' ')
        xyz = list(map(float, splitted[:3]))

        if keep_xyz(xyz, xmin, xmax, ymin, ymax, zmin, zmax):
            kept.append(line)

    new_path = args.txt_path.replace('.txt', '_limited.txt')
    with open(new_path, 'w') as f:
        for line in kept:
            f.write(f'{line}\n')
