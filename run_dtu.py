#
# Sparse-GS2Mesh on sparse-view DTU: train -> extract mesh -> cull -> Chamfer distance
#
#   python run_dtu.py --gpu 0 --scans 24 37 --output output/dtu_little
#
import os
import sys
import csv
import json
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
DTU_SCANS = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

# candidate baselines in % of the scene radius (Sec 4.2)
BASELINES = {
    'little': list(range(3, 16, 1)),
    'large': list(range(20, 81, 4)),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sparse-GS2Mesh on sparse-view DTU")
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--scans', type=int, nargs='+', default=DTU_SCANS)
    parser.add_argument('--setting', choices=['little', 'large'], default='little', help='overlap of the 3 input views')
    parser.add_argument('--data_dir', type=str, default=os.path.join(ROOT, 'data', 'DTU_little_GT'),
                        help='scan folders (images, sparse/0, mask, flow, cameras.npz)')
    parser.add_argument('--dtu_dir', type=str, default=None,
                        help='official DTU evaluation data (Points/stl, ObsMask). default: <data_dir>/SampleSet/MVS_Data')
    parser.add_argument('--output', type=str, default=os.path.join(ROOT, 'output', 'dtu'))
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_mesh', action='store_true')
    parser.add_argument('--skip_eval', action='store_true')
    return parser.parse_args()


def run(cmd, env):
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=ROOT)


def main():
    args = parse_args()
    dtu_dir = args.dtu_dir or os.path.join(args.data_dir, 'SampleSet', 'MVS_Data')
    os.makedirs(args.output, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    py = sys.executable

    csv_file = os.path.join(args.output, 'chamfer.csv')
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(['scan', 'd2s', 's2d', 'chamfer'])

    for scan in args.scans:
        source = os.path.join(args.data_dir, f'scan{scan}')
        model = os.path.join(args.output, f'scan{scan}')
        common = ['-s', source, '-m', model, '-r', '1']

        if not args.skip_train:
            run([py, 'train.py', *common, '--renderer_baseline_candidate', *map(str, BASELINES[args.setting])], env)

        if not args.skip_mesh:
            run([py, 'extract_mesh.py', *common], env)

        if not args.skip_eval:
            eval_dir = os.path.join(model, 'eval')
            os.makedirs(eval_dir, exist_ok=True)
            culled = os.path.join(eval_dir, f'culled_scan{scan}.ply')
            run([py, 'evaluation/dtu/cull_mesh.py', '--scan_dir', source,
                 '--mesh', os.path.join(model, 'mesh', 'mesh_cleaned.ply'), '--out', culled], env)
            run([py, 'evaluation/dtu/eval.py', '--data', culled, '--scan', str(scan), '--mode', 'mesh',
                 '--dataset_dir', dtu_dir, '--vis_out_dir', eval_dir], env)
            with open(os.path.join(eval_dir, 'results.json')) as f:
                res = json.load(f)
            with open(csv_file, 'a', newline='') as f:
                csv.writer(f).writerow([scan, res['mean_d2s'], res['mean_s2d'], res['overall']])
            print(f"scan{scan}: chamfer {res['overall']:.4f}")


if __name__ == "__main__":
    main()
