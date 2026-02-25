import sys
import argparse
import os.path as osp
import datetime
import dateutil.tz
import ast
import uuid
import pickle as pickle
import base64
import joblib

from . import config


def run_experiment(argv):
    default_log_dir = config.LOG_DIR
    now = datetime.datetime.now(dateutil.tz.tzlocal())

    # avoid name clashes when running distributed jobs
    rand_id = str(uuid.uuid4())[:5]
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')

    default_exp_name = 'experiment_%s_%s' % (timestamp, rand_id)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_parallel', type=int, default=1,
                        help='Number of parallel workers to perform rollouts. 0 => don\'t start any workers')
    parser.add_argument(
        '--exp_name', type=str, default=default_exp_name, help='Name of the experiment.')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Path to save the log and iteration snapshot.')
    parser.add_argument('--snapshot_mode', type=str, default='all',
                        help='Mode to save the snapshot. Can be either "all" '
                             '(all iterations will be saved), "last" (only '
                             'the last iteration will be saved), "gap" (every'
                             '`snapshot_gap` iterations are saved), or "none" '
                             '(do not save snapshots)')
    parser.add_argument('--snapshot_gap', type=int, default=1,
                        help='Gap between snapshot iterations.')
    parser.add_argument('--tabular_log_file', type=str, default='progress.csv',
                        help='Name of the tabular log file (in csv).')
    parser.add_argument('--text_log_file', type=str, default='debug.log',
                        help='Name of the text log file (in pure text).')
    parser.add_argument('--params_log_file', type=str, default='params.json',
                        help='Name of the parameter log file (in json).')
    parser.add_argument('--variant_log_file', type=str, default='variant.json',
                        help='Name of the variant log file (in json).')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Name of the pickle file to resume experiment from.')
    parser.add_argument('--plot', type=ast.literal_eval, default=False,
                        help='Whether to plot the iteration results')
    parser.add_argument('--log_tabular_only', type=ast.literal_eval, default=False,
                        help='Whether to only print the tabular log information (in a horizontal format)')
    parser.add_argument('--seed', type=int,
                        help='Random seed for numpy')
    parser.add_argument('--args_data', type=str,
                        help='Pickled data for stub objects')
    parser.add_argument('--variant_data', type=str,
                        help='Pickled data for variant configuration')
    parser.add_argument('--use_cloudpickle', type=ast.literal_eval, default=False)

    args = parser.parse_args(argv[1:])

    if args.log_dir is None:
        log_dir = osp.join(default_log_dir, args.exp_name)
    else:
        log_dir = args.log_dir

    if args.variant_data is not None:
        variant_data = pickle.loads(base64.b64decode(args.variant_data))
    else:
        variant_data = None

    if args.resume_from is not None:
        data = joblib.load(args.resume_from)
        assert 'algo' in data
        algo = data['algo']
        algo.train()
    else:
        if args.use_cloudpickle:
            import cloudpickle
            method_call = cloudpickle.loads(base64.b64decode(args.args_data))
            method_call(variant_data, log_dir, args.exp_name)
        else:
            raise RuntimeError(
                "Non-cloudpickle deserialization is no longer supported. "
                "Use use_cloudpickle=True (the default in chester)."
            )


if __name__ == "__main__":
    run_experiment(sys.argv)
