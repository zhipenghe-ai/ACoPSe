import argparse
import json
from pathlib import Path


def parse_bool(value: str) -> bool:
    """Convert an explicit Boolean CLI value."""
    normalized = value.strip().lower()
    if normalized in ('true', '1', 'yes', 'on'):
        return True
    if normalized in ('false', '0', 'no', 'off') or value == '':
        return False
    raise argparse.ArgumentTypeError('expected a Boolean value (true/false, 1/0, yes/no, or on/off)')


def build_parser(preset):
    """Load the experiment defaults selected by the original launcher."""
    config_path = Path(__file__).with_name(f'{preset}.json')
    with config_path.open(encoding='utf-8') as config_file:
        defaults = json.load(config_file)

    parser = argparse.ArgumentParser(description=f"ACoPSe {defaults['DATASET']} experiment")
    parser.add_argument('--TRAIN', type=parse_bool, help='Train with true or evaluate with false (default: true)')
    parser.add_argument('--DATASET', type=str, help='MIRFlickr, NUSWIDE or COCO')
    parser.add_argument('--lambda1', type=float, help='Feature reconstruction weight')
    parser.add_argument('--lambda2', type=float, help='Affinity reconstruction weight')
    parser.add_argument('--lambda3', type=float, help='Prototype loss weight')
    parser.add_argument('--lambda4', type=float, help='Reserved legacy DCL parameter')
    parser.add_argument('--beta', type=float, help='Hash-function cosine loss weight')
    parser.add_argument('--LR_IMG', type=float, help='Image hash-function learning rate')
    parser.add_argument('--LR_TXT', type=float, help='Text hash-function learning rate')
    parser.add_argument('--LR_MyNet', type=float, help='Shared code generator learning rate')
    parser.add_argument('--K', type=int, help='Number of lowest affinities zeroed per row')
    parser.add_argument('--a1', type=float, help='Weight of image affinities in the initial mix')
    parser.add_argument('--a2', type=float, help='Weight of the pruned affinity matrix in the final mix')
    parser.add_argument('--HASH_BIT', type=int, help='Code length in bits')
    parser.add_argument('--BATCH_SIZE', type=int, help='Batch size')
    parser.add_argument('--GPU_ID', type=int, help='CUDA device index')
    parser.add_argument('--SEED', type=int, help='Torch and CUDA random seed')
    parser.add_argument('--NUM_WORKERS', type=int, help='Reserved legacy worker parameter')
    parser.add_argument('--EPOCH_INTERVAL', type=int, help='In-epoch progress logging setting')
    parser.add_argument('--NUM_EPOCH', type=int, help='Number of outer training epochs')
    parser.add_argument('--EVAL_INTERVAL', type=int, help='Epochs between retrieval evaluations')
    parser.add_argument('--MODEL_DIR', type=str, help='Checkpoint directory')
    if preset == 'mirflickr':
        parser.add_argument('--T', type=float, help='Legacy memory-bank temperature')
        parser.add_argument('--M', type=float, help='Legacy memory-bank momentum')
    parser.add_argument('--k_means', type=float, help='Prototype loss temperature')
    parser.add_argument('--cluster_num', type=int, help='Number of prototype slots')
    parser.add_argument('--EX_EPOCH', type=int, help='Extra hash-function epochs in the NUS-WIDE launcher')
    parser.add_argument('--temperature', type=float, help='Fusion DCL temperature')
    parser.add_argument('--warmup', type=int, help='Warm-up epochs in the NUS-WIDE launcher')
    parser.add_argument('--DATA_DIR', type=str, help='Root containing the pre-extracted dataset folders')
    parser.add_argument('--LOG_DIR', type=str, help='Run log directory')
    parser.set_defaults(**defaults)
    return parser
