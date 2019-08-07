import argparse
import os
import json
from os.path import join


def process_arguments(args):
    args.checkpoint_path = join(args.checkpoint_path, args.name)
    args.prediction_path = join(args.prediction_path, args.name)
    args.n_layers = int(args.gnn.split(':')[1].split(',')[0])
    args.n_heads = int(args.gnn.split(':')[1].split(',')[1])
    args.max_query_len = 50
    args.max_doc_len = 512


def save_settings(args):
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.prediction_path, exist_ok=True)
    json.dump(args.__dict__, open(join(args.checkpoint_path, "run_settings.json"), 'w'))


def set_config():
    parser = argparse.ArgumentParser()
    data_path = 'output'

    # Required parameters
    parser.add_argument("--name", type=str, default='default')
    parser.add_argument("--prediction_path", type=str, default=join(data_path, 'submissions'))
    parser.add_argument("--checkpoint_path", type=str, default=join(data_path, 'checkpoints'))

    parser.add_argument("--ckpt_id", type=int, default=0)
    parser.add_argument("--bert_model", type=str, default='bert-base-uncased',
                        help='Currently only support bert-base-uncased and bert-large-uncased')

    # learning and log
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--qat_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_bert_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument('--decay', type=float, default=1.0)
    parser.add_argument('--early_stop_epoch', type=int, default=0)
    parser.add_argument("--verbose_step", default=50, type=int)
    parser.add_argument("--grad_accumulate_step", default=1, type=int)

    parser.add_argument('--q_update', action='store_true', help='Whether update query')
    parser.add_argument('--basicblock_trans', action='store_true', help='transformer version basicblock')
    parser.add_argument("--prediction_trans", action='store_true', help='transformer version prediction layer')
    parser.add_argument("--trans_drop", type=float, default=0.5)
    parser.add_argument("--trans_heads", type=int, default=3)

    # device
    parser.add_argument("--encoder_gpu", default='1', type=str, help="device to place bert encoder.")
    parser.add_argument("--model_gpu", default='0', type=str, help="device to place model.")
    parser.add_argument("--input_dim", type=int, default=768, help="bert-base=768, bert-large=1024")

    # bi attn
    parser.add_argument("--bi_attn_drop", type=float, default=0.3)
    parser.add_argument("--hidden_dim", type=int, default=300)

    # graph net
    parser.add_argument('--tok2ent', default='mean_max', type=str, help='{mean, mean_max}')
    parser.add_argument('--gnn', default='gat:2,2', type=str, help='gat:n_layer, n_head')
    parser.add_argument("--gnn_drop", type=float, default=0.5)
    parser.add_argument("--gat_attn_drop", type=float, default=0.5)
    parser.add_argument('--q_attn', action='store_true', help='whether use query attention in GAT')
    parser.add_argument("--lstm_drop", type=float, default=0.3)

    # loss
    parser.add_argument("--type_lambda", type=float, default=1)
    parser.add_argument("--sp_lambda", type=float, default=5)
    parser.add_argument('--bfs_clf', action='store_true', help='Add BCELoss on bfs mask')
    parser.add_argument('--bfs_lambda', type=float, default=1)
    parser.add_argument("--sp_threshold", type=float, default=0.5)

    args = parser.parse_args()

    process_arguments(args)
    save_settings(args)

    return args
