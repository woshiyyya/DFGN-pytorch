from pytorch_pretrained_bert.modeling import BertModel
from model.GFN import *
from utils import *
from tools.data_iterator_pack import IGNORE_INDEX
import numpy as np
from config import set_config
from tools.data_iterator_pack import DataIteratorPack
from text_to_tok_pack import *


def large_batch_encode(bert_model, batch, encoder_gpus, max_bert_bsz):
    doc_ids, doc_mask, segment_ids = batch['context_idxs'], batch['context_mask'], batch['segment_idxs']
    N = doc_ids.shape[0]

    doc_ids = doc_ids.cuda(encoder_gpus[0])
    doc_mask = doc_mask.cuda(encoder_gpus[0])
    segment_ids = segment_ids.cuda(encoder_gpus[0])
    doc_encoding = []

    ptr = 0
    while ptr < N:
        # TODO finetune bert
        all_doc_encoder_layers = bert_model(input_ids=doc_ids[ptr:ptr+max_bert_bsz],
                                            token_type_ids=segment_ids[ptr:ptr+max_bert_bsz],
                                            attention_mask=doc_mask[ptr:ptr+max_bert_bsz],
                                            output_all_encoded_layers=False)
        tem_doc_encoding = all_doc_encoder_layers.detach()
        doc_encoding.append(tem_doc_encoding)
        ptr += max_bert_bsz
        del all_doc_encoder_layers

    doc_encoding = torch.cat(doc_encoding, dim=0)
    return doc_encoding


def dispatch(context_encoding, context_mask, batch, device):
    batch['context_encoding'] = context_encoding.cuda(device)
    batch['context_mask'] = context_mask.float().cuda(device)
    return batch


def compute_loss(batch, start, end, sp, Type, masks):
    loss1 = criterion(start, batch['y1']) + criterion(end, batch['y2'])
    loss2 = args.type_lambda * criterion(Type, batch['q_type'])
    loss3 = args.sp_lambda * criterion(sp.view(-1, 2), batch['is_support'].long().view(-1))
    loss = loss1 + loss2 + loss3

    loss4 = 0
    if args.bfs_clf:
        for l in range(args.n_layers):
            pred_mask = masks[l].view(-1)
            gold_mask = batch['bfs_mask'][:, l, :].contiguous().view(-1)
            loss4 += binary_criterion(pred_mask, gold_mask)
        loss += args.bfs_lambda * loss4

    return loss, loss1, loss2, loss3, loss4


def predict(model, dataloader, example_dict, feature_dict, prediction_file):
    model.eval()
    answer_dict = {}
    sp_dict = {}
    dataloader.refresh()
    total_test_loss = [0] * 5
    for batch in tqdm(dataloader):
        context_encoding = large_batch_encode(encoder, batch, encoder_gpus, args.max_bert_size)
        batch = dispatch(context_encoding, batch['context_mask'], batch=batch, device=model_gpu)
        del context_encoding

        start, end, sp, Type, softmask, ent, yp1, yp2 = model(batch, return_yp=True)

        loss_list = compute_loss(batch, start, end, sp, Type, softmask)

        for i, l in enumerate(loss_list):
            if not isinstance(l, int):
                total_test_loss[i] += l.item()

        answer_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'], yp1.data.cpu().numpy().tolist(),
                                         yp2.data.cpu().numpy().tolist(), np.argmax(Type.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(sp[:, :, 1]).data.cpu().numpy()
        # predict_support_np = torch.sigmoid(sp).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = []
            cur_id = batch['ids'][i]
            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):
                    break
                if predict_support_np[i, j] > args.sp_threshold:
                    cur_sp_pred.append(example_dict[cur_id].sent_names[j])
            sp_dict.update({cur_id: cur_sp_pred})

    prediction = {'answer': answer_dict, 'sp': sp_dict}
    with open(prediction_file, 'w') as f:
        json.dump(prediction, f)

    for i, l in enumerate(total_test_loss):
        print("Test Loss{}: {}".format(i, l / len(dataloader)))
    test_loss_record.append(sum(total_test_loss[:3]) / len(dataloader))
    model.train()


if __name__ == "__main__":
    args = set_config()
    os.makedirs('ckpt', exist_ok=True)
    os.makedirs('output/submission', exist_ok=True)
    setting_file = 'ckpt/run_settings.json'
    checkpoint_path = 'ckpt/checkpoint.pth'
    prediction_path = 'output/submission/prediction.json'

    load_settings(args, setting_file)
    for k, v in args.__dict__.items():
        print(k, v)

    args.encoder_gpu = args.model_gpu = "0"
    args.batch_size = 5

    # Allocate Models on GPU
    encoder_gpus = [int(i) for i in args.encoder_gpu.split(',')]
    model_gpu = 'cuda:{}'.format(args.model_gpu)

    encoder = BertModel.from_pretrained(args.bert_model)
    encoder.cuda(encoder_gpus[0])
    encoder = torch.nn.DataParallel(encoder, device_ids=encoder_gpus)
    encoder.eval()

    with gzip.open('data/dev_example.pkl.gz', 'rb') as fin:
        examples = pickle.load(fin)
        example_dict = {e.qas_id: e for e in examples}
    
    with gzip.open('data/dev_feature.pkl.gz', 'rb') as fin:
        features = pickle.load(fin)
        feature_dict = {f.qas_id: f for f in features}
    
    with gzip.open('data/dev_graph.pkl.gz', 'rb') as fin:
        graph_dict = pickle.load(fin)
    
    args.n_type = 2  # here when graph=11
    eval_dataset = DataIteratorPack(features=features,
                                    example_dict=example_dict,
                                    graph_dict=graph_dict,
                                    bsz=args.batch_size,
                                    device='cuda:{}'.format(args.model_gpu),
                                    sent_limit=25,
                                    entity_limit=80,
                                    sequential=True,
                                    n_layers=args.n_layers)

    model = GraphFusionNet(config=args)
    model.load_state_dict(torch.load(checkpoint_path))
    model.cuda(model_gpu)

    criterion = nn.CrossEntropyLoss(size_average=True, ignore_index=IGNORE_INDEX)
    binary_criterion = nn.BCEWithLogitsLoss(size_average=True)

    test_loss_record = []
    predict(model=model,
            dataloader=eval_dataset,
            example_dict=example_dict,
            feature_dict=feature_dict,
            prediction_file=prediction_path)

