import argparse
from os.path import join
from tqdm import tqdm
from pytorch_pretrained_bert.modeling import BertModel
from model.GFN import *
from utils import *
from tools.data_iterator_pack import IGNORE_INDEX
import numpy as np
import queue
import threading
import time
import random
from config import set_config
from tools.data_helper import DataHelper
from text_to_tok_pack import *

BUF_SIZE = 5
data_queue = queue.Queue(BUF_SIZE)


class ProducerThread(threading.Thread):
    def __init__(self, data_loader, bert_model):
        super(ProducerThread, self).__init__()
        self.Loader = data_loader
        self.bert = bert_model

    def run(self):
        pbar = tqdm(total=len(Loader))
        while not self.Loader.empty():
            if not data_queue.full():
                batch = next(iter(self.Loader))
                context_encoding = large_batch_encode(self.bert, batch, encoder_gpus, args.max_bert_size)
                data_queue.put((context_encoding, batch))
                pbar.update(1)
            else:
                time.sleep(random.random())
        pbar.close()
        return


class ConsumerThread(threading.Thread):
    def __init__(self, data_loader, model):
        super(ConsumerThread, self).__init__()
        self.model = model
        self.Loader = data_loader

    def run(self):
        time.sleep(5)
        while not data_queue.empty() or not self.Loader.empty():
            if not data_queue.empty():
                context_encoding, batch = data_queue.get()
                batch = dispatch(context_encoding, batch['context_mask'], batch=batch, device=model_gpu)
                train_batch(self.model, batch)
                del context_encoding, batch
            else:
                time.sleep(random.random())
        time.sleep(10)
        predict(self.model, eval_dataset, dev_example_dict, dev_feature_dict,
                join(args.prediction_path, 'pred_epoch_{}.json'.format(epc)))
        torch.save(self.model.state_dict(), join(args.checkpoint_path, "ckpt_epoch_{}.pth".format(epc)))
        lock.release()


def large_batch_encode(bert_model, batch, encoder_gpus, max_bert_bsz):
    doc_ids, doc_mask, segment_ids = batch['context_idxs'], batch['context_mask'], batch['segment_idxs']
    N = doc_ids.shape[0]

    doc_ids = doc_ids.cuda(encoder_gpus[0])
    doc_mask = doc_mask.cuda(encoder_gpus[0])
    segment_ids = segment_ids.cuda(encoder_gpus[0])
    doc_encoding = []

    ptr = 0
    while ptr < N:
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


def train_batch(model, batch):
    global global_step, total_train_loss

    start, end, sp, Type, softmask, ent, yp1, yp2 = model(batch, return_yp=True)
    loss_list = compute_loss(batch, start, end, sp, Type, softmask)
    loss_list[0].backward()

    if (global_step + 1) % args.grad_accumulate_step == 0:
        optimizer.step()
        optimizer.zero_grad()

    global_step += 1

    for i, l in enumerate(loss_list):
        if not isinstance(l, int):
            total_train_loss[i] += l.item()

    if global_step % VERBOSE_STEP == 0:
        print("{} -- In Epoch{}: ".format(args.name, epc))
        for i, l in enumerate(total_train_loss):
            print("Avg-LOSS{}/batch/step: {}".format(i, l / VERBOSE_STEP))
        total_train_loss = [0] * 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = set_config()

    # Allocate Models on GPU
    encoder_gpus = [int(i) for i in args.encoder_gpu.split(',')]
    model_gpu = 'cuda:{}'.format(args.model_gpu)

    encoder = BertModel.from_pretrained(args.bert_model)
    encoder.cuda(encoder_gpus[0])
    encoder = torch.nn.DataParallel(encoder, device_ids=encoder_gpus)
    encoder.eval()

    helper = DataHelper(gz=True, config=args)
    args.n_type = helper.n_type

    # Set datasets
    Full_Loader = helper.train_loader
    Subset_Loader = helper.train_sub_loader
    dev_example_dict = helper.dev_example_dict
    dev_feature_dict = helper.dev_feature_dict
    eval_dataset = helper.dev_loader

    # Set Model
    model = GraphFusionNet(config=args)
    model.cuda(model_gpu)
    model.train()

    # Initialize optimizer and criterions
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(size_average=True, ignore_index=IGNORE_INDEX)
    binary_criterion = nn.BCEWithLogitsLoss(size_average=True)

    # Training
    global_step = epc = 0
    total_train_loss = [0] * 5
    test_loss_record = []
    lock = threading.Lock()
    VERBOSE_STEP = args.verbose_step
    while True:
        lock.acquire()
        if epc == args.qat_epochs + args.epochs:
            exit(0)
        epc += 1

        # learning rate decay
        if epc > 1:
            lr = lr * args.decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('lr = {}'.format(lr))

        # Early Stopping
        if epc > args.early_stop_epoch + 1:
            if test_loss_record[-1] > test_loss_record[-(1 + args.early_stop_epoch)]:
                print("Early Stop in epoch{}".format(epc))
                for i, test_loss in enumerate(test_loss_record):
                    print(i, test_loss_record)
                exit(0)

        if epc <= args.qat_epochs:
            Loader = Subset_Loader
        else:
            Loader = Full_Loader
        Loader.refresh()

        producer = ProducerThread(Loader, encoder)
        consumer = ConsumerThread(Loader, model)
        producer.start()
        consumer.start()

