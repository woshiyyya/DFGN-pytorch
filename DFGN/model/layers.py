import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import rnn
from torch.autograd import Variable
from utils import get_weights, get_act
from pytorch_pretrained_bert.modeling import BertLayer


def tok_to_ent(tok2ent):
    if tok2ent == 'mean':
        return MeanPooling
    elif tok2ent == 'mean_max':
        return MeanMaxPooling
    else:
        raise NotImplementedError


def mean_pooling(input, mask):
    mean_pooled = input.sum(dim=1) / mask.sum(dim=1, keepdim=True)
    return mean_pooled


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, doc_state, entity_mapping, entity_lens):
        entity_states = entity_mapping.unsqueeze(3) * doc_state.unsqueeze(1)  # N x E x L x d
        mean_pooled = torch.sum(entity_states, dim=2) / entity_lens.unsqueeze(2)
        return mean_pooled


class MeanMaxPooling(nn.Module):
    def __init__(self):
        super(MeanMaxPooling, self).__init__()

    def forward(self, doc_state, entity_mapping, entity_lens):
        """
        :param doc_state:  N x L x d
        :param entity_mapping:  N x E x L
        :param entity_lens:  N x E
        :return: N x E x 2d
        """
        entity_states = entity_mapping.unsqueeze(3) * doc_state.unsqueeze(1)  # N x E x L x d
        max_pooled = torch.max(entity_states, dim=2)[0]
        mean_pooled = torch.sum(entity_states, dim=2) / entity_lens.unsqueeze(2)
        output = torch.cat([max_pooled, mean_pooled], dim=2)  # N x E x 2d
        return output


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """
    Sinusoid position encoding table
    """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class GATSelfAttention(nn.Module):
    def __init__(self, in_dim, out_dim, config, layer_id=0, head_id=0):
        """ One head GAT """
        super(GATSelfAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = config.gnn_drop
        self.q_attn = config.q_attn
        self.query_dim = in_dim
        self.n_type = config.n_type

        # Case study
        self.layer_id = layer_id
        self.head_id = head_id
        self.step = 0

        self.W_type = nn.ParameterList()
        self.a_type = nn.ParameterList()
        self.qattn_W1 = nn.ParameterList()
        self.qattn_W2 = nn.ParameterList()
        for i in range(self.n_type):
            self.W_type.append(get_weights((in_dim, out_dim)))
            self.a_type.append(get_weights((out_dim * 2, 1)))

            if config.q_attn:
                q_dim = config.hidden_dim if config.q_update else config.input_dim
                self.qattn_W1.append(get_weights((q_dim, out_dim * 2)))
                self.qattn_W2.append(get_weights((out_dim * 2, out_dim * 2)))

        self.act = get_act('lrelu:0.2')

    def forward(self, input_state, adj, entity_mask, adj_mask=None, query_vec=None):
        zero_vec = torch.zeros_like(adj)
        scores = 0

        for i in range(self.n_type):
            h = torch.matmul(input_state, self.W_type[i])
            h = F.dropout(h, self.dropout, self.training)
            N, E, d = h.shape

            a_input = torch.cat([h.repeat(1, 1, E).view(N, E * E, -1), h.repeat(1, E, 1)], dim=-1)
            a_input = a_input.view(-1, E, E, 2*d)

            if self.q_attn:
                q_gate = F.relu(torch.matmul(query_vec, self.qattn_W1[i]))
                q_gate = torch.sigmoid(torch.matmul(q_gate, self.qattn_W2[i]))
                a_input = a_input * q_gate[:, None, None, :]
                score = self.act(torch.matmul(a_input, self.a_type[i]).squeeze(3))
            else:
                score = self.act(torch.matmul(a_input, self.a_type[i]).squeeze(3))
            scores += torch.where(adj == i+1, score, zero_vec)

        zero_vec = -9e15 * torch.ones_like(scores)
        scores = torch.where(adj > 0, scores, zero_vec)

        # Ahead Alloc
        if adj_mask is not None:
            h = h * adj_mask

        coefs = F.softmax(scores, dim=2)  # N * E * E
        h = coefs.unsqueeze(3) * h.unsqueeze(2)  # N * E * E * d
        h = torch.sum(h, dim=1)
        return h


class AttentionLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, n_head, config, layer_id=0):
        super(AttentionLayer, self).__init__()
        assert hid_dim % n_head == 0
        self.dropout = config.gnn_drop

        self.attn_funcs = nn.ModuleList()
        for i in range(n_head):
            self.attn_funcs.append(
                GATSelfAttention(in_dim=in_dim, out_dim=hid_dim // n_head, config=config, layer_id=layer_id, head_id=i))

        if in_dim != hid_dim:
            self.align_dim = nn.Linear(in_dim, hid_dim)
            nn.init.xavier_uniform_(self.align_dim.weight, gain=1.414)
        else:
            self.align_dim = lambda x: x

    def forward(self, input, adj, entity_mask, adj_mask=None, query_vec=None):
        hidden_list = []
        for attn in self.attn_funcs:
            h = attn(input, adj, entity_mask, adj_mask=adj_mask, query_vec=query_vec)
            hidden_list.append(h)

        h = torch.cat(hidden_list, dim=-1)
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(h)
        return h


class InteractionLayer(nn.Module):
    def __init__(self, input_dim, out_dim, config):
        super(InteractionLayer, self).__init__()
        self.config = config
        self.use_trans = config.basicblock_trans

        if config.basicblock_trans:
            bert_config = BertConfig(input_dim, config.trans_heads, config.trans_drop)
            self.transformer = BertLayer(bert_config)
            self.transformer_linear = nn.Linear(input_dim, out_dim)
        else:
            self.lstm = LSTMWrapper(input_dim, out_dim // 2, 1)

    def forward(self, doc_state, entity_state, doc_length, entity_mapping, entity_length, context_mask):
        """
        :param doc_state: N x L x dc
        :param entity_state: N x E x de
        :param entity_mapping: N x E x L
        :return: doc_state: N x L x out_dim, entity_state: N x L x out_dim (x2)
        """
        expand_entity_state = torch.sum(entity_state.unsqueeze(2) * entity_mapping.unsqueeze(3), dim=1)  # N x E x L x d
        input_state = torch.cat([expand_entity_state, doc_state], dim=2)

        if self.use_trans:
            extended_attention_mask = context_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

            doc_state = self.transformer(input_state, extended_attention_mask)
            doc_state = self.transformer_linear(doc_state)
        else:
            doc_state = self.lstm(input_state, doc_length)

        return doc_state


class BasicBlock(nn.Module):
    def __init__(self, hidden_dim, q_dim, layer, config):
        super(BasicBlock, self).__init__()
        self.config = config
        self.layer = layer
        self.gnn_type = config.gnn.split(':')[0]
        if config.tok2ent == 'mean_max':
            input_dim = hidden_dim * 2
        else:
            input_dim = hidden_dim
        self.tok2ent = tok_to_ent(config.tok2ent)()
        self.query_weight = get_weights((q_dim, input_dim))
        self.temp = np.sqrt(q_dim * input_dim)
        self.gat = AttentionLayer(input_dim, hidden_dim, config.n_heads, config, layer_id=layer)
        self.int_layer = InteractionLayer(hidden_dim * 2, hidden_dim, config)

    def forward(self, doc_state, query_vec, batch):
        context_mask = batch['context_mask']
        entity_mapping = batch['entity_mapping']
        entity_length = batch['entity_lens']
        entity_mask = batch['entity_mask']
        doc_length = batch['context_lens']
        adj = batch['entity_graphs']

        entity_state = self.tok2ent(doc_state, entity_mapping, entity_length)

        query = torch.matmul(query_vec, self.query_weight)
        query_scores = torch.bmm(entity_state, query.unsqueeze(2)) / self.temp
        softmask = query_scores * entity_mask.unsqueeze(2)  # N x E x 1  BCELossWithLogits
        adj_mask = torch.sigmoid(softmask)

        entity_state = self.gat(entity_state, adj, entity_mask, adj_mask=adj_mask, query_vec=query_vec)
        doc_state = self.int_layer(doc_state, entity_state, doc_length, entity_mapping, entity_length, context_mask)
        return doc_state, entity_state, softmask


class BiAttention(nn.Module):
    def __init__(self, input_dim, memory_dim, hid_dim, dropout):
        super(BiAttention, self).__init__()
        self.dropout = dropout
        self.input_linear_1 = nn.Linear(input_dim, 1, bias=False)
        self.memory_linear_1 = nn.Linear(memory_dim, 1, bias=False)

        self.input_linear_2 = nn.Linear(input_dim, hid_dim, bias=True)
        self.memory_linear_2 = nn.Linear(memory_dim, hid_dim, bias=True)

        self.dot_scale = np.sqrt(input_dim)

    def forward(self, input, memory, mask):
        """
        :param input: context_encoding N * Ld * d
        :param memory: query_encoding N * Lm * d
        :param mask: query_mask N * Lm
        :return:
        """
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = F.dropout(input, self.dropout, training=self.training)  # N x Ld x d
        memory = F.dropout(memory, self.dropout, training=self.training)  # N x Lm x d

        input_dot = self.input_linear_1(input)  # N x Ld x 1
        memory_dot = self.memory_linear_1(memory).view(bsz, 1, memory_len)  # N x 1 x Lm
        # N * Ld * Lm
        cross_dot = torch.bmm(input, memory.permute(0, 2, 1).contiguous()) / self.dot_scale
        # [f1, f2]^T [w1, w2] + <f1 * w3, f2>
        # (N * Ld * 1) + (N * 1 * Lm) + (N * Ld * Lm)
        att = input_dot + memory_dot + cross_dot  # N x Ld x Lm
        # N * Ld * Lm
        att = att - 1e30 * (1 - mask[:, None])

        input = self.input_linear_2(input)
        memory = self.memory_linear_2(memory)

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1), memory


class LSTMWrapper(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layer, concat=False, bidir=True, dropout=0.3, return_last=True):
        super(LSTMWrapper, self).__init__()
        self.rnns = nn.ModuleList()
        for i in range(n_layer):
            if i == 0:
                input_dim_ = input_dim
                output_dim_ = hidden_dim
            else:
                input_dim_ = hidden_dim if not bidir else hidden_dim * 2
                output_dim_ = hidden_dim
            self.rnns.append(nn.LSTM(input_dim_, output_dim_, 1, bidirectional=bidir, batch_first=True))
        self.dropout = dropout
        self.concat = concat
        self.n_layer = n_layer
        self.return_last = return_last

    def forward(self, input, input_lengths=None):
        # input_length must be in decreasing order
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []

        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()

        for i in range(self.n_layer):
            output = F.dropout(output, p=self.dropout, training=self.training)

            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)

            output, _ = self.rnns[i](output)

            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:  # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)

            outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]


class PredictionLayer(nn.Module):
    """
    Identical to baseline prediction layer
    """
    def __init__(self, config, q_dim):
        super(PredictionLayer, self).__init__()
        self.config = config
        input_dim = config.hidden_dim
        h_dim = config.hidden_dim

        self.hidden = h_dim

        # Cascade Network
        self.entity_linear_0 = nn.Linear(h_dim + q_dim, h_dim)
        self.entity_linear_1 = nn.Linear(h_dim, 1)

        self.sp_lstm = LSTMWrapper(input_dim=input_dim, hidden_dim=h_dim, n_layer=1, dropout=config.lstm_drop)
        self.sp_linear = nn.Linear(h_dim * 2, 1)

        self.start_lstm = LSTMWrapper(input_dim=input_dim + 1, hidden_dim=h_dim, n_layer=1, dropout=config.lstm_drop)
        self.start_linear = nn.Linear(h_dim * 2, 1)

        self.end_lstm = LSTMWrapper(input_dim=input_dim + 2*h_dim + 1, hidden_dim=h_dim, n_layer=1, dropout=config.lstm_drop)
        self.end_linear = nn.Linear(h_dim * 2, 1)

        self.type_lstm = LSTMWrapper(input_dim=input_dim + 2*h_dim + 1, hidden_dim=h_dim, n_layer=1, dropout=config.lstm_drop)
        self.type_linear = nn.Linear(h_dim * 2, 3)

        self.cache_S = 0
        self.cache_mask = None

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch, context_input, query_vec, entity_state, packing_mask=None, return_yp=False):
        """
        :param batch
        :param context_input:   [N x L x hid]
        :param query_vec:       [N x q_dim]
        :param entity_state:    [N x E x hid]
        :param entity_mask:     [N x E]
        :param context_mask:    [N x L]
        :param context_lens:    [N]
        :param start_mapping:   [N x max_sent x L]      000100000000
        :param end_mapping:     [N x max_sent x L]      000000001000
        :param all_mapping:     [N x L x max_sent]      000111111000
        :param packing_mask:    [N x L] or None
        :param return_yp
        :return:
        """
        context_mask = batch['context_mask']
        entity_mask = batch['entity_mask']
        context_lens = batch['context_lens']
        start_mapping = batch['start_mapping']
        end_mapping = batch['end_mapping']
        all_mapping = batch['all_mapping']

        entity_prediction = None
        if entity_state is not None:
            expand_query = query_vec.unsqueeze(1).repeat((1, entity_state.shape[1], 1))
            entity_logits = self.entity_linear_0(torch.cat([entity_state, expand_query], dim=2))
            entity_logits = self.entity_linear_1(F.relu(entity_logits))
            entity_prediction = entity_logits.squeeze(2) - 1e30 * (1 - entity_mask)

        sp_output = self.sp_lstm(context_input, context_lens)  # N x L x 2d
        start_output = torch.bmm(start_mapping, sp_output[:, :, self.hidden:])   # N x max_sent x d
        end_output = torch.bmm(end_mapping, sp_output[:, :, :self.hidden])       # N x max_sent x d
        sp_logits = torch.cat([start_output, end_output], dim=-1)  # N x max_sent x 2d

        sp_logits = self.sp_linear(sp_logits)  # N x max_sent x 1
        sp_logits_aux = Variable(sp_logits.data.new(sp_logits.size(0), sp_logits.size(1), 1).zero_())
        sp_prediction = torch.cat([sp_logits_aux, sp_logits], dim=-1).contiguous()

        sp_forward = torch.bmm(all_mapping, sp_logits).contiguous()  # N x L x 1

        start_input = torch.cat([context_input, sp_forward], dim=-1)
        start_output = self.start_lstm(start_input, context_lens)
        start_prediction = self.start_linear(start_output).squeeze(2) - 1e30 * (1 - context_mask)  # N x L

        end_input = torch.cat([context_input, start_output, sp_forward], dim=-1)
        end_output = self.end_lstm(end_input, context_lens)
        end_prediction = self.end_linear(end_output).squeeze(2) - 1e30 * (1 - context_mask)  # N x L

        type_input = torch.cat([context_input, end_output, sp_forward], dim=-1)
        type_output = torch.max(self.type_lstm(type_input, context_lens), dim=1)[0]
        type_logits = type_output
        type_prediction = self.type_linear(type_logits)

        if not return_yp:
            return start_prediction, end_prediction, sp_prediction, type_prediction, entity_prediction

        outer = start_prediction[:, :, None] + end_prediction[:, None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if packing_mask is not None:
            outer = outer - 1e30 * packing_mask[:, :, None]
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]
        return start_prediction, end_prediction, sp_prediction, type_prediction, entity_prediction, yp1, yp2


class BertConfig(object):
    def __init__(self, hidden_dim, n_heads, dropout):
        self.hidden_size = hidden_dim
        self.intermediate_size = hidden_dim
        self.num_attention_heads = n_heads
        self.hidden_dropout_prob = dropout
        self.attention_probs_dropout_prob = dropout
        self.hidden_act = "gelu"


class PositionalEncoder(nn.Module):
    def __init__(self, h_dim, config):
        super(PositionalEncoder, self).__init__()
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(config.max_doc_len, h_dim, padding_idx=0),
            freeze=True)
        self.device = 'cuda:{}'.format(config.model_gpu)
        self.fixed_position_vec = torch.LongTensor(list(range(config.max_doc_len))).cuda(self.device)

    def forward(self, context_mapping):
        """
        :param context_mapping: N x L
        :return: position_encoding: N x L x d
        """
        N, L = context_mapping.shape
        trunc_position_vec = self.fixed_position_vec[:L].contiguous()
        context_position = trunc_position_vec.unsqueeze(0) * context_mapping
        position_encoding = self.position_enc(context_position)
        return position_encoding


class TransformerPredictionLayer(nn.Module):
    def __init__(self, config, q_dim):
        super(TransformerPredictionLayer, self).__init__()
        self.config = config
        h_dim = config.hidden_dim

        self.hidden = h_dim

        self.position_encoder = PositionalEncoder(h_dim, config)

        # Cascade Network
        bert_config = BertConfig(config.hidden_dim, config.trans_heads, config.trans_drop)

        self.sp_transformer = BertLayer(bert_config)
        self.sp_linear = nn.Linear(h_dim * 2, 1)

        self.start_input_linear = nn.Linear(h_dim + 1, h_dim)
        self.start_transformer = BertLayer(bert_config)
        self.start_linear = nn.Linear(h_dim, 1)

        self.end_input_linear = nn.Linear(2 * h_dim + 1, h_dim)
        self.end_transformer = BertLayer(bert_config)
        self.end_linear = nn.Linear(h_dim, 1)

        self.type_input_linear = nn.Linear(2 * h_dim + 1, h_dim)
        self.type_transformer = BertLayer(bert_config)
        self.type_linear = nn.Linear(h_dim, 3)

        self.cache_S = 0
        self.cache_mask = None

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch, context_input, query_vec, entity_state, packing_mask=None, return_yp=False):
        """
        :param context_input:   [N x L x hid]
        :param query_vec:       [N x q_dim]
        :param entity_state:    [N x E x hid]
        :param entity_mask:     [N x E]
        :param context_mask:    [N x L]
        :param context_lens:    [N]
        :param start_mapping:   [N x max_sent x L]      000100000000
        :param end_mapping:     [N x max_sent x L]      000000001000
        :param all_mapping:     [N x L x max_sent]      000111111000
        :param packing_mask     [N x L]
        :param return_yp:       bool
        :return:
        """
        context_mask = batch['context_mask']
        start_mapping = batch['start_mapping']
        end_mapping = batch['end_mapping']
        all_mapping = batch['all_mapping']

        position_encoding = self.position_encoder(context_mask.long())

        extended_attention_mask = context_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        entity_prediction = None

        context_input = context_input + position_encoding
        sp_output = self.sp_transformer(context_input, extended_attention_mask)  # N x L x d
        sp_start_output = torch.bmm(start_mapping, sp_output)   # N x max_sent x d
        sp_end_output = torch.bmm(end_mapping, sp_output)       # N x max_sent x d
        sp_logits = torch.cat([sp_start_output, sp_end_output], dim=-1)  # N x max_sent x 2d

        sp_logits = self.sp_linear(sp_logits)  # N x max_sent x 1
        # sp_prediction = sp_logits.squeeze()
        sp_logits_aux = Variable(sp_logits.data.new(sp_logits.size(0), sp_logits.size(1), 1).zero_())
        sp_prediction = torch.cat([sp_logits_aux, sp_logits], dim=-1).contiguous()
        sp_forward = torch.bmm(all_mapping, sp_logits).contiguous()  # N x L x 1

        start_input = torch.cat([context_input, sp_forward], dim=-1)
        start_input = self.start_input_linear(start_input) + position_encoding
        start_output = self.start_transformer(start_input, extended_attention_mask)
        start_prediction = self.start_linear(start_output).squeeze(2) - 1e30 * (1 - context_mask)  # N x L

        end_input = torch.cat([context_input, start_output, sp_forward], dim=-1)
        end_input = self.end_input_linear(end_input) + position_encoding
        end_output = self.end_transformer(end_input, extended_attention_mask)
        end_prediction = self.end_linear(end_output).squeeze(2) - 1e30 * (1 - context_mask)  # N x L

        type_input = torch.cat([context_input, end_output, sp_forward], dim=-1)
        type_input = self.type_input_linear(type_input) + position_encoding
        type_output = torch.max(self.type_transformer(type_input, extended_attention_mask), dim=1)[0]
        type_logits = type_output
        type_prediction = self.type_linear(type_logits)

        if not return_yp:
            return start_prediction, end_prediction, sp_prediction, type_prediction, entity_prediction

        outer = start_prediction[:, :, None] + end_prediction[:, None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if packing_mask is not None:
            outer = outer - 1e30 * packing_mask[:, :, None]
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]
        return start_prediction, end_prediction, sp_prediction, type_prediction, entity_prediction, yp1, yp2
