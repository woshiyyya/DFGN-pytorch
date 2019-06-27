from model.layers import *


class GraphFusionNet(nn.Module):
    """
    Packing Query Version
    """
    def __init__(self, config):
        super(GraphFusionNet, self).__init__()
        self.config = config
        self.n_layers = config.n_layers
        self.max_query_length = 50

        self.bi_attention = BiAttention(input_dim=config.input_dim,
                                        memory_dim=config.input_dim,
                                        hid_dim=config.hidden_dim,
                                        dropout=config.bi_attn_drop)
        self.bi_attn_linear = nn.Linear(config.hidden_dim * 4, config.hidden_dim)

        h_dim = config.hidden_dim
        q_dim = config.hidden_dim if config.q_update else config.input_dim

        self.basicblocks = nn.ModuleList()
        self.query_update_layers = nn.ModuleList()
        self.query_update_linears = nn.ModuleList()

        for layer in range(self.n_layers):
            self.basicblocks.append(BasicBlock(h_dim, q_dim, layer, config))
            if config.q_update:
                self.query_update_layers.append(BiAttention(h_dim, h_dim, h_dim, config.bi_attn_drop))
                self.query_update_linears.append(nn.Linear(h_dim * 4, h_dim))

        q_dim = h_dim if config.q_update else config.input_dim
        if config.prediction_trans:
            self.predict_layer = TransformerPredictionLayer(self.config, q_dim)
        else:
            self.predict_layer = PredictionLayer(self.config, q_dim)

    def forward(self, batch, return_yp, debug=False):
        query_mapping = batch['query_mapping']
        entity_mask = batch['entity_mask']
        context_encoding = batch['context_encoding']

        # extract query encoding
        trunc_query_mapping = query_mapping[:, :self.max_query_length].contiguous()
        trunc_query_state = (context_encoding * query_mapping.unsqueeze(2))[:, :self.max_query_length, :].contiguous()
        # bert encoding query vec
        query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        attn_output, trunc_query_state = self.bi_attention(context_encoding, trunc_query_state, trunc_query_mapping)
        input_state = self.bi_attn_linear(attn_output)

        if self.config.q_update:
            query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        softmasks = []
        entity_state = None
        for l in range(self.n_layers):
            input_state, entity_state, softmask = self.basicblocks[l](input_state, query_vec, batch)
            softmasks.append(softmask)
            if self.config.q_update:
                query_attn_output, _ = self.query_update_layers[l](trunc_query_state, entity_state, entity_mask)
                trunc_query_state = self.query_update_linears[l](query_attn_output)
                query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        predictions = self.predict_layer(batch, input_state, query_vec, entity_state, query_mapping, return_yp)
        start, end, sp, Type, ent, yp1, yp2 = predictions

        if return_yp:
            return start, end, sp, Type, softmasks, ent, yp1, yp2
        else:
            return start, end, sp, Type, softmasks, ent


