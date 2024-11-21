import torch as th
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .torch_gcn import GCN
from .torch_gat import GAT

class BertClassifier(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit


class BertGCN(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, n_hidden=200, dropout=0.5, device='cpu'):
        super(BertGCN, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.device = th.device(device)

        # Tokenizer dan model BERT
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features

        # Linear classifier
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)

        # Graph Convolutional Network (GCN)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers - 1,
            activation=F.elu,
            dropout=dropout
        )

    def forward(self, g, idx):
        # Debugging data awal
        print(f"Forward pass: idx shape={idx.shape}, idx dtype={idx.dtype}")
        print(f"Graph data input IDs shape: {g.input_ids.shape}, attention mask shape: {g.attention_mask.shape}")
        print(f"Graph data cls_feats shape: {g.cls_feats.shape if g.cls_feats is not None else 'None'}, device: {g.cls_feats.device if g.cls_feats is not None else 'None'}")

        # CLS features
        if self.training:
            cls_feats = self.bert_model(g.input_ids[idx], g.attention_mask[idx])[0][:, 0]
            g.cls_feats[idx] = cls_feats
        else:
            if g.cls_feats is None:
                raise ValueError("g.cls_feats is None. Ensure it is properly initialized during training.")
            cls_feats = g.cls_feats[idx]

        # Debugging CLS features
        print(f"CLS features shape: {cls_feats.shape}, device: {cls_feats.device}")

        # Classifier logits
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)

        # Debugging Classifier logits
        print(f"Classifier logit shape: {cls_logit.shape}, classifier pred shape: {cls_pred.shape}")

        # Memastikan g.cls_feats dan g.edge_weight ada dan dipindahkan ke perangkat yang benar
        if g.cls_feats is None or g.edge_weight is None:
            print("g.cls_feats")
            print(g.cls_feats)
            
            print("g.edge_weight")
            print(g.edge_weight)
            raise ValueError("g.cls_feats or g.edge_weight is None. Ensure they are correctly initialized.")

        # g.cls_feats = g.cls_feats.to(self.device)
        # g.edge_weight = g.edge_weight.to(self.device)

        # Debugging GCN input
        print(f"GCN input feats shape: {g.cls_feats.shape}, edge weight shape: {g.edge_weight.shape}, edge weight device: {g.edge_weight.device}")

        # GCN logits
        gcn_logit = self.gcn(g.cls_feats, g, g.edge_weight)

        # Debugging GCN logits
        print(f"GCN logit shape: {gcn_logit.shape}")

        # Kombinasi CLS dan GCN logits
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit[idx])
        pred = (gcn_pred + 1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)

        # Debugging prediksi akhir
        print(f"Predictions shape: {pred.shape}")
        return pred


class BertGAT(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, heads=8, n_hidden=32, dropout=0.5):
        super(BertGAT, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GAT(
                 num_layers=gcn_layers-1,
                 in_dim=self.feat_dim,
                 num_hidden=n_hidden,
                 num_classes=nb_class,
                 heads=[heads] * (gcn_layers-1) + [1],
                 activation=F.elu,
                 feat_drop=dropout,
                 attn_drop=dropout,
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.input_ids[idx], g.attention_mask[idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.cls_feats[idx] = cls_feats
        else:
            cls_feats = g.cls_feats[idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.cls_feats, g)[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred
