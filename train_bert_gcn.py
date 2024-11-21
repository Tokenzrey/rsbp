# Mengimpor library yang diperlukan
import torch as th
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from utils import *
# import dgl
import torch.utils.data as Data
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss
from sklearn.metrics import accuracy_score
import numpy as np
import os
import shutil
import argparse
import sys
import logging
from datetime import datetime
from torch.optim import lr_scheduler
from model import BertGCN, BertGAT
from torch_geometric.data import Data as PyGData
from torch_geometric.utils import from_scipy_sparse_matrix

# Membuat parser untuk menerima parameter dari command line
parser = argparse.ArgumentParser()

# Menentukan panjang maksimum input yang akan diproses oleh BERT
parser.add_argument('--max_length', type=int, default=128, help='Panjang input maksimum untuk BERT')

# Menentukan ukuran batch yang akan digunakan dalam pelatihan
parser.add_argument('--batch_size', type=int, default=64)

# Menentukan faktor keseimbangan antara prediksi BERT dan GCN
parser.add_argument('-m', '--m', type=float, default=0.7, help='Faktor keseimbangan antara prediksi BERT dan GCN')

# Menentukan jumlah epoch untuk pelatihan
parser.add_argument('--nb_epochs', type=int, default=50)

# Menentukan jenis model BERT yang akan digunakan
parser.add_argument('--bert_init', type=str, default='roberta-base',
                    choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])

# Menentukan path checkpoint dari model BERT yang sudah dilatih sebelumnya
parser.add_argument('--pretrained_bert_ckpt', default=None)

# Menentukan dataset yang akan digunakan untuk pelatihan
parser.add_argument('--dataset', default='20ng', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr'])

# Menentukan direktori untuk menyimpan checkpoint model
parser.add_argument('--checkpoint_dir', default=None, help='Direktori checkpoint, menggunakan [bert_init]_[gcn_model]_[dataset] jika tidak ditentukan')

# Menentukan model graf yang digunakan, bisa GCN atau GAT
parser.add_argument('--gcn_model', type=str, default='gcn', choices=['gcn', 'gat'])

# Menentukan jumlah layer GCN yang akan digunakan
parser.add_argument('--gcn_layers', type=int, default=2)

# Menentukan dimensi layer tersembunyi pada GCN; untuk GAT adalah n_hidden * heads
parser.add_argument('--n_hidden', type=int, default=200, help='Dimensi layer tersembunyi pada GCN, untuk GAT adalah n_hidden * heads')

# Menentukan jumlah perhatian (attention heads) pada GAT
parser.add_argument('--heads', type=int, default=8, help='Jumlah attention heads untuk GAT')

# Menentukan dropout rate selama pelatihan
parser.add_argument('--dropout', type=float, default=0.5)

# Menentukan learning rate untuk GCN
parser.add_argument('--gcn_lr', type=float, default=1e-3)

# Menentukan learning rate untuk BERT
parser.add_argument('--bert_lr', type=float, default=1e-5)

# Memparsing argumen dari command line
args = parser.parse_args()

# Menyimpan argumen yang diparsing ke dalam variabel
max_length = args.max_length
batch_size = args.batch_size
m = args.m
nb_epochs = args.nb_epochs
bert_init = args.bert_init
pretrained_bert_ckpt = args.pretrained_bert_ckpt
dataset = args.dataset
checkpoint_dir = args.checkpoint_dir
gcn_model = args.gcn_model
gcn_layers = args.gcn_layers
n_hidden = args.n_hidden
heads = args.heads
dropout = args.dropout
gcn_lr = args.gcn_lr
bert_lr = args.bert_lr

# Menentukan direktori checkpoint secara otomatis jika tidak ditentukan
if checkpoint_dir is None:
    ckpt_dir = './checkpoint/{}_{}_{}'.format(bert_init, gcn_model, dataset)
else:
    ckpt_dir = checkpoint_dir

# Membuat direktori checkpoint jika belum ada
os.makedirs(ckpt_dir, exist_ok=True)

# Menyalin file skrip ini ke direktori checkpoint untuk referensi konfigurasi pelatihan
shutil.copy(os.path.basename(__file__), ckpt_dir)

# Menyiapkan handler untuk menampilkan log di console
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(message)s'))
sh.setLevel(logging.INFO)

# Menyiapkan handler untuk menyimpan log ke file 'training.log' di direktori checkpoint
fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
fh.setLevel(logging.INFO)

# Membuat logger utama untuk pelatihan, menambahkan handler untuk console dan file
logger = logging.getLogger('training logger')
logger.addHandler(sh)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

# Menentukan perangkat yang digunakan untuk pelatihan: CPU atau GPU (jika tersedia)
cpu = th.device('cpu')
gpu = th.device('cuda:0') if th.cuda.is_available() else th.device('cpu')

# Menampilkan argumen yang digunakan dan direktori checkpoint ke dalam log
logger.info('arguments:')
logger.info(str(args))
logger.info('checkpoints akan disimpan di {}'.format(ckpt_dir))

# Model
# Memuat data korpus, termasuk adjacency matrix dan label, untuk diproses lebih lanjut
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
'''
adj: adjacency matrix berbentuk sparse (jarang) dengan ukuran n*n, menunjukkan hubungan antar node
y_train, y_val, y_test: matriks one-hot dengan ukuran n*c, di mana n adalah jumlah node dan c adalah jumlah kelas
train_mask, val_mask, test_mask: array boolean berdimensi n, menunjukkan node mana yang termasuk ke dalam masing-masing set
'''

# Menghitung jumlah node dokumen yang sebenarnya untuk tiap subset data dan jumlah kelas dalam dataset
nb_node = features.shape[0]          # Jumlah total node dalam graph
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()  # Jumlah node untuk tiap subset
nb_word = nb_node - nb_train - nb_val - nb_test  # Menghitung node kata dalam graf (bukan dokumen)
nb_class = y_train.shape[1]           # Jumlah kelas, diambil dari dimensi kolom matriks one-hot label

# Jika menggunakan GCN
model = BertGCN(nb_class=nb_class, pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                    n_hidden=n_hidden, dropout=dropout)

# Memuat checkpoint BERT yang sudah di-finetune sebelumnya jika tersedia
if pretrained_bert_ckpt is not None:
    ckpt = th.load(pretrained_bert_ckpt, map_location=gpu)  # Memuat checkpoint ke GPU
    model.bert_model.load_state_dict(ckpt['bert_model'])     # Mengisi parameter bert_model
    model.classifier.load_state_dict(ckpt['classifier'])     # Mengisi parameter classifier

# Memuat teks dokumen dari file corpus dan membersihkan teks dari karakter tertentu
corpse_file = './data/corpus/' + dataset + '_shuffle.txt'
with open(corpse_file, 'r') as f:
    text = f.read()               # Membaca seluruh teks dalam file
    text = text.replace('\\', '')  # Menghapus karakter backslash
    text = text.split('\n')        # Memisahkan teks menjadi daftar per baris dokumen

# Fungsi untuk mengonversi teks dokumen menjadi encoding yang dapat digunakan sebagai input BERT
def encode_input(text, tokenizer):
    # Menggunakan tokenizer BERT untuk membuat input ID dan attention mask
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
    return input.input_ids, input.attention_mask

# Mengonversi dokumen menjadi input BERT (input_ids dan attention_mask)
input_ids, attention_mask = encode_input(text, model.tokenizer)

# Menyisipkan padding nol untuk node kata (non-dokumen) di tengah dokumen training dan testing
input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids[-nb_test:]])
attention_mask = th.cat([attention_mask[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask[-nb_test:]])

# Mengonversi label one-hot ke format ID kelas untuk digunakan di PyTorch
y = y_train + y_test + y_val         # Menggabungkan semua label untuk training, validation, dan test
y_train = y_train.argmax(axis=1)     # Mengambil ID kelas untuk label training
y = y.argmax(axis=1)                 # Mengambil ID kelas untuk semua label gabungan

# Mask dokumen yang digunakan untuk pembaruan fitur node dokumen
doc_mask = train_mask + val_mask + test_mask  # Menggabungkan mask untuk node dokumen dari train, val, dan test set

# Membangun graf DGL dari adjacency matrix
# Normalisasi adjacency matrix dengan menambahkan self-loop
adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))

# Convert adjacency matrix to a PyG graph data object
edge_index, edge_weight = from_scipy_sparse_matrix(adj_norm.astype('float32'))
# Membuat objek graf PyTorch Geometric
graph_data = PyGData(edge_index=edge_index, edge_attr=edge_weight)

print("edge_weight")
print(edge_weight)
# Add features to graph data
graph_data.input_ids = input_ids
graph_data.attention_mask = attention_mask
graph_data.label = th.LongTensor(y)
graph_data.train_mask = th.BoolTensor(train_mask)
graph_data.val_mask = th.BoolTensor(val_mask)
graph_data.test_mask = th.BoolTensor(test_mask)
graph_data.label_train = th.LongTensor(y_train)
graph_data.cls_feats = th.zeros((graph_data.num_nodes, model.feat_dim))
graph_data.edge_weight = graph_data.edge_attr

logger.info(f'Jumlah node: {graph_data.num_nodes}')
logger.info(f'Jumlah edge: {graph_data.num_edges}')
if graph_data.num_nodes == 0 or graph_data.num_edges == 0:
    raise ValueError("Graph data is empty. Please check adjacency matrix and features.")

# Menampilkan informasi tentang graf yang telah dibuat
logger.info('graph information (PyTorch Geometric):')
logger.info(str(graph_data))

# Membuat index loader untuk membagi node-node dokumen ke dalam subset training, validation, dan testing

# Membuat indeks untuk node training, validation, dan testing
train_idx = Data.TensorDataset(th.arange(0, nb_train, dtype=th.long))
val_idx = Data.TensorDataset(th.arange(nb_train, nb_train + nb_val, dtype=th.long))
test_idx = Data.TensorDataset(th.arange(nb_node-nb_test, nb_node, dtype=th.long))

# Menggabungkan semua indeks dokumen menjadi satu dataset
doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

# Membuat DataLoader untuk subset training dengan shuffle diaktifkan
idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=False)

# Membuat DataLoader untuk subset validation dan test tanpa shuffle
idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)

# Membuat DataLoader untuk seluruh dokumen (train, val, test) dengan shuffle diaktifkan
idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=False)

def update_feature():
    global model, graph_data, doc_mask
    
    print("Memulai proses update fitur node...")
    # Menggunakan batch besar dan tanpa gradien untuk mempercepat proses
    dataloader = Data.DataLoader(
        Data.TensorDataset(graph_data.input_ids[doc_mask], graph_data.attention_mask[doc_mask]),
        batch_size=1024
    )
    
    with th.no_grad():
        # Mengatur model ke mode evaluasi (tanpa GPU)
        model.eval()
        cls_list = []
        print("Model diatur ke mode evaluasi.")
        
        # Mengiterasi setiap batch dokumen untuk menghasilkan embedding dari BERT
        for i, batch in enumerate(dataloader):
            print(f"Memproses batch {i+1}/{len(dataloader)}...")
            input_ids, attention_mask = batch
            print(f"input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}")
            
            # Mengambil embedding dari [CLS] token di lapisan terakhir BERT untuk setiap dokumen
            output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
            print(f"Output [CLS] embedding shape: {output.shape}")
            
            cls_list.append(output)
        
        # Menggabungkan embedding [CLS] dari semua batch
        cls_feat = th.cat(cls_list, axis=0)
        print(f"Combined [CLS] features shape: {cls_feat.shape}")
    
    # Memperbarui fitur CLS untuk node dokumen yang ada di `doc_mask`
    graph_data.cls_feats[doc_mask] = cls_feat
    print("Fitur CLS diperbarui untuk semua node dokumen.")
    
    return graph_data

# Menginisialisasi optimizer untuk parameter BERT, classifier, dan GCN
# Learning rate untuk BERT dan classifier menggunakan bert_lr, sedangkan GCN menggunakan gcn_lr
optimizer = th.optim.Adam([
        {'params': model.bert_model.parameters(), 'lr': bert_lr},
        {'params': model.classifier.parameters(), 'lr': bert_lr},
        {'params': model.gcn.parameters(), 'lr': gcn_lr},
    ], lr=1e-3
)

# Scheduler untuk menurunkan learning rate setelah 30 epoch dengan faktor 0.1
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

def train_step(engine, batch):
    global model, graph_data, optimizer
    print("Memulai langkah pelatihan...")

    # Mengaktifkan mode pelatihan
    model.train()  
    optimizer.zero_grad()  # Menginisialisasi gradien menjadi nol
    print("Gradien diinisialisasi.")

    # Batch index langsung digunakan tanpa GPU
    (idx,) = batch
    print(f"Idx: {idx}, dtype: {idx.dtype}")
    print(f"Batch index: {idx}")

    # Mengambil node yang termasuk dalam subset training berdasarkan train mask
    train_mask = graph_data.train_mask[idx].type(th.BoolTensor)
    print(f"Train mask shape: {train_mask.shape}, jumlah node training: {train_mask.sum().item()}")

    print(f"Graph data input IDs shape: {graph_data.input_ids.shape}")
    print(f"Graph data attention mask shape: {graph_data.attention_mask.shape}")
    print(f"Graph data labels shape: {graph_data.label_train.shape}")
    print(f"Graph data cls_feats shape: {graph_data.cls_feats.shape}")

    
    # Memastikan tidak ada penggunaan GPU pada bagian ini
    y_pred = model(graph_data, idx)[train_mask]
    print(f"Prediksi model dihitung. Shape: {y_pred.shape}")

    # Mengambil label yang sesuai dengan node di subset training
    y_true = graph_data.label_train[idx][train_mask]
    print(f"Label sebenarnya diperoleh. Shape: {y_true.shape}")

    # Menghitung loss menggunakan negative log likelihood (NLL)
    loss = F.nll_loss(y_pred, y_true)
    print(f"Loss dihitung: {loss.item()}")

    # Backpropagation: menghitung gradien dan memperbarui parameter model
    loss.backward()
    print("Gradien dihitung.")
    optimizer.step()
    print("Parameter model diperbarui.")

    # Melepaskan fitur untuk menghemat memori
    graph_data.cls_feats.detach_()
    print("Fitur CLS dilepaskan dari computational graph.")

    # Mengambil nilai loss sebagai scalar untuk logging
    train_loss = loss.item()

    # Menghitung akurasi training tanpa menghitung gradien
    with th.no_grad():
        if train_mask.sum() > 0:  # Memastikan ada node yang termasuk dalam subset training
            y_true = y_true.detach().cpu()  # Memastikan label ada di CPU
            y_pred = y_pred.argmax(axis=1).detach().cpu()  # Memastikan prediksi ada di CPU dan mengambil kelas dengan nilai tertinggi
            train_acc = accuracy_score(y_true, y_pred)  # Menghitung akurasi
            print(f"Akurasi training dihitung: {train_acc}")
        else:
            train_acc = 1  # Jika tidak ada node training, akurasi diatur ke 1
            print("Tidak ada node untuk training. Akurasi diatur ke 1.")

    print("Langkah pelatihan selesai.\n")
    return train_loss, train_acc  # Mengembalikan loss dan akurasi untuk logging

# Membuat engine Ignite untuk menjalankan langkah pelatihan
trainer = Engine(train_step)

# Fungsi yang dipanggil setelah setiap epoch selesai
@trainer.on(Events.EPOCH_COMPLETED)
def reset_graph(trainer):
    scheduler.step()  # Menurunkan learning rate berdasarkan scheduler
    update_feature()  # Memperbarui fitur embedding untuk node dokumen
    th.cuda.empty_cache()  # Mengosongkan cache GPU untuk menghemat memori

# Fungsi untuk satu langkah pengujian (tanpa pelatihan)
def test_step(engine, batch):
    global model, graph_data
    with th.no_grad():  # Menjalankan mode evaluasi tanpa gradien
        model.eval()  # Mengaktifkan mode evaluasi

        # Tidak perlu memindahkan ke GPU, tetap gunakan batch secara langsung
        (idx,) = batch  

        # Menghitung prediksi model untuk node di batch tersebut
        y_pred = model(graph_data, idx)

        # Mengambil label sebenarnya untuk node yang diuji
        y_true = graph_data.label[idx]
        return y_pred, y_true  # Mengembalikan prediksi dan label sebenarnya

# Membuat engine Ignite untuk menjalankan langkah pengujian
evaluator = Engine(test_step)

# Mendefinisikan metrik evaluasi yang akan digunakan: akurasi dan loss
metrics = {
    'acc': Accuracy(),            # Mengukur akurasi prediksi
    'nll': Loss(th.nn.NLLLoss())  # Mengukur loss menggunakan negative log likelihood
}

# Melampirkan metrik ke evaluator sehingga setiap kali evaluator dijalankan, metrik ini akan dihitung
for n, f in metrics.items():
    f.attach(evaluator, n)

# Fungsi yang dipanggil setiap kali epoch pelatihan selesai
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    # Menjalankan evaluator pada data training untuk menghitung metrik
    evaluator.run(idx_loader_train)
    metrics = evaluator.state.metrics  # Mengambil metrik dari state evaluator
    train_acc, train_nll = metrics["acc"], metrics["nll"]  # Akurasi dan loss pada data training

    # Menjalankan evaluator pada data validasi untuk menghitung metrik
    evaluator.run(idx_loader_val)
    metrics = evaluator.state.metrics
    val_acc, val_nll = metrics["acc"], metrics["nll"]  # Akurasi dan loss pada data validasi

    # Menjalankan evaluator pada data testing untuk menghitung metrik
    evaluator.run(idx_loader_test)
    metrics = evaluator.state.metrics
    test_acc, test_nll = metrics["acc"], metrics["nll"]  # Akurasi dan loss pada data testing

    # Mencatat hasil akurasi dan loss ke logger
    logger.info(
        "Epoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f}  Test acc: {:.4f} loss: {:.4f}"
        .format(trainer.state.epoch, train_acc, train_nll, val_acc, val_nll, test_acc, test_nll)
    )

    # Menyimpan checkpoint model jika akurasi validasi meningkat
    if val_acc > log_training_results.best_val_acc:
        logger.info("New checkpoint")
        th.save(
            {
                'bert_model': model.bert_model.state_dict(),  # Menyimpan parameter model BERT
                'classifier': model.classifier.state_dict(),  # Menyimpan parameter classifier
                'gcn': model.gcn.state_dict(),                # Menyimpan parameter GCN
                'optimizer': optimizer.state_dict(),          # Menyimpan state optimizer
                'epoch': trainer.state.epoch,                 # Menyimpan nomor epoch saat ini
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'  # Path file untuk menyimpan checkpoint
            )
        )
        log_training_results.best_val_acc = val_acc  # Memperbarui akurasi validasi terbaik

# Inisialisasi akurasi validasi terbaik dengan nilai 0
log_training_results.best_val_acc = 0

# Memperbarui fitur node dokumen sebelum pelatihan dimulai
graph_data = update_feature()

# Memulai proses pelatihan dengan engine trainer
trainer.run(idx_loader, max_epochs=nb_epochs)