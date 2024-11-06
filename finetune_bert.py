import torch as th
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from utils import *
import dgl
import torch.utils.data as Data
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import accuracy_score
import argparse, shutil, logging
from torch.optim import lr_scheduler
from model import BertClassifier

# Membuat parser untuk menerima argumen dari command line, yang memungkinkan pengguna 
# menentukan parameter training saat menjalankan skrip ini
parser = argparse.ArgumentParser()

# Menambahkan argumen untuk menentukan panjang input maksimum untuk BERT
parser.add_argument('--max_length', type=int, default=128, help='panjang input maksimum untuk BERT')

# Menambahkan argumen untuk ukuran batch saat training
parser.add_argument('--batch_size', type=int, default=128)

# Menambahkan argumen untuk jumlah epoch (siklus pelatihan)
parser.add_argument('--nb_epochs', type=int, default=60)

# Menambahkan argumen untuk learning rate BERT
parser.add_argument('--bert_lr', type=float, default=1e-4)

# Menambahkan argumen untuk menentukan dataset yang digunakan
parser.add_argument('--dataset', default='20ng', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr'])

# Menambahkan argumen untuk memilih model BERT yang akan digunakan
parser.add_argument('--bert_init', type=str, default='roberta-base',
                    choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])

# Menambahkan argumen untuk direktori penyimpanan checkpoint, jika tidak ditentukan,
# maka akan menggunakan nama [bert_init]_[dataset] sebagai direktori default
parser.add_argument('--checkpoint_dir', default=None, help='direktori checkpoint, [bert_init]_[dataset] jika tidak ditentukan')

# Memparsing argumen yang diterima
args = parser.parse_args()

# Menyimpan nilai argumen ke variabel untuk memudahkan akses di bagian kode berikutnya
max_length = args.max_length
batch_size = args.batch_size
nb_epochs = args.nb_epochs
bert_lr = args.bert_lr
dataset = args.dataset
bert_init = args.bert_init
checkpoint_dir = args.checkpoint_dir

# Menentukan direktori checkpoint berdasarkan argumen `checkpoint_dir` atau membangunnya dari nama model dan dataset
if checkpoint_dir is None:
    ckpt_dir = './checkpoint/{}_{}'.format(bert_init, dataset)
else:
    ckpt_dir = checkpoint_dir

# Membuat direktori checkpoint jika belum ada
os.makedirs(ckpt_dir, exist_ok=True)

# Menyalin file skrip saat ini ke direktori checkpoint untuk referensi konfigurasi pelatihan
shutil.copy(os.path.basename(__file__), ckpt_dir)

# Konfigurasi sistem logging

# Menyiapkan handler untuk menampilkan log di layar (console)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(message)s'))  # Mengatur format log tanpa tambahan informasi
sh.setLevel(logging.INFO)  # Menentukan level logging INFO untuk StreamHandler

# Menyiapkan handler untuk menyimpan log ke file 'training.log' di direktori checkpoint
fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))  # Mengatur format log
fh.setLevel(logging.INFO)  # Menentukan level logging INFO untuk FileHandler

# Membuat logger utama untuk pelatihan, dan menambahkan kedua handler (console dan file) ke logger ini
logger = logging.getLogger('training logger')
logger.addHandler(sh)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

# Menentukan perangkat yang digunakan untuk pelatihan: CPU atau GPU (jika tersedia)
cpu = th.device('cpu')
gpu = th.device('cuda:0')

# Menampilkan argumen yang digunakan dan direktori checkpoint
logger.info('arguments:')
logger.info(str(args))
logger.info('checkpoints akan disimpan di {}'.format(ckpt_dir))

# Data Preprocess: Memuat dataset dan mempersiapkan data untuk pelatihan

# Memuat adjacency matrix, fitur dokumen, label, dan mask untuk data training, validasi, dan testing
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
'''
y_train, y_val, y_test: matriks n*c, di mana n adalah jumlah node dan c adalah jumlah kelas
train_mask, val_mask, test_mask: array boolean n-dimensi yang menunjukkan apakah node adalah bagian dari train/val/test
train_size, test_size: variabel yang tidak digunakan (disediakan untuk informasi tambahan)
'''

# Menghitung jumlah node untuk setiap bagian data dan jumlah kelas
nb_node = adj.shape[0]  # Total jumlah node dalam graf (dokumen dan kata)
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()  # Jumlah node untuk train, val, dan test
nb_word = nb_node - nb_train - nb_val - nb_test  # Jumlah node yang berisi kata, bukan dokumen
nb_class = y_train.shape[1]  # Jumlah kelas dari label training

# Menginstansiasi model BERT dengan jumlah kelas yang sesuai
model = BertClassifier(pretrained_model=bert_init, nb_class=nb_class)

# Mengonversi label one-hot menjadi ID kelas untuk kompatibilitas dengan PyTorch
y = th.LongTensor((y_train + y_val + y_test).argmax(axis=1))  # Menyimpan kelas dengan ID numerik dari argmax
label = {}  # Dictionary untuk menyimpan label
label['train'], label['val'], label['test'] = y[:nb_train], y[nb_train:nb_train+nb_val], y[-nb_test:]

# Memuat teks dokumen dan menghapus karakter backslash, kemudian memisahkannya per baris (dokumen)
corpus_file = './data/corpus/' + dataset + '_shuffle.txt'
with open(corpus_file, 'r') as f:
    text = f.read()
    text = text.replace('\\', '')  # Menghapus karakter backslash
    text = text.split('\n')  # Memisahkan setiap dokumen berdasarkan baris baru

# Fungsi untuk mengonversi teks menjadi encoding input yang dapat digunakan oleh BERT
def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=max_length, truncation=True, padding=True, return_tensors='pt')
    return input.input_ids, input.attention_mask

# Inisialisasi dictionary untuk input_ids dan attention_mask yang akan digunakan untuk setiap subset data
input_ids, attention_mask = {}, {}

# Melakukan encoding pada seluruh teks dokumen
input_ids_, attention_mask_ = encode_input(text, model.tokenizer)

# Membagi input encoding ke dalam set train, val, dan test sesuai dengan ukuran masing-masing subset
input_ids['train'], input_ids['val'], input_ids['test'] = input_ids_[:nb_train], input_ids_[nb_train:nb_train+nb_val], input_ids_[-nb_test:]
attention_mask['train'], attention_mask['val'], attention_mask['test'] = attention_mask_[:nb_train], attention_mask_[nb_train:nb_train+nb_val], attention_mask_[-nb_test:]

# Membuat DataLoader untuk train, val, dan test menggunakan TensorDataset
datasets = {}  # Menyimpan dataset per split (train, val, test)
loader = {}    # Menyimpan DataLoader untuk tiap subset data
for split in ['train', 'val', 'test']:
    # Membuat TensorDataset dari input_ids, attention_mask, dan label per split
    datasets[split] = Data.TensorDataset(input_ids[split], attention_mask[split], label[split])
    # Membuat DataLoader dengan ukuran batch yang ditentukan dan opsi shuffle=True untuk melatih model
    loader[split] = Data.DataLoader(datasets[split], batch_size=batch_size, shuffle=True)

# Menginisialisasi optimizer untuk mengoptimalkan parameter model menggunakan algoritma Adam
optimizer = th.optim.Adam(model.parameters(), lr=bert_lr)

# Scheduler untuk mengatur learning rate secara dinamis
# Mengurangi learning rate dengan faktor 0.1 setelah epoch ke-30
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

# Fungsi untuk satu langkah training pada satu batch data
def train_step(engine, batch):
    global model, optimizer
    # Menetapkan mode model ke training
    model.train()
    # Memindahkan model ke GPU untuk mempercepat komputasi
    model = model.to(gpu)
    
    # Mengatur gradien optimizer ke nol sebelum backpropagation
    optimizer.zero_grad()
    
    # Memindahkan data batch (input_ids, attention_mask, label) ke GPU
    (input_ids, attention_mask, label) = [x.to(gpu) for x in batch]
    
    # Melakukan prediksi dengan model
    y_pred = model(input_ids, attention_mask)
    
    # Mengonversi label ke tipe data long yang diperlukan oleh PyTorch
    y_true = label.type(th.long)
    
    # Menghitung loss menggunakan cross-entropy, loss yang umum untuk klasifikasi multi-kelas
    loss = F.cross_entropy(y_pred, y_true)
    
    # Melakukan backpropagation untuk menghitung gradien dari loss
    loss.backward()
    
    # Memperbarui parameter model berdasarkan gradien yang telah dihitung
    optimizer.step()
    
    # Mengambil nilai loss untuk logging
    train_loss = loss.item()
    
    # Menghitung akurasi training
    with th.no_grad():
        # Memindahkan label dan prediksi ke CPU dan menghitung akurasi
        y_true = y_true.detach().cpu()
        y_pred = y_pred.argmax(axis=1).detach().cpu()  # Mendapatkan prediksi kelas
        train_acc = accuracy_score(y_true, y_pred)
    
    # Mengembalikan loss dan akurasi untuk logging
    return train_loss, train_acc

# Membuat engine untuk training dengan Ignite, menggunakan train_step sebagai proses training
trainer = Engine(train_step)

# Fungsi untuk satu langkah evaluasi (testing) pada satu batch data
def test_step(engine, batch):
    global model
    # Mengatur mode model ke evaluasi untuk mematikan dropout, dll.
    with th.no_grad():
        model.eval()
        model = model.to(gpu)  # Memindahkan model ke GPU untuk komputasi
        
        # Memindahkan data batch (input_ids, attention_mask, label) ke GPU
        (input_ids, attention_mask, label) = [x.to(gpu) for x in batch]
        
        # Melakukan prediksi dengan model tanpa mengatur gradien
        y_pred = model(input_ids, attention_mask)
        y_true = label  # Label sebenarnya (ground truth) untuk perhitungan akurasi
        
        # Mengembalikan prediksi dan label untuk perhitungan metrik pada evaluasi
        return y_pred, y_true

# Membuat evaluator dengan Engine Ignite yang menjalankan fungsi `test_step` untuk evaluasi
evaluator = Engine(test_step)

# Mendefinisikan metrik yang akan dievaluasi
# Menggunakan akurasi dan cross-entropy loss sebagai metrik evaluasi
metrics = {
    'acc': Accuracy(),  # Mengukur akurasi prediksi
    'nll': Loss(th.nn.CrossEntropyLoss())  # Mengukur loss negatif log-likelihood (cross-entropy)
}

# Melampirkan setiap metrik ke evaluator sehingga metrik dihitung selama evaluasi
for n, f in metrics.items():
    f.attach(evaluator, n)

# Fungsi callback untuk mencatat hasil setelah setiap epoch selesai
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    # Menjalankan evaluasi pada set training dan mencatat metrik
    evaluator.run(loader['train'])
    metrics = evaluator.state.metrics
    train_acc, train_nll = metrics["acc"], metrics["nll"]  # Mengambil akurasi dan loss training
    
    # Menjalankan evaluasi pada set validasi dan mencatat metrik
    evaluator.run(loader['val'])
    metrics = evaluator.state.metrics
    val_acc, val_nll = metrics["acc"], metrics["nll"]  # Mengambil akurasi dan loss validasi
    
    # Menjalankan evaluasi pada set testing dan mencatat metrik
    evaluator.run(loader['test'])
    metrics = evaluator.state.metrics
    test_acc, test_nll = metrics["acc"], metrics["nll"]  # Mengambil akurasi dan loss testing
    
    # Logging hasil akurasi dan loss untuk train, val, dan test set pada epoch tersebut
    logger.info(
        "\rEpoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f}  Test acc: {:.4f} loss: {:.4f}"
        .format(trainer.state.epoch, train_acc, train_nll, val_acc, val_nll, test_acc, test_nll)
    )

    # Menyimpan checkpoint jika akurasi validasi meningkat
    if val_acc > log_training_results.best_val_acc:
        logger.info("New checkpoint")
        th.save(
            {
                'bert_model': model.bert_model.state_dict(),  # Menyimpan parameter BERT
                'classifier': model.classifier.state_dict(),  # Menyimpan parameter classifier
                'optimizer': optimizer.state_dict(),  # Menyimpan status optimizer
                'epoch': trainer.state.epoch,  # Menyimpan nomor epoch
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'  # Menyimpan ke file checkpoint
            )
        )
        log_training_results.best_val_acc = val_acc  # Memperbarui akurasi validasi terbaik

    # Menjalankan scheduler untuk mengubah learning rate jika diperlukan
    scheduler.step()

# Inisialisasi akurasi validasi terbaik ke 0 sebelum pelatihan dimulai
log_training_results.best_val_acc = 0

# Memulai pelatihan dengan menjalankan trainer untuk `nb_epochs` epoch
trainer.run(loader['train'], max_epochs=nb_epochs)
