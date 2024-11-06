import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine

if len(sys.argv) != 2:
	sys.exit("Use: python build_graph.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
# build corpus
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")

# Read Word Vectors
# word_vector_file = 'data/glove.6B/glove.6B.300d.txt'
# word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
#_, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])

# Dimensi embedding untuk setiap kata
word_embeddings_dim = 300

# Dictionary untuk menyimpan vektor embedding kata
word_vector_map = {}

# Inisialisasi daftar untuk nama dokumen dan daftar dokumen untuk training dan testing
doc_name_list = []
doc_train_list = []
doc_test_list = []

# Membuka file dataset yang berisi daftar dokumen
f = open('data/' + dataset + '.txt', 'r')
lines = f.readlines()

# Memisahkan dokumen berdasarkan nama dan kategori (train/test)
for line in lines:
    # Menambahkan setiap baris dokumen ke dalam doc_name_list setelah menghapus karakter tak diperlukan
    doc_name_list.append(line.strip())
    temp = line.split("\t")
    # Memasukkan dokumen ke dalam daftar test jika ada tanda 'test' pada kategorinya
    if temp[1].find('test') != -1:
        doc_test_list.append(line.strip())
    # Memasukkan dokumen ke dalam daftar train jika ada tanda 'train' pada kategorinya
    elif temp[1].find('train') != -1:
        doc_train_list.append(line.strip())
f.close()

# Menyiapkan daftar untuk menampung isi konten dokumen yang telah dibersihkan
doc_content_list = []
f = open('data/corpus/' + dataset + '.clean.txt', 'r')
lines = f.readlines()

# Mengisi doc_content_list dengan isi setiap dokumen
for line in lines:
    doc_content_list.append(line.strip())
f.close()

# Membuat daftar indeks untuk dokumen training
train_ids = []
for train_name in doc_train_list:
    # Mendapatkan indeks dari setiap dokumen training berdasarkan nama
    train_id = doc_name_list.index(train_name)
    # Menambahkan indeks ke daftar train_ids
    train_ids.append(train_id)
print(train_ids)

# Mengacak urutan indeks dokumen training untuk memperbaiki distribusi data saat training
random.shuffle(train_ids)

# (Opsional) Mengambil sebagian data training untuk labeling parsial
#train_ids = train_ids[:int(0.2 * len(train_ids))]

# Menggabungkan semua indeks training menjadi satu string dengan pemisah baris baru
train_ids_str = '\n'.join(str(index) for index in train_ids)

# Menyimpan daftar indeks training ke dalam file
f = open('data/' + dataset + '.train.index', 'w')
f.write(train_ids_str)
f.close()

# Membuat daftar indeks untuk dokumen testing
test_ids = []
for test_name in doc_test_list:
    # Mendapatkan indeks dari setiap dokumen testing berdasarkan nama
    test_id = doc_name_list.index(test_name)
    # Menambahkan indeks ke daftar test_ids
    test_ids.append(test_id)
print(test_ids)

# Mengacak urutan indeks dokumen testing
random.shuffle(test_ids)

# Menggabungkan semua indeks testing menjadi satu string dengan pemisah baris baru
test_ids_str = '\n'.join(str(index) for index in test_ids)

# Menyimpan daftar indeks testing ke dalam file
f = open('data/' + dataset + '.test.index', 'w')
f.write(test_ids_str)
f.close()

# Menggabungkan daftar indeks dokumen training dan testing
ids = train_ids + test_ids
print(ids)
print(len(ids))

# Membuat daftar nama dan konten dokumen setelah diacak
shuffle_doc_name_list = []
shuffle_doc_words_list = []

# Mengisi daftar nama dan konten dokumen dengan indeks yang telah diacak
for id in ids:
    shuffle_doc_name_list.append(doc_name_list[int(id)])
    shuffle_doc_words_list.append(doc_content_list[int(id)])

# Menggabungkan nama dokumen yang diacak menjadi satu string
shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)

# Menggabungkan isi dokumen yang diacak menjadi satu string
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

# Menyimpan daftar nama dokumen yang diacak ke dalam file
f = open('data/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_name_str)
f.close()

# Menyimpan daftar isi dokumen yang diacak ke dalam file
f = open('data/corpus/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_words_str)
f.close()

# Membangun Vocabulary (Kosa Kata)

# Dictionary untuk menyimpan frekuensi kemunculan setiap kata di seluruh dokumen
word_freq = {}

# Himpunan untuk menyimpan semua kata unik dalam dokumen
word_set = set()

# Mengisi word_set dan word_freq dengan kata-kata dari seluruh dokumen
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()  # Memisahkan kata dalam dokumen
    for word in words:
        # Menambahkan kata ke dalam himpunan kata unik
        word_set.add(word)
        # Menambahkan frekuensi kata ke word_freq, atau menginisialisasi jika kata belum ada
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

# Mengonversi word_set menjadi daftar vocab dan menghitung ukurannya
vocab = list(word_set)
vocab_size = len(vocab)

# Membuat daftar dokumen untuk setiap kata yang muncul
word_doc_list = {}

# Looping untuk menghubungkan setiap kata dengan dokumen tempat kata tersebut muncul
for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    appeared = set()  # Himpunan untuk melacak kata-kata yang sudah dihitung dalam dokumen ini
    for word in words:
        # Hanya proses jika kata belum muncul dalam dokumen ini
        if word in appeared:
            continue
        # Jika kata sudah ada dalam word_doc_list, tambahkan indeks dokumen
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        # Jika kata belum ada, buat entri baru dalam word_doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)  # Menandai kata sebagai sudah muncul dalam dokumen ini

# Dictionary untuk menyimpan frekuensi dokumen untuk setiap kata
word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    # Menyimpan jumlah dokumen yang mengandung kata sebagai word_doc_freq
    word_doc_freq[word] = len(doc_list)

# Membuat peta kata ke indeks unik, untuk penggunaan lebih lanjut
word_id_map = {}
for i in range(vocab_size):
    # Menyimpan setiap kata dalam vocab dengan indeks uniknya
    word_id_map[vocab[i]] = i

# Mengonversi vocabulary menjadi string, dengan setiap kata dipisahkan oleh baris baru
vocab_str = '\n'.join(vocab)

# Menyimpan vocabulary ke file teks untuk referensi
f = open('data/corpus/' + dataset + '_vocab.txt', 'w')
f.write(vocab_str)
f.close()

'''
Word definitions begin
'''
'''
definitions = []

for word in vocab:
    word = word.strip()
    synsets = wn.synsets(clean_str(word))
    word_defs = []
    for synset in synsets:
        syn_def = synset.definition()
        word_defs.append(syn_def)
    word_des = ' '.join(word_defs)
    if word_des == '':
        word_des = '<PAD>'
    definitions.append(word_des)

string = '\n'.join(definitions)


f = open('data/corpus/' + dataset + '_vocab_def.txt', 'w')
f.write(string)
f.close()

tfidf_vec = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vec.fit_transform(definitions)
tfidf_matrix_array = tfidf_matrix.toarray()
print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))

word_vectors = []

for i in range(len(vocab)):
    word = vocab[i]
    vector = tfidf_matrix_array[i]
    str_vector = []
    for j in range(len(vector)):
        str_vector.append(str(vector[j]))
    temp = ' '.join(str_vector)
    word_vector = word + ' ' + temp
    word_vectors.append(word_vector)

string = '\n'.join(word_vectors)

f = open('data/corpus/' + dataset + '_word_vectors.txt', 'w')
f.write(string)
f.close()

word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
_, embd, word_vector_map = loadWord2Vec(word_vector_file)
word_embeddings_dim = len(embd[0])
'''

'''
Word definitions end
'''

# Membuat Daftar Label Unik

# Himpunan untuk menyimpan label unik dari dokumen
label_set = set()

# Mengumpulkan label dari setiap dokumen dalam daftar yang telah diacak
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')  # Memisahkan metadata dokumen berdasarkan tab
    label_set.add(temp[2])       # Menambahkan label (kolom ke-3) ke dalam himpunan

# Mengonversi himpunan label menjadi daftar dan mengurutkannya
label_list = list(label_set)

# Mengonversi daftar label menjadi string dengan setiap label pada baris baru
label_list_str = '\n'.join(label_list)

# Menyimpan daftar label ke file teks untuk referensi
f = open('data/corpus/' + dataset + '_labels.txt', 'w')
f.write(label_list_str)
f.close()

# Matriks Fitur untuk Dokumen Training
# Mendefinisikan ukuran training set dan validation set
train_size = len(train_ids)         # Total jumlah dokumen dalam training set
val_size = int(0.1 * train_size)    # Menentukan ukuran validation set sebagai 10% dari training set
real_train_size = train_size - val_size  # Jumlah dokumen training aktual setelah memisahkan validation set

# Mengambil nama dokumen yang termasuk dalam real training set
real_train_doc_names = shuffle_doc_name_list[:real_train_size]
real_train_doc_names_str = '\n'.join(real_train_doc_names)  # Menggabungkan nama dokumen dengan baris baru

# Menyimpan nama dokumen real training set ke file teks
f = open('data/' + dataset + '.real_train.name', 'w')
f.write(real_train_doc_names_str)
f.close()

# Membuat Matriks Sparse untuk Fitur Dokumen Training

# Inisialisasi list untuk membangun matriks sparse
row_x = []
col_x = []
data_x = []

# Mengisi matriks fitur `x` dengan vektor embedding kata yang diambil dari setiap dokumen
for i in range(real_train_size):
    # Inisialisasi vektor dokumen sebagai nol dengan ukuran word_embeddings_dim
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)  # Panjang dokumen dalam jumlah kata

    # Menambahkan embedding kata ke dalam vektor dokumen
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)  # Menjumlahkan embedding kata-kata dalam dokumen

    # Mengisi row, col, dan data dengan rata-rata embedding kata sebagai fitur dokumen
    for j in range(word_embeddings_dim):
        row_x.append(i)               # Indeks baris
        col_x.append(j)               # Indeks kolom
        data_x.append(doc_vec[j] / doc_len)  # Nilai data: rata-rata embedding setiap kata

# Membuat matriks sparse CSR dari data yang dikumpulkan
x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(real_train_size, word_embeddings_dim))

# Membuat Label Fitur untuk Dokumen Training

# Inisialisasi list untuk menyimpan label dalam bentuk one-hot
y = []
for i in range(real_train_size):
    doc_meta = shuffle_doc_name_list[i]  # Metadata dokumen yang diambil dari daftar nama dokumen yang diacak
    temp = doc_meta.split('\t')
    label = temp[2]                     # Mendapatkan label dari metadata dokumen

    # Membuat vektor one-hot untuk label dokumen ini
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)  # Mendapatkan indeks label dalam label_list
    one_hot[label_index] = 1               # Menandai indeks label dalam vektor one-hot
    y.append(one_hot)                      # Menambahkan vektor one-hot ke list y

# Mengonversi y menjadi array numpy untuk efisiensi
y = np.array(y)
print(y)

# Matriks Fitur untuk Dokumen Testing (tx) - Tanpa Fitur Awal

# Menentukan ukuran testing set berdasarkan jumlah dokumen dalam test_ids
test_size = len(test_ids)

# Inisialisasi list untuk membangun matriks sparse untuk fitur dokumen testing
row_tx = []
col_tx = []
data_tx = []

# Mengisi matriks fitur `tx` untuk setiap dokumen testing dengan rata-rata embedding kata
for i in range(test_size):
    # Inisialisasi vektor dokumen sebagai nol untuk mengakumulasi embedding kata
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    # Mengambil kata-kata dari dokumen ke-(i + train_size) agar dimulai setelah dokumen training
    doc_words = shuffle_doc_words_list[i + train_size]
    words = doc_words.split()
    doc_len = len(words)  # Menghitung panjang dokumen dalam jumlah kata

    # Menambahkan embedding kata-kata dalam dokumen ke vektor dokumen
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)  # Mengakumulasi embedding kata

    # Mengisi row, col, dan data dengan rata-rata embedding sebagai fitur dokumen
    for j in range(word_embeddings_dim):
        row_tx.append(i)                  # Indeks baris
        col_tx.append(j)                  # Indeks kolom
        data_tx.append(doc_vec[j] / doc_len)  # Rata-rata embedding setiap kata

# Membuat matriks sparse CSR `tx` dari data yang telah dikumpulkan
tx = sp.csr_matrix((data_tx, (row_tx, col_tx)), shape=(test_size, word_embeddings_dim))

# Membuat Label Fitur untuk Dokumen Testing (ty)

# Inisialisasi list untuk menyimpan label one-hot untuk setiap dokumen testing
ty = []
for i in range(test_size):
    # Mengambil metadata dokumen dari daftar yang diacak, dimulai setelah dokumen training
    doc_meta = shuffle_doc_name_list[i + train_size]
    temp = doc_meta.split('\t')
    label = temp[2]                       # Mendapatkan label dokumen dari metadata

    # Membuat vektor one-hot untuk label dokumen ini
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)  # Mendapatkan indeks label dalam label_list
    one_hot[label_index] = 1               # Menandai indeks label dalam vektor one-hot
    ty.append(one_hot)                     # Menambahkan vektor one-hot ke list ty

# Mengonversi ty menjadi array numpy untuk efisiensi dan kompatibilitas
ty = np.array(ty)
print(ty)

# allx: the the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> words

# Membuat Vektor Kata untuk Vocabulary

# Menginisialisasi vektor embedding acak untuk setiap kata dalam vocabulary
word_vectors = np.random.uniform(-0.01, 0.01, (vocab_size, word_embeddings_dim))

# Memperbarui word_vectors dengan embedding kata yang ada di word_vector_map
for i in range(len(vocab)):
    word = vocab[i]  # Mengambil kata dari vocabulary
    # Jika kata ada di word_vector_map, ambil embedding-nya
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector  # Memperbarui vektor kata dengan embedding yang tersedia

# Membuat Matriks Sparse `allx` untuk Menggabungkan Fitur Dokumen Berlabel dan Kata dalam Vocabulary

# Inisialisasi list untuk membuat matriks sparse `allx`
row_allx = []
col_allx = []
data_allx = []

# Menambahkan fitur dokumen training ke `allx` dengan rata-rata embedding kata
for i in range(train_size):
    # Inisialisasi vektor dokumen sebagai nol untuk mengakumulasi embedding kata
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]  # Mengambil kata-kata dalam dokumen training
    words = doc_words.split()
    doc_len = len(words)  # Menghitung panjang dokumen

    # Menambahkan embedding kata ke vektor dokumen
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    # Mengisi row_allx, col_allx, dan data_allx dengan rata-rata embedding kata sebagai fitur dokumen
    for j in range(word_embeddings_dim):
        row_allx.append(int(i))               # Indeks baris untuk dokumen ke-i
        col_allx.append(j)                    # Indeks kolom untuk setiap dimensi embedding
        data_allx.append(doc_vec[j] / doc_len)  # Rata-rata embedding setiap kata

# Menambahkan embedding kata dalam vocabulary ke `allx`
for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))       # Indeks baris untuk kata ke-i setelah dokumen
        col_allx.append(j)                         # Indeks kolom untuk setiap dimensi embedding
        data_allx.append(word_vectors.item((i, j)))  # Nilai embedding kata

# Mengonversi row_allx, col_allx, dan data_allx menjadi array numpy
row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

# Membuat matriks sparse CSR `allx` dengan ukuran (train_size + vocab_size, word_embeddings_dim)
allx = sp.csr_matrix((data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

# Membuat Label `ally` untuk Dokumen Berlabel dan Kata dalam Vocabulary

# Membuat vektor label `ally` untuk dokumen training
ally = []
for i in range(train_size):
    doc_meta = shuffle_doc_name_list[i]  # Mengambil metadata dokumen
    temp = doc_meta.split('\t')
    label = temp[2]  # Mendapatkan label dokumen
    # Membuat vektor one-hot untuk label
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1  # Menandai indeks label yang sesuai
    ally.append(one_hot)  # Menambahkan vektor one-hot ke ally

# Menambahkan label kosong untuk setiap kata dalam vocabulary (karena kata tidak memiliki label)
for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]  # Membuat vektor nol untuk kata
    ally.append(one_hot)  # Menambahkan vektor nol ke ally

# Mengonversi ally menjadi array numpy untuk efisiensi
ally = np.array(ally)

# Menampilkan bentuk dari berbagai matriks fitur dan label untuk verifikasi
print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

'''
Doc word heterogeneous graph
'''

# Menghitung kemunculan bersama kata-kata dengan jendela konteks
window_size = 20
windows = []

# Membagi dokumen menjadi jendela konteks berukuran window_size
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()  # Memisahkan kata-kata dalam dokumen
    length = len(words)
    # Jika dokumen lebih pendek dari window_size, ambil seluruh kata sebagai satu jendela
    if length <= window_size:
        windows.append(words)
    else:
        # Membuat jendela yang bergerak sepanjang dokumen, dengan panjang window_size
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)  # Menambahkan jendela ke dalam daftar

# Menghitung frekuensi kata dalam setiap jendela konteks
word_window_freq = {}
for window in windows:
    appeared = set()  # Menghindari penghitungan kata yang sama lebih dari sekali per jendela
    for i in range(len(window)):
        # Hanya hitung jika kata belum muncul dalam jendela ini
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1  # Menambah frekuensi jika kata sudah ada
        else:
            word_window_freq[window[i]] = 1  # Inisialisasi frekuensi jika kata baru muncul
        appeared.add(window[i])

# Menghitung jumlah kemunculan bersama untuk setiap pasangan kata dalam jendela
word_pair_count = {}
for window in windows:
    # Loop melalui pasangan kata dalam jendela
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            # Menghitung frekuensi pasangan kata dalam satu arah
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # Menghitung frekuensi pasangan dalam arah sebaliknya
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

# Inisialisasi list untuk membangun matriks sparse
row = []
col = []
weight = []

# Menggunakan PMI sebagai bobot untuk graph
num_window = len(windows)

# Menghitung PMI untuk setiap pasangan kata dan menambahkannya ke matriks adjacency
for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])  # Indeks kata pertama
    j = int(temp[1])  # Indeks kata kedua
    count = word_pair_count[key]  # Jumlah kemunculan bersama pasangan kata
    word_freq_i = word_window_freq[vocab[i]]  # Frekuensi kata pertama
    word_freq_j = word_window_freq[vocab[j]]  # Frekuensi kata kedua

    # Menghitung Pointwise Mutual Information (PMI)
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
    
    # Hanya tambahkan ke graph jika PMI > 0
    if pmi <= 0:
        continue
    # Menambahkan baris, kolom, dan bobot ke matriks sparse
    row.append(train_size + i)  # Offset oleh train_size untuk memasukkan dokumen
    col.append(train_size + j)
    weight.append(pmi)  # Menambahkan nilai PMI sebagai bobot

# word vector cosine similarity as weights

'''
for i in range(vocab_size):
    for j in range(vocab_size):
        if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
            vector_i = np.array(word_vector_map[vocab[i]])
            vector_j = np.array(word_vector_map[vocab[j]])
            similarity = 1.0 - cosine(vector_i, vector_j)
            if similarity > 0.9:
                print(vocab[i], vocab[j], similarity)
                row.append(train_size + i)
                col.append(train_size + j)
                weight.append(similarity)
'''
# Menghitung Frekuensi Kata per Dokumen

# Dictionary untuk menyimpan frekuensi kemunculan kata dalam setiap dokumen
doc_word_freq = {}

# Loop melalui setiap dokumen untuk menghitung frekuensi kata
for doc_id in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]  # Mendapatkan ID unik kata
        doc_word_str = str(doc_id) + ',' + str(word_id)  # Membuat kunci dokumen dan kata
        # Menambah frekuensi kata di dokumen ini atau menginisialisasi jika baru muncul
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

# Membuat Matriks Sparse untuk Hubungan Dokumen-Kata Berdasarkan TF-IDF

# Loop untuk setiap dokumen dan kata untuk membangun adjacency matrix
for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_word_set = set()  # Menyimpan kata unik yang muncul dalam dokumen ini
    for word in words:
        # Hanya hitung kata sekali per dokumen
        if word in doc_word_set:
            continue
        j = word_id_map[word]  # ID unik kata
        key = str(i) + ',' + str(j)  # Kunci untuk dokumen dan kata
        freq = doc_word_freq[key]  # Frekuensi kata dalam dokumen ini
        # Menentukan baris indeks: dokumen training atau testing
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        # Kolom untuk kata dalam vocabulary
        col.append(train_size + j)
        # Menghitung Inverse Document Frequency (IDF)
        idf = log(1.0 * len(shuffle_doc_words_list) / word_doc_freq[vocab[j]])
        # Menyimpan bobot hasil perkalian TF dan IDF untuk adjacency matrix
        weight.append(freq * idf)
        doc_word_set.add(word)  # Menandai kata sebagai sudah muncul dalam dokumen ini

# Menentukan ukuran node total dalam graph: semua dokumen dan semua kata
node_size = train_size + vocab_size + test_size

# Membuat matriks sparse CSR untuk adjacency matrix yang menyimpan bobot TF-IDF
adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))

# Menyimpan Objek yang Dibuat ke dalam File dengan Pickle

# Menyimpan matriks `x` (fitur dokumen training)
f = open("data/ind.{}.x".format(dataset), 'wb')
pkl.dump(x, f)
f.close()

# Menyimpan label `y` (label dokumen training)
f = open("data/ind.{}.y".format(dataset), 'wb')
pkl.dump(y, f)
f.close()

# Menyimpan matriks `tx` (fitur dokumen testing)
f = open("data/ind.{}.tx".format(dataset), 'wb')
pkl.dump(tx, f)
f.close()

# Menyimpan label `ty` (label dokumen testing)
f = open("data/ind.{}.ty".format(dataset), 'wb')
pkl.dump(ty, f)
f.close()

# Menyimpan `allx` (fitur gabungan dokumen berlabel dan kata dalam vocabulary)
f = open("data/ind.{}.allx".format(dataset), 'wb')
pkl.dump(allx, f)
f.close()

# Menyimpan `ally` (label gabungan dokumen berlabel dan kata dalam vocabulary)
f = open("data/ind.{}.ally".format(dataset), 'wb')
pkl.dump(ally, f)
f.close()

# Menyimpan adjacency matrix `adj` yang berisi hubungan dokumen-kata
f = open("data/ind.{}.adj".format(dataset), 'wb')
pkl.dump(adj, f)
f.close()