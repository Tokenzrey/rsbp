# BertGCN
This repo contains code for [BertGCN: Transductive Text Classification by Combining GCN and BERT](https://arxiv.org/abs/2105.05727).

## Dependencies
1. **Download CUDA 12.1** (Wajib)
   - Unduh dan instal CUDA Toolkit 12.1 dari [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive).
   - Pastikan Anda telah menambahkan path CUDA (`bin` dan `lib`) ke variabel lingkungan `PATH`.

2. **Clone Repository**
   - Clone repository proyek ini menggunakan perintah berikut:
     ```bash
     git clone <URL_REPOSITORY>
     ```
   - Fetch branch terbaru dan switch ke branch `switch_to_pip`:
     ```bash
     git fetch origin
     git checkout switch_to_pip
     ```

3. **Download Python 3.11**
   - Unduh dan instal **Python 3.11** dari [Python.org](https://www.python.org/downloads/).

4. **Buat Virtual Environment**
   - Pergi ke direktori proyek dan buat environment virtual menggunakan Python 3.11:
     ```bash
     "path\ke\python3.11" -m venv myvenv
     ```
   - Misalnya:
     ```bash
     "C:\Python311\python.exe" -m venv myvenv
     ```

5. **Aktifkan Virtual Environment**
   - Aktifkan virtual environment yang baru saja dibuat:
     - **Windows**:
       ```bash
       myvenv\Scripts\activate
       ```
     - **Linux/macOS**:
       ```bash
       source myvenv/bin/activate
       ```

6. **Instal Dependensi dari `requirements.txt`**
   - Buka file `requirements.txt` dan instal setiap paket **satu per satu** dengan menggunakan perintah berikut:
     ```bash
     pip install <nama_paket>
     ```
   - Contoh:
     ```bash
     pip install torch==2.0.0+cu121
     pip install transformers==4.34.0
     pip install dgl==2.1.0+cu121
     pip install torch-ignite==0.4.11
     pip install numpy>=1.22
     pip install scikit-learn>=1.0
     pip install nltk>=3.6
     ```

7. **Verifikasi Instalasi**
   - Pastikan semua pustaka telah terpasang dengan benar dengan menjalankan:
     ```bash
     pip list
     ```
   - Pastikan CUDA terdeteksi oleh PyTorch:
     ```python
     python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
     ```

## Usage

1. Run `python build_graph.py [dataset]` to build the text graph.

2. Run `python finetune_bert.py --dataset [dataset]` 
to finetune the BERT model over target dataset. The model and training logs will be saved to `checkpoint/[bert_init]_[dataset]/` by default. 
Run `python finetune_bert.py -h` to see the full list of hyperparameters.

3. Run `python train_bert_gcn.py --dataset [dataset] --pretrained_bert_ckpt [pretrained_bert_ckpt] -m [m]`
to train the BertGCN. 
`[m]` is the factor balancing BERT and GCN prediction \(lambda in the paper\). 
The model and training logs will be saved to `checkpoint/[bert_init]_[gcn_model]_[dataset]/` by default. 
Run `python train_bert_gcn.py -h` to see the full list of hyperparameters.

Trained BertGCN parameters can be downloaded [here](https://drive.google.com/file/d/1YUl7q34S3pu8KH17yOI68tvcedkrQ39a).

## Acknowledgement

The data preprocess and graph construction are from [TextGCN](https://github.com/yao8839836/text_gcn)