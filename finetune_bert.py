import torch as th
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from utils import *
import torch.utils.data as Data
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import accuracy_score
import argparse, shutil, logging, sys
from torch.optim import lr_scheduler
from model import BertClassifier
from tqdm import tqdm  # Progress bar library

# Create a parser to accept command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=128, help='maximum input length for BERT')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--nb_epochs', type=int, default=60)
parser.add_argument('--bert_lr', type=float, default=1e-4)
parser.add_argument('--dataset', default='20ng', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr'])
parser.add_argument('--bert_init', type=str, default='roberta-base', choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, defaults to [bert_init]_[dataset] if not specified')
args = parser.parse_args()

# Extract arguments
max_length = args.max_length
batch_size = args.batch_size
nb_epochs = args.nb_epochs
bert_lr = args.bert_lr
dataset = args.dataset
bert_init = args.bert_init
checkpoint_dir = args.checkpoint_dir

# Determine checkpoint directory
if checkpoint_dir is None:
    ckpt_dir = './checkpoint/{}_{}'.format(bert_init, dataset)
else:
    ckpt_dir = checkpoint_dir
os.makedirs(ckpt_dir, exist_ok=True)

# Copy the current script to the checkpoint directory for reference
shutil.copy(os.path.basename(__file__), ckpt_dir)

# Configure logging
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(message)s'))
sh.setLevel(logging.INFO)
fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
fh.setLevel(logging.INFO)

logger = logging.getLogger('training logger')
logger.addHandler(sh)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

# Define device for training
cpu = th.device('cpu')
gpu = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
logger.info('Using device: {}'.format(gpu))

# Log parsed arguments and checkpoint directory
logger.info('Arguments:')
logger.info(str(args))
logger.info('Checkpoints will be saved in {}'.format(ckpt_dir))

# Data Preprocessing
logger.info("Loading dataset and preparing data...")
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
nb_node = adj.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_class = y_train.shape[1]

# Debugging: Print loaded data information
print("Dataset loaded:")
print("Number of nodes:", nb_node)
print("Training size:", nb_train)
print("Validation size:", nb_val)
print("Testing size:", nb_test)
print("Number of classes:", nb_class)

# Instantiate model
model = BertClassifier(pretrained_model=bert_init, nb_class=nb_class)
print("BertClassifier model instantiated with", bert_init)

# Convert one-hot labels to class IDs
y = th.LongTensor((y_train + y_val + y_test).argmax(axis=1))
label = {'train': y[:nb_train], 'val': y[nb_train:nb_train+nb_val], 'test': y[-nb_test:]}
print("Labels converted to class IDs")

# Load document text and clean it
logger.info("Loading and encoding document text...")
corpus_file = './data/corpus/' + dataset + '_shuffle.txt'
with open(corpus_file, 'r') as f:
    text = f.read().replace('\\', '').split('\n')

print("Corpus loaded with {} documents.".format(len(text)))

# Encode input using tokenizer
def encode_input(text, tokenizer):
    print("Encoding input with tokenizer...")
    input_data = tokenizer(text, max_length=max_length, truncation=True, padding=True, return_tensors='pt')
    print("Input encoding complete.")
    return input_data.input_ids, input_data.attention_mask

input_ids_, attention_mask_ = encode_input(text, model.tokenizer)
input_ids, attention_mask = {}, {}
input_ids['train'], input_ids['val'], input_ids['test'] = input_ids_[:nb_train], input_ids_[nb_train:nb_train+nb_val], input_ids_[-nb_test:]
attention_mask['train'], attention_mask['val'], attention_mask['test'] = attention_mask_[:nb_train], attention_mask_[nb_train:nb_val+nb_train], attention_mask_[-nb_test:]

print("Data encoding complete. Train/Val/Test split of input IDs and attention masks created.")

# Create DataLoader
logger.info("Creating DataLoader for train, validation, and test datasets...")
datasets, loader = {}, {}
for split in ['train', 'val', 'test']:
    datasets[split] = Data.TensorDataset(input_ids[split], attention_mask[split], label[split])
    loader[split] = Data.DataLoader(datasets[split], batch_size=batch_size, shuffle=(split == 'train'))
    print(f"{split.capitalize()} DataLoader created with {len(datasets[split])} samples.")

# Initialize optimizer and scheduler
optimizer = th.optim.Adam(model.parameters(), lr=bert_lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
print("Optimizer and scheduler initialized.")

# Training step
def train_step(engine, batch):
    global model, optimizer
    model.train()
    model = model.to(gpu)
    optimizer.zero_grad()
    input_ids, attention_mask, label = [x.to(gpu) for x in batch]
    y_pred = model(input_ids, attention_mask)
    y_true = label.type(th.long)
    loss = F.cross_entropy(y_pred, y_true)
    loss.backward()
    optimizer.step()

    # Calculate training accuracy
    train_loss = loss.item()
    with th.no_grad():
        y_true, y_pred = y_true.cpu(), y_pred.argmax(axis=1).cpu()
        train_acc = accuracy_score(y_true, y_pred)

    print(f"Train step - Loss: {train_loss}, Accuracy: {train_acc}")
    return train_loss, train_acc

trainer = Engine(train_step)

# Evaluation step
def test_step(engine, batch):
    global model
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        input_ids, attention_mask, label = [x.to(gpu) for x in batch]
        y_pred = model(input_ids, attention_mask)
        print(f"Test step - Batch Size: {input_ids.shape[0]}")
        return y_pred, label

evaluator = Engine(test_step)

# Metrics for evaluation
metrics = {
    'acc': Accuracy(),
    'nll': Loss(th.nn.CrossEntropyLoss())
}
for n, f in metrics.items():
    f.attach(evaluator, n)

# Logging function after each epoch
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    logger.info("Evaluating on train dataset...")
    evaluator.run(loader['train'])
    train_metrics = evaluator.state.metrics
    train_acc, train_nll = train_metrics["acc"], train_metrics["nll"]

    logger.info("Evaluating on validation dataset...")
    evaluator.run(loader['val'])
    val_metrics = evaluator.state.metrics
    val_acc, val_nll = val_metrics["acc"], val_metrics["nll"]

    logger.info("Evaluating on test dataset...")
    evaluator.run(loader['test'])
    test_metrics = evaluator.state.metrics
    test_acc, test_nll = test_metrics["acc"], test_metrics["nll"]

    logger.info(
        "Epoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f}  Test acc: {:.4f} loss: {:.4f}"
        .format(trainer.state.epoch, train_acc, train_nll, val_acc, val_nll, test_acc, test_nll)
    )

    # Save checkpoint if validation accuracy improves
    if val_acc > log_training_results.best_val_acc:
        logger.info("New checkpoint")
        th.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(ckpt_dir, 'checkpoint.pth')
        )
        log_training_results.best_val_acc = val_acc

    scheduler.step()

log_training_results.best_val_acc = 0

# Start training with progress indication
logger.info("Starting training...")
for epoch in tqdm(range(nb_epochs), desc="Epochs"):
    trainer.run(loader['train'], max_epochs=1)
