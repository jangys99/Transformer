import os, sys, time
import wandb
import torch
import warnings
warnings.filterwarnings('ignore')

from torch import nn, optim
from tqdm import tqdm

from config import *
from models.build_model import build_model
from data import Multi30k
from utils import get_bleu_score, greedy_decode


DATASET = Multi30k()

class EarlyStopping:
    def __init__(self, mode, patience):
        self.mode = mode # ['max', 'min']
        self.score = 0 if mode == 'max' else torch.inf
        self.patience = patience
        
    def update(self, score):
        if self.mode == 'max':
            if self.score > score:
                self.patience -= 1
            else:
                self.score = score
                
        else:
            if self.score < score:
                self.patience -= 1
            else:
                self.score = score
        if self.patience == 0:
            return True
        return False
    
    
def train(model, data_loader, optimizer, criterion, epoch, checkpoint_dir):
    model.train()
    epoch_loss = 0
    
    for idx, (src, tgt) in enumerate(data_loader):
        src = src.to(model.device)
        tgt = tgt.to(model.device)
        tgt_x = tgt[:, :-1]
        tgt_y = tgt[:, 1:]
        
        optimizer.zero_grad()
        
        output, _ = model(src, tgt_x)
        
        y_hat = output.contiguous().view(-1, output.shape[-1])
        y_gt = tgt_y.contiguous().view(-1)
        loss = criterion(y_hat, y_gt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
    num_samples = idx + 1
    
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f"{epoch:04d}.pt")
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                   }, checkpoint_file)
        
    return epoch_loss / num_samples


def evaluate(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0

    total_bleu = []
    with torch.no_grad():
        for idx, (src, tgt) in enumerate(data_loader):
            src = src.to(model.device)
            tgt = tgt.to(model.device)
            tgt_x = tgt[:, :-1]
            tgt_y = tgt[:, 1:]

            output, _ = model(src, tgt_x)

            y_hat = output.contiguous().view(-1, output.shape[-1])
            y_gt = tgt_y.contiguous().view(-1)
            loss = criterion(y_hat, y_gt)

            epoch_loss += loss.item()
            score = get_bleu_score(output, tgt_y, DATASET.vocab_tgt, DATASET.specials)
            total_bleu.append(score)
        num_samples = idx + 1

    loss_avr = epoch_loss / num_samples
    bleu_score = sum(total_bleu) / len(total_bleu)
    return loss_avr, bleu_score


def main():
    wandb.init(project="transformer", entity="youngsoo", name=VERSION)

    model = build_model(len(DATASET.vocab_src), len(DATASET.vocab_tgt), device=DEVICE, dr_rate=DROPOUT_RATE)

    def initialize_weights(model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.kaiming_uniform_(model.weight.data)

    model.apply(initialize_weights)

    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

    criterion = nn.CrossEntropyLoss(ignore_index=DATASET.pad_idx)

    train_iter, valid_iter, test_iter = DATASET.get_iter(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    epoch_bar = tqdm(range(N_EPOCH))
    
    for epoch in epoch_bar:
        epoch_bar.set_description(f"*****epoch: {epoch:02}*****")
        train_loss = train(model, train_iter, optimizer, criterion, epoch, CHECKPOINT_DIR)
        valid_loss, bleu_score  = evaluate(model, valid_iter, criterion)
        if epoch > WARM_UP_STEP:
            scheduler.step(valid_loss)
        print(f"\ntrain_loss: {train_loss:.5f}, valid_loss: {valid_loss:.5f}, bleu_score: {bleu_score:.5f}")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": valid_loss,
            "bleu_score": bleu_score,
        })

        print(DATASET.translate(model, "A little girl climbing into a wooden playhouse .", greedy_decode))
        # expected output: "Ein kleines Mädchen klettert in ein Spielhaus aus Holz ."
        
        if es.update(valid_loss):
            break  
        
    test_loss, bleu_score = evaluate(model, test_iter, criterion)
    print(f"test_loss: {test_loss:.5f}, bleu_score: {bleu_score:.5f}")
    
    wandb.log({
        "test_loss": {test_loss},
        "bleu_score": {bleu_score},
        })
    wandb.finish()
    
if __name__ == "__main__":
    
    torch.manual_seed(0)
    es = EarlyStopping(mode='min', patience=10)
    main()