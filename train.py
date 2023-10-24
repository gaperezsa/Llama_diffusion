import argparse
from tqdm import tqdm
import time
import torch
from utils import data_loading
from utils import model


@torch.enable_grad()
def train_epoch(model, data_loader, optimizer, loss_function, device, superv_iters=19):
    model.train()
    total_loss = 0
    for X, y, cond, _ in tqdm(data_loader, desc="Training", unit="batch"):
        X, y, cond = X.to(device), y.to(device), cond.to(device)
        optimizer.zero_grad()
        predictions = model(X)
        loss = loss_function(predictions[:, superv_iters:], y[:, superv_iters:])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


@torch.no_grad()
def test_epoch(model, data_loader, loss_function, device):
    model.eval()
    total_loss = 0
    for X, y, cond, _ in tqdm(data_loader, desc="Testing", unit="batch"):
        X, y, cond = X.to(device), y.to(device), cond.to(device)
        predictions = model(X) # predictions is of gt_lengthshape [bs, seq_len, feats_dim]
        loss = loss_function(predictions, y)
        total_loss += loss.item() * X.size(0)
    
    return total_loss / data_loader.dataset.__len__()


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data preprocessing
    all_xs, all_prompts, all_conditions = data_loading.preprocess_data(args.n_files)

    # Create dataloaders
    train_data_loader, val_data_loader, test_data_loader = data_loading.get_dataloaders(all_xs, all_prompts, all_conditions, args.train_percentage, args.batch_size)

    data_dim = all_xs.shape[2]
    diffusion_steps = all_xs.shape[0]-1

    # Load Llama and tokenizer
    llama_model = model.LinearAdaptedLlama(llama_weights_path = "/home/gperezsantamaria/local_data/autoregressive_difussion/Llama_playground/Llama_weights/Llama-2-7b").to(device)

    # Log model size to wandb
    param_count = sum(p.numel() for p in llama_model.parameters() if p.requires_grad)
    print(f"Model size: {param_count}")

    # Training set_up
    params_to_optimize = filter(lambda p: p.requires_grad, llama_model.parameters())
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr)
    loss_function = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        # Training
        start = time.time()
        train_loss = train_epoch(llama_model, train_data_loader, optimizer, loss_function, device, superv_iters=args.superv_iters)
        end = time.time()

        # Val and Testing
        # _ = test_epoch(seq_model, train_data_loader, loss_function, device)
        val_loss = test_epoch(llama_model, val_data_loader, loss_function, device)
        # test_loss = test_epoch(seq_model, test_data_loader, loss_function, device)

        print(f"Epoch {epoch} | Train loss: {train_loss} | Val loss: {val_loss} | Train Time: {end-start}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an auto-regressive model for sequential image-like data.")
    parser.add_argument("--entity_name", type=str, default="gaperezsa", help="Entity name for Weights & Biases.")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--n_files", type=int, default=10, help="Number of files to use for training.")
    parser.add_argument("--train_percentage", type=float, default=0.8, help="Percentage of data to use for training.")
    parser.add_argument("--lengths_to_test", type=int, nargs='+', default=[1, 5, 10, 15, 16, 17, 18, 19, 20], help="Lengths to test for autorregressive prediction.")
    parser.add_argument("--ar_eval_freq", type=int, default=100, help="Frequency (in epochs) at which to evaluate autorregressive prediction.")
    parser.add_argument("--superv_iters", type=int, default=17, help="Number of supervised iterations (from this number on, i.e. N=15 will supervise predictions 15, 16, ... until the end).")
    args = parser.parse_args()
    main(args)