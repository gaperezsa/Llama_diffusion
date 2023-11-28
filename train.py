import argparse
from tqdm import tqdm
import time
import torch
from utils import data_loading
from utils import diffusion_model
from utils import model
from utils import eval_utils
import wandb


###### Train and Testing epochs ######

@torch.enable_grad()
def train_epoch(model, data_loader, optimizer, loss_function, device, superv_iters=1, append_prompt=False):
    model.train()
    total_loss = 0
    for X, y, cond, prompts in tqdm(data_loader, desc="Training", unit="batch"):
        X, y, cond = X.to(device), y.to(device), cond.to(device)
        optimizer.zero_grad()
        if append_prompt:
            predictions = model(X, prompts)
        else:
            predictions = model(X)
        loss = loss_function(predictions[:, superv_iters:], y[:, superv_iters:])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


@torch.no_grad()
def test_epoch(model, data_loader, loss_function, device, append_prompt=False):
    model.eval()
    total_loss = 0
    for X, y, cond, prompts in tqdm(data_loader, desc="Testing", unit="batch"):
        X, y, cond = X.to(device), y.to(device), cond.to(device)
        if append_prompt:
            predictions = model(X, prompts)
        else:
            predictions = model(X)
        loss = loss_function(predictions, y)
        total_loss += loss.item()
    
    return total_loss / len(data_loader)



#### Main pipeline ####

def main(args):

    # Initialize wandb
    wandb.init(project="LLM_latent", entity=args.entity_name, config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data preprocessing
    all_xs, all_prompts, all_conditions = data_loading.preprocess_data(args.n_files)

    # Load the pretrained model
    diff_model = None if args.debug else diffusion_model.load_pretrained_model(args.ckpt, args.config)

    # Create dataloaders
    train_data_loader, val_data_loader, test_data_loader = data_loading.get_dataloaders(all_xs, all_prompts, all_conditions, args.train_percentage, args.batch_size)
    visual_train_data_loader, visual_val_data_loader, visual_test_data_loader = data_loading.get_dataloaders(all_xs, all_prompts, all_conditions, args.train_percentage, (args.batch_size//4)+1)


    # Load Llama and tokenizer
    llama_model = model.LinearAdaptedLlama(llama_weights_path = "/home/gperezsantamaria/local_data/autoregressive_difussion/Llama_playground/Llama_weights/Llama-2-7b",tokenizer_max_length = args.tokenizer_max_length, freeze=args.freeze, normal_layers_finetuning = args.normal_layers_finetuning, residual_connection=args.residual, internal_latent_residual_connection = args.internal_latent_residual).to(device)

    # Log model size to wandb
    param_count = sum(p.numel() for p in llama_model.parameters() if p.requires_grad)
    print(f"Linearly adapted Llama model size: {param_count}")
    wandb.log({"Linearly adapted Llama model_size": param_count})

    # Training set_up
    params_to_optimize = filter(lambda p: p.requires_grad, llama_model.parameters())
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr)
    loss_function = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        # Training
        start = time.time()
        train_loss = train_epoch(llama_model, train_data_loader, optimizer, loss_function, device, superv_iters=args.superv_iters, append_prompt=args.prompt_appended)
        end = time.time()

        # Val and Testing
        val_loss = test_epoch(llama_model, val_data_loader, loss_function, device, append_prompt=args.prompt_appended)
        #test_loss = test_epoch(llama_model, test_data_loader, loss_function, device)

        # Console logging
        print(f"Epoch {epoch} | Train loss: {train_loss} | Val loss: {val_loss} | Train Time: {end-start}")

        # Log to wandb
        wandb.log({
            "train_loss": train_loss, 
            "val_loss": val_loss, 
            # "test_loss": test_loss
        })

        # Evaluate
        if (epoch % args.ar_eval_freq == 0) or (epoch+1 == args.epochs): 
            # Visualization after each epoch
            for gt_length in tqdm(args.lengths_to_test, desc="Reconstructing and visualizing"):
                eval_images = eval_utils.visualize_reconstruction(llama_model, diff_model, visual_val_data_loader, gt_length=gt_length, append_prompt=args.prompt_appended)
                wandb.log({f"eval_reconstruction_gt{gt_length}": eval_images})
                train_images = eval_utils.visualize_reconstruction(llama_model, diff_model, visual_train_data_loader, gt_length=gt_length, append_prompt=args.prompt_appended)
                wandb.log({f"train_reconstruction_gt{gt_length}": train_images})
                eval_mid_images = eval_utils.visualize_mid_latents(llama_model, diff_model, visual_val_data_loader,gt_length)
                wandb.log({f"eval_reconstruction with access to {gt_length}gt, predicting {gt_length+1}": eval_mid_images})
                train_mid_images = eval_utils.visualize_mid_latents(llama_model, diff_model, visual_train_data_loader,gt_length)
                wandb.log({f"train_reconstruction with access to {gt_length}gt, predicting {gt_length+1}": train_mid_images})

            # Compute error and PSNR when generating autorregressively
            train_losses, train_psnrs = eval_utils.test_autorreg_epoch(llama_model, diff_model, visual_train_data_loader, device, lengths_to_test=args.lengths_to_test, append_prompt=args.prompt_appended)
            val_losses, val_psnrs = eval_utils.test_autorreg_epoch(llama_model, diff_model, visual_val_data_loader, device, lengths_to_test=args.lengths_to_test, append_prompt=args.prompt_appended)
            # test_losses, test_psnrs = test_autorreg_epoch(model, seq_model, test_data_loader, device, lengths_to_test=args.lengths_to_test)

            # test_losses and test_psnrs are dicts with keys = lengths_to_test (scalars), and (scalar) values = average loss/psnr across batches
            # Upload these to wandb, changing the key names to show what they are
            wandb.log({f"train_loss_gt{gt_length}": train_loss for gt_length, train_loss in train_losses.items()})
            wandb.log({f"train_psnr_gt{gt_length}": train_psnr for gt_length, train_psnr in train_psnrs.items()})

            wandb.log({f"val_loss_gt{gt_length}": val_loss for gt_length, val_loss in val_losses.items()})
            wandb.log({f"val_psnr_gt{gt_length}": val_psnr for gt_length, val_psnr in val_psnrs.items()})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an auto-regressive model for sequential image-like data.")
    parser.add_argument("--entity_name", type=str, default="gaperezsa", help="Entity name for Weights & Biases.")
    parser.add_argument("--ckpt", type=str, default="/home/gperezsantamaria/autoregressive_difussion/Weights/v2-1_512-ema-pruned.ckpt", help="Path to the checkpoint file.")
    parser.add_argument("--config", type=str, default="/home/gperezsantamaria/autoregressive_difussion/configs/stable-diffusion/v2-inference_gen_dataset.yaml", help="Path to the configuration file.")
    parser.add_argument("--freeze", action="store_true",default=False, help="Freeze Llama backbone")
    parser.add_argument("--normal_layers_finetuning", action="store_true",default=False, help="Unfreeze only normalization layers of Llama backbone")
    parser.add_argument("--residual", action="store_true",default=False, help="Do a residual conncetion between the input latent and the prediciton of the linear adapted llama")
    parser.add_argument("--internal_latent_residual", action="store_true",default=False, help="Do a residual conncetion between the input latent and the prediciton of the linear adapted llama")
    parser.add_argument("--prompt_appended", action="store_true",default=False, help="Wether to tokenize the corresponding prompt (caption) for every image and append it to the input sequence")
    parser.add_argument("--tokenizer_max_length", type=int, default=20, help="If prompts are to be prepended, what is the standardized length of tokens per batch")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--n_files", type=int, default=1, help="Number of files to use for training.")
    parser.add_argument("--train_percentage", type=float, default=0.8, help="Percentage of data to use for training.")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    parser.add_argument("--lengths_to_test", type=int, nargs='+', default=[10, 15, 17], help="Lengths to test for autorregressive prediction.")
    parser.add_argument("--ar_eval_freq", type=int, default=50, help="Frequency (in epochs) at which to evaluate autorregressive prediction.")
    parser.add_argument("--superv_iters", type=int, default=0, help="Number of supervised iterations (from this number on, i.e. N=15 will supervise predictions 15, 16, ... until the end).")
    args = parser.parse_args()
    main(args)