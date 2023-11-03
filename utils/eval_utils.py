import torch
from utils import diffusion_model
import wandb
from PIL import Image
import numpy as np
from tqdm import tqdm


###### Evaluation and metrics ######
def downsample_image(image, factor=4):
    """lengths_to_test
    Downsample the image by a specific factor using PIL.
    image is a numpy array of shape [height, width, 3]
    """
    # Generate PIL image from array
    pil_img = Image.fromarray((image * 255).astype(np.uint8))
    width, height = pil_img.size
    pil_img = pil_img.resize((width // factor, height // factor))
    
    # Convert back to numpy array and normalize to [0, 1]
    return np.array(pil_img) / 255.0
    

def predict_autorreg(Llama_model, X, gt_length):
    # Print warning to user if gt_length is greater than X.size(1)
    if gt_length > X.size(1):
        print(f'Warning: gt_length ({gt_length}) is greater than X.size(1) ({X.size(1)}). Setting gt_length to X.size(1)')
        gt_length = X.size(1)
    # Get the ground truth portion of the sequence
    input_seq = X[:, :gt_length] # first gt_length vectors in seq. Shape is [bs, gt_length, feats_dim]
    # First forward to initialize hidden and cell states of LSTM
    pred = Llama_model(input_seq)
    # Only care about the last prediction
    new_pred = pred[:, -1] # Shape is [bs, 1, feats_dim]
    new_pred = new_pred[:,None,:]
    input_seq = torch.cat((input_seq,new_pred), dim=1)  # Update the latent vector with predicted values
    # (Auto-regressively) Predict until last vectors
    for _ in range(gt_length, X.size(1)):
        pred = Llama_model(pred) # pred all timesteps in seq. Shape is [bs, gt_length, feats_dim]
        new_pred = pred[:, -1] # Shape is [bs, 1, feats_dim]
        new_pred = new_pred[:,None,:]
        input_seq = torch.cat((input_seq,new_pred), dim=1)  # Update the latent vector with predicted values

    final_prediction = input_seq

    return final_prediction

@torch.no_grad()
def test_autorreg_epoch(Llama_model, diff_model, data_loader, device, lengths_to_test=[1, 5, 10, 20]):
    diff_model.eval()
    psnrs = [] # list of dicts, one for each batch. Each dict has keys = lengths_to_test, and values = list of psnrs for each sample in batch
    losses = []
    for X, y, cond, _ in tqdm(data_loader):
        # Compute gt images of this batch
        X, y, cond = X.to(device), y.to(device), cond.to(device)
        y = y[:, -1]
        gt_ims = diffusion_model.to_img(diff_model, y.view(X.size(0), 4, 64, 64), to_numpy=False) # [bs, 512, 512, 3]
        batch_losses = { }
        batch_psnrs = { }
        for gt_length in lengths_to_test:
            preds = predict_autorreg(Llama_model, X, gt_length=gt_length) # [bs, S, feats_dim], where S = 1+X.size(1)-gt_length
            preds = preds[:, -1] # [bs, feats_dim]
            pred_ims = diffusion_model.to_img(diff_model, preds.view(X.size(0), 4, 64, 64), to_numpy=False) # [bs, 512, 512, 3] and values in [0, 1]
            # Compute loss in latent space
            loss = torch.mean((preds - y)**2, dim=1) # [bs]
            batch_losses[gt_length] = loss # b/c loss_function is averaged across batch
            # Compute PSNR in image space
            mse = torch.mean((pred_ims - gt_ims) ** 2, dim=(1,2,3)) # avg across height, width, channels -> [bs]
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            batch_psnrs[gt_length] = psnr

        psnrs.append(batch_psnrs)
        losses.append(batch_losses)

    # Now we want to return a dict with keys = lengths_to_test, and values = average psnr across batches
    psnrs = { gt_length : torch.cat([batch_psnrs[gt_length] for batch_psnrs in psnrs], dim=0).mean().item() for gt_length in lengths_to_test }
    losses = { gt_length : torch.cat([batch_losses[gt_length] for batch_losses in losses], dim=0).mean().item() for gt_length in lengths_to_test }
    
    return losses, psnrs

def visualize_reconstruction(llama_model, diff_model, test_data_loader, gt_length=20, sample_ids=[0, 1, 2, 3, 4, 5]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_gt_imgs = []
    all_pred_imgs = []
    captions = []
    
    # Iterate over sample_ids
    for sample_id in sample_ids:
        # Extract sample `sample_id` from test data loader
        X, y, cond, prompt = test_data_loader.dataset[sample_id]
        X, cond = X.unsqueeze(0).to(device), cond.unsqueeze(0).to(device)

        gt_img = diffusion_model.to_img(diff_model, y[-1].view(1, 4, 64, 64).to(device))
        all_preds = predict_autorreg(llama_model, X, gt_length=gt_length)
        last_pred = all_preds[0, -1]  
        reshaped_latent = last_pred.view(1, 4, 64, 64).to(device)
        pred_img = diffusion_model.to_img(diff_model, reshaped_latent)
        
        # Downsample imagespreds
        gt_img_downsampled = downsample_image(gt_img[0])
        pred_img_downsampled = downsample_image(pred_img[0])
        
        # Store images for later concatenation
        all_gt_imgs.append(gt_img_downsampled)
        all_pred_imgs.append(pred_img_downsampled)
        
        # Construct caption and append
        captions.append(f"Prompt: {prompt} | Top: Ground Truth, Bottom: Predicted")
    
    # Concatenate ground truth and predicted images vertically for each sample
    # Then concatenate the results horizontally to form the grid
    combined_images = np.concatenate(
        [np.concatenate([gt, pred], axis=0) for gt, pred in zip(all_gt_imgs, all_pred_imgs)], 
        axis=1
    )

    # Join all the captions into one
    full_caption = " || ".join(captions)

    images = wandb.Image(combined_images, caption=full_caption)
    
    return images


def visualize_mid_latents(Llama_model, diff_model, test_data_loader, num_initial_steps=20, sample_ids=[0, 1, 2, 3, 4, 5]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_gt_imgs = []
    all_pred_imgs = []
    captions = []
    
    # Iterate over sample_ids
    for sample_id in sample_ids:
        # Extract sample `sample_id` from test data loader
        X, y, cond, prompt = test_data_loader.dataset[sample_id]
        X, cond = X.unsqueeze(0).to(device), cond.unsqueeze(0).to(device)
        try:
            gt_img = diffusion_model.to_img(diff_model, X[:,num_initial_steps,:].view(1, 4, 64, 64).to(device))
        except:
            gt_img = diffusion_model.to_img(diff_model, y[-1].view(1, 4, 64, 64).to(device))
        preds = predict_autorreg(Llama_model, X, gt_length=num_initial_steps)
        pred = preds[0,num_initial_steps]
        reshaped_latent = pred.view(1, 4, 64, 64).to(device)
        pred_img = diffusion_model.to_img(diff_model, reshaped_latent, to_numpy=True)
        
        # Downsample images
        gt_img_downsampled = downsample_image(gt_img[0])
        pred_img_downsampled = downsample_image(pred_img[0])
        
        # Store images for later concatenation
        all_gt_imgs.append(gt_img_downsampled)
        all_pred_imgs.append(pred_img_downsampled)
        
        # Construct caption and append
        captions.append(f"Prompt: {prompt} | Top: Ground Truth, Bottom: Predicted")

    # Concatenate ground truth and predicted images vertically for each sample
    # Then concatenate the results horizontally to form the grid
    combined_images = np.concatenate(
        [np.concatenate([gt, pred], axis=0) for gt, pred in zip(all_gt_imgs, all_pred_imgs)], 
        axis=1
    )

    # Join all the captions into one
    full_caption = " || ".join(captions)

    images = wandb.Image(combined_images, caption=full_caption)
    
    return images