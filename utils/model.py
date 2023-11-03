import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_meta_pretrained_llama_model(path):

    if path is None:
        print("model path is null, attepting to load default 7B llama model")
        path = "/home/gperezsantamaria/local_data/autoregressive_difussion/Llama_playground/Llama_weights/Llama-2-7b"
    
    tokenizer = LlamaTokenizer.from_pretrained(path)
    model = LlamaForCausalLM.from_pretrained(path)
    
    return model, tokenizer

def freeze_llama_model_parameters(llama_model):
    for param in llama_model.parameters():
        param.requires_grad = False


class LinearAdaptedLlama(nn.Module):
    def __init__(self, llama_weights_path, llama_model=None, input_dim=16384, output_dim=16384, hidden_dim=4096, freeze=False, residual_connection=False):
        super(LinearAdaptedLlama, self).__init__()
        
        # Initialize the Llama model
        if llama_model is not None:
            self.llama_model = llama_model
        else:
            self.llama_model, _ = load_meta_pretrained_llama_model(llama_weights_path)

        if freeze:
            # Freeze Llama model parameters
            freeze_llama_model_parameters(self.llama_model)

        self.residual_connection_flag = residual_connection

        
        # MLP for adapting the input to Llama's input dimension
        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # MLP for adapting Llama's output back to the original dimension
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # Adapt the input
        
        input_embeds = self.input_mlp(x)
        
        # Forward pass through Llama
        # Assuming x has shape [batch_size, sequence_length, feature_dim]
        outputs = self.llama_model(inputs_embeds=input_embeds, output_hidden_states=True)

        # Get the hidden states from the last layer
        last_hidden_state = outputs.hidden_states[-1]
    
        # Adapt the output
        out = self.output_mlp(last_hidden_state)

        # If flag is activated it means we are calculating delta for every diffusion step rather than the next latent directly
        if self.residual_connection_flag:
            out = out + x
        
        return out