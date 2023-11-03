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
    def __init__(self, llama_weights_path, llama_model=None, tokenizer=None, input_dim=16384, output_dim=16384, hidden_dim=4096, tokenizer_max_length=20, freeze=False, residual_connection=False):
        super(LinearAdaptedLlama, self).__init__()
        
        # Initialize the Llama model
        if llama_model is not None:
            self.llama_model = llama_model
            try:
                self.tokenizer = tokenizer
            except:
                print("you cant pass the model without passing the tokenizer as well!!")
                self.llama_model, self.tokenizer = load_meta_pretrained_llama_model(llama_weights_path)

        else:
            self.llama_model, self.tokenizer = load_meta_pretrained_llama_model(llama_weights_path)

        # Batches need to match sequence length dimension, padding is required
        self.tokenizer_max_length = tokenizer_max_length
        self.tokenizer.truncation_side = "right"
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.unk_token

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
        
    def forward(self, x, prompts=None):
        
        # Adapt the input
        latent_input_embedding = self.input_mlp(x)

        # Check if prompt tokenizing is necessary
        if prompts is not None:
            # Use tokenizer and extract ids and attention masks
            batch_tokens = [(self.tokenizer(prompt, padding='max_length', truncation = True, max_length=self.tokenizer_max_length, return_tensors="pt").to(device)) for prompt in prompts]
            batch_tokens_ids = [batch_tokens_i['input_ids'] for batch_tokens_i in batch_tokens]

            # Attention mask such that padded token are ignored. Ones are added after truncation such that the latents are taken attended to
            batch_tokens_att_mask = torch.cat([batch_tokens_i['attention_mask'] for batch_tokens_i in batch_tokens])
            batch_tokens_att_mask = torch.cat((batch_tokens_att_mask,torch.ones((x.shape[0],x.shape[1])).to(device) ),dim=1)

            # Convert ids to Llama's input dimension
            token_input_embedding = torch.cat([self.llama_model.model.embed_tokens(tokens) for tokens in batch_tokens_ids])

            # Concatenate the tokens embedding with the diffusion latents embedding
            input_embeds = torch.cat((token_input_embedding,latent_input_embedding), dim=1)

            # Forward pass through Llama
            # Assuming x has shape [batch_size, sequence_length, feature_dim]
            # In this case, we use the calculated per-batch attention mask to ignore paddings
            outputs = self.llama_model(inputs_embeds=input_embeds, attention_mask=batch_tokens_att_mask,  output_hidden_states=True)
        
        # Otherwise, the embedding of the latent is the input to Llama
        else:
            input_embeds = latent_input_embedding

            # Forward pass through Llama
            # Assuming x has shape [batch_size, sequence_length, feature_dim]
            outputs = self.llama_model(inputs_embeds=input_embeds, output_hidden_states=True)

        # Get the hidden states from the last layer
        last_hidden_state = outputs.hidden_states[-1]
    
        # Adapt the output
        out = self.output_mlp(last_hidden_state)

        #keep the last {seq_length} outputs of the model forward pass if the beggining were prompts
        if prompts is not None:
            out = out[:,-x.shape[1]:]


        # If flag is activated it means we are calculating delta for every diffusion step rather than the next latent directly
        if self.residual_connection_flag:
            out = out + x
        
        return out