conda activate LLM_latent

#frozen LLama with input MLP: given the first 6 latents learning non autoregressively to directly predict the 20th latent
python train.py --freeze --not_autoregressive --ar_eval_freq=50 --batch_size=16 --epochs=154 --lr=0.00021262666696020535 --n_files=428 --target_gt=20 --training_gts=6

#frozen LLama with input MLP: can we overfit to a small dataset with the internal residual conncetion?
python train.py --freeze --internal_latent_residual --ar_eval_freq=100 --batch_size=16 --epochs=493 --lr=0.00011020409514511416 --n_files=7

#frozen LLama with input MLP: using a residual connection and prepending the prompt cutting it at exactly 25 tokens, how do we fare?
python train.py --freeze --residual --prompt_appended --batch_size=16 --epochs=106 --lr=0.0011730905040525888 --n_files=512 --tokenizer_max_length=25