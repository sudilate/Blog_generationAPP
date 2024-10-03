import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Define paths
model_pth_path = "//Users//amitsarang//.llama//checkpoints//Llama3.2-1B"  # Your .pth file path
config_path = "//Users//amitsarang//.llama//checkpoints//Llama3.2-1B"  # Path to the model configuration (e.g., config.json)
output_dir = "//Users//amitsarang//Blog_Generator"  # Directory to save the converted model
# model_id = "Llama3.2-1B"
# Load the LLaMA model from the .pth file
model = LlamaForCausalLM.from_pretrained(
    model_pth_path,
    config=config_path,
    ignore_mismatched_sizes=True  # This may help if there are size mismatches
)

# Save the model in .bin format for Hugging Face or Langchain-style usage
model.save_pretrained(output_dir)

# Optionally, save tokenizer if available
tokenizer = LlamaTokenizer.from_pretrained(config_path)
tokenizer.save_pretrained(output_dir)

print(f"Model converted and saved to {output_dir}")
