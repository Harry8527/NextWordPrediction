from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Function to get the next word suggestion
def predict_next_word(prompt_text, max_new_tokens=5):
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True)
    # print(inputs)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_k=50,top_p=0.95, temperature=0.8, pad_token_id=model.config.pad_token_id)
    print(output_ids)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # print(generated_text)
    # print(type(generated_text))
    
    # Extract only the new word added
    new_text = generated_text[len(prompt_text):].strip()
    print(new_text)
    next_word = new_text.split()[0] if new_text else ""
    return next_word

while True:
    prompt = input("\nEnter a sentence: ")
    if prompt.strip().lower() in ["exit", "quit"]:
        break
    suggestion = predict_next_word(prompt)
    print("Next word suggestion:", suggestion)

# In the long term or as a future version of this I can built the following:
    # Make a webpage where people can write messages, and randomly anyone message will be shortlisted at 8 PM, and highlighted as best sentence of the day etc.
    # After 5 words this word prediction model should get triggered automatically.
        # For example, say if my sentence is as:
        # India is the best country in the world, not only because of its cultural heritage but also because a wide diversity of people live here together.
        # For this sentence after the word "country" the next word prediction should get  activated, and start suggesting next word.
    
    