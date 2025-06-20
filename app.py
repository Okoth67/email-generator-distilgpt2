import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load fine‑tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("C:/Users/Admin/distilgpt2-email-finetuned")
model = AutoModelForCausalLM.from_pretrained("C:/Users/Admin/distilgpt2-email-finetuned")

def generate_email(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        inputs.input_ids,
        max_new_tokens=150,
        temperature=0.8,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

demo = gr.Interface(
    fn=generate_email,
    inputs=gr.Textbox(lines=2, placeholder="Enter your email prompt..."),
    outputs="text",
    title="Email Generator",
    description="Generate professional emails with a fine‑tuned DistilGPT2 model."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
