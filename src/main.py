from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import torch

#MODELOS PRUEBA
#mistralai/Mistral-7B-Instruct-v0.2 --> https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
#meta-llama/Llama-2-7b-chat-hf--> https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
#bigcode/starcoder2-15b --> https://huggingface.co/bigcode/starcoder2-15b
#microsoft/phi-2 --> https://huggingface.co/microsoft/phi-2
#openai-community/gpt2 --> https://huggingface.co/openai-community/gpt2
#bigscience/bloom-560m --> https://huggingface.co/bigscience/bloom-560m
#TheBloke/vicuna-7B-v1.3-GPTQ --> https://huggingface.co/TheBloke/vicuna-7B-v1.3-GPTQ
#upstage/SOLAR-10.7B-Instruct-v1.0 --> https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using " + DEVICE + " device.")

def generate_llama_2_input(msg):
    formatted_msg = f"Human: {msg}\nAI:"
    return formatted_msg.strip()

def main_cuantizy():
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config = quantization_config, low_cpu_mem_usage = True)

    input_text = "Write me about Spain and Mexico."
    input_text = input_text if MODEL_NAME != "meta-llama/Llama-2-7b-chat-hf" else generate_llama_2_input(input_text)
    input_ids = tokenizer(input_text, return_tensors="pt").to(DEVICE)
    
    model = model
    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))
    
if __name__ == "__main__":
    # En el caso de que te tengas que loggear con HF para solicitar permisos a un repositorio
    # login()
    main_cuantizy()