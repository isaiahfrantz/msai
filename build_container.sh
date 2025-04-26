#!/usr/bin/bash

# build bitnet-1bit (MS 1bit) container with prompt script
# be in build dir

git clone --recurse-submodules https://github.com/microsoft/BitNet.git
cd BitNet
git submodule update --init --recursive

# update base image
podman pull ubuntu:25.04

# add Dockerfile
tee Dockerfile <<EOF
FROM ubuntu:latest

# Install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    cmake \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Create virtual environment
RUN python3 -m venv /opt/venv

# Activate and install requirements
ENV PATH="/opt/venv/bin:\$PATH"

# Install Python dependencies (safely in Ubuntu >=22.04)
RUN git clone --filter=blob:none --quiet https://github.com/shumingma/transformers.git /tmp/pip-req-build-yk87dsc7 \
    && cd /tmp/pip-req-build-yk87dsc7 \
    && pip3 install . \
    && cd -
# RUN pip3 install --no-cache-dir git+https://github.com/shumingma/transformers.git

RUN pip3 install --no-cache-dir -r requirements.txt

# Build the project (change the model here if needed)
RUN python3 setup_env.py --hf-repo tiiuae/Falcon3-7B-Instruct-1.58bit -q i2_s || { echo 'Setup failed'; tail -n 20 /var/log/syslog; exit 1; }
# RUN python3 setup_env.py --hf-repo tiiuae/Falcon3-7B-Instruct-1.58bit -q i2_s

# Set the default command to run prompt.py and accept input from CLI
ENTRYPOINT ["python3", "prompt.py"]
EOF

# create promtp.py script
tee prompt.py <<EOF
import sys
from bitnet import BitNetModel
from transformers import AutoTokenizer

def main():
    if len(sys.argv) < 2:
        print("Usage: python prompt.py '<your prompt>'")
        sys.exit(1)

    prompt = sys.argv[1]

    # Load tokenizer and model
    model_name = "tiiuae/Falcon3-7B-Instruct-1.58bit"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading BitNet model...")
    model = BitNetModel.from_pretrained(model_name)

    # Tokenize input
    print(f"Tokenizing prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate response
    print("Generating response...")
    outputs = model.generate(**inputs, max_new_tokens=50)

    # Decode and print result
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n=== Model Response ===")
    print(result)

if __name__ == "__main__":
    main()
EOF

# example from ms
tee ms-example.py <<EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/bitnet-b1.58-2B-4T"

# if len(sys.argv) < 3:
#     print("Usage: python prompt.py '<your prompt>' '<your question>'")
#     sys.exit(1)

# toto: add flags for these instead of positional args
# prompt = sys.argv[1] || "You are a helpful AI assistant."
# question = sys.argv[2] || "How are you?"

prompt = "You are a helpful AI assistant."
question = "How are you?"

print(f"prompt=>{prompt}<\nquestion=>{question}<")


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)

# Apply the chat template
messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": question},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
chat_input = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
chat_outputs = model.generate(**chat_input, max_new_tokens=50)
response = tokenizer.decode(chat_outputs[0][chat_input['input_ids'].shape[-1]:], skip_special_tokens=True) # Decode only the response part
print("\nAssistant Response:", respoznse)
EOF

podman build -t bitnet-image .
