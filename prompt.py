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

