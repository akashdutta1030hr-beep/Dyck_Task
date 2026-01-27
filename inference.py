import torch
from unsloth import FastLanguageModel

# ============================================================================
# Configuration
# ============================================================================
BASE_MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_PATH = "results"  # Path to fine-tuned model
MAX_LENGTH = 512

# ============================================================================
# Load Fine-tuned Model
# ============================================================================
print("Loading fine-tuned model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_LENGTH,
    dtype=torch.float16,
    load_in_4bit=True,
)

# Ensure the pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set model to evaluation mode
FastLanguageModel.for_inference(model)

print("Model loaded successfully! ✓\n")

# ============================================================================
# Inference Function
# ============================================================================
def complete_dyck_sequence(partial_sequence, max_new_tokens=256, temperature=0.8, top_p=0.95):
    """
    Complete a partial Dyck sequence using the fine-tuned model.
    
    Args:
        partial_sequence: The partial Dyck sequence to complete
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        top_p: Nucleus sampling parameter
    
    Returns:
        The model's generated response (reasoning + completion)
    """
    # Format the input as a chat message (matching training format)
    user_message = (
        f"Complete the following Dyck language sequence by adding the minimal necessary closing brackets.\n\n"
        f"Sequence: {partial_sequence}\n\n"
        f"Rules:\n"
        f"- Add only the closing brackets needed to match all unmatched opening brackets\n"
        f"- Do not add any extra bracket pairs beyond what is required\n\n"
        f"Provide only the complete valid sequence."
    )
    
    # Format as messages for chat template
    messages = [
        {"role": "user", "content": user_message}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part (excluding the input prompt)
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    return generated_text.strip()

# ============================================================================
# Interactive Inference
# ============================================================================
def interactive_inference():
    """Run interactive inference session."""
    print("=" * 70)
    print("Dyck Sequence Completion - Interactive Inference")
    print("=" * 70)
    print("Enter a partial Dyck sequence to complete.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            # Get user input
            partial_seq = input("Enter partial Dyck sequence: ").strip()
            
            if partial_seq.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not partial_seq:
                print("Please enter a valid sequence.\n")
                continue
            
            print("\nGenerating completion...")
            print("-" * 70)
            
            # Generate response
            response = complete_dyck_sequence(partial_seq)
            
            # Display response
            print("Model Response:")
            print(response)
            print("-" * 70)
            print()
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            continue

# ============================================================================
# Batch Inference Example
# ============================================================================
def batch_inference_examples():
    """Run inference on example sequences."""
    print("=" * 70)
    print("Dyck Sequence Completion - Example Inferences")
    print("=" * 70)
    
    examples = [
        "{{[{⟨⟦({[{(⟨{[(⟨(<<{",
        "[⟨([<⟨{<⟦[{[<⟨<⟨⟦⟨{⟨",
        "⟨⟨⟨⟨([<<⟨(⟨[<⟨⟦⟨⟨[([",
        "⟨<[{{{{({[⟦⟨[<{<{<⟨<",
    ]
    
    for i, partial_seq in enumerate(examples, 1):
        print(f"\nExample {i}: {partial_seq}")
        print("-" * 70)
        
        response = complete_dyck_sequence(partial_seq)
        print("Model Response:")
        print(response)
        print("=" * 70)

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Single inference mode
        partial_sequence = sys.argv[1]
        print(f"Input sequence: {partial_sequence}\n")
        print("Generating completion...\n")
        response = complete_dyck_sequence(partial_sequence)
        print("=" * 70)
        print("Model Response:")
        print("=" * 70)
        print(response)
    elif len(sys.argv) > 1 and sys.argv[1] == "--examples":
        # Run example inferences
        batch_inference_examples()
    else:
        # Interactive mode
        interactive_inference()
import torch
from unsloth import FastLanguageModel

# ============================================================================
# Configuration
# ============================================================================
BASE_MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_PATH = "results"  # Path to fine-tuned model
MAX_LENGTH = 512

# ============================================================================
# Load Fine-tuned Model
# ============================================================================
print("Loading fine-tuned model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_LENGTH,
    dtype=torch.float16,
    load_in_4bit=True,
)

# Ensure the pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set model to evaluation mode
FastLanguageModel.for_inference(model)

print("Model loaded successfully! ✓\n")

# ============================================================================
# Inference Function
# ============================================================================
def complete_dyck_sequence(partial_sequence, max_new_tokens=256, temperature=0.8, top_p=0.95):
    """
    Complete a partial Dyck sequence using the fine-tuned model.
    
    Args:
        partial_sequence: The partial Dyck sequence to complete
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        top_p: Nucleus sampling parameter
    
    Returns:
        The model's generated response (reasoning + completion)
    """
    # Format the input as a chat message (matching training format)
    user_message = (
        f"Complete the following Dyck language sequence by adding the minimal necessary closing brackets.\n\n"
        f"Sequence: {partial_sequence}\n\n"
        f"Rules:\n"
        f"- Add only the closing brackets needed to match all unmatched opening brackets\n"
        f"- Do not add any extra bracket pairs beyond what is required\n\n"
        f"Provide only the complete valid sequence."
    )
    
    # Format as messages for chat template
    messages = [
        {"role": "user", "content": user_message}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part (excluding the input prompt)
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    return generated_text.strip()

# ============================================================================
# Interactive Inference
# ============================================================================
def interactive_inference():
    """Run interactive inference session."""
    print("=" * 70)
    print("Dyck Sequence Completion - Interactive Inference")
    print("=" * 70)
    print("Enter a partial Dyck sequence to complete.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            # Get user input
            partial_seq = input("Enter partial Dyck sequence: ").strip()
            
            if partial_seq.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not partial_seq:
                print("Please enter a valid sequence.\n")
                continue
            
            print("\nGenerating completion...")
            print("-" * 70)
            
            # Generate response
            response = complete_dyck_sequence(partial_seq)
            
            # Display response
            print("Model Response:")
            print(response)
            print("-" * 70)
            print()
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            continue

# ============================================================================
# Batch Inference Example
# ============================================================================
def batch_inference_examples():
    """Run inference on example sequences."""
    print("=" * 70)
    print("Dyck Sequence Completion - Example Inferences")
    print("=" * 70)
    
    examples = [
        "{{[{⟨⟦({[{(⟨{[(⟨(<<{",
        "[⟨([<⟨{<⟦[{[<⟨<⟨⟦⟨{⟨",
        "⟨⟨⟨⟨([<<⟨(⟨[<⟨⟦⟨⟨[([",
        "⟨<[{{{{({[⟦⟨[<{<{<⟨<",
    ]
    
    for i, partial_seq in enumerate(examples, 1):
        print(f"\nExample {i}: {partial_seq}")
        print("-" * 70)
        
        response = complete_dyck_sequence(partial_seq)
        print("Model Response:")
        print(response)
        print("=" * 70)

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Single inference mode
        partial_sequence = sys.argv[1]
        print(f"Input sequence: {partial_sequence}\n")
        print("Generating completion...\n")
        response = complete_dyck_sequence(partial_sequence)
        print("=" * 70)
        print("Model Response:")
        print("=" * 70)
        print(response)
    elif len(sys.argv) > 1 and sys.argv[1] == "--examples":
        # Run example inferences
        batch_inference_examples()
    else:
        # Interactive mode
        interactive_inference()
