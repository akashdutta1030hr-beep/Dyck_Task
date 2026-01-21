# Dyck Language Dataset Generator

A Python tool for generating training datasets for Dyck language bracket matching tasks with step-by-step reasoning.

## Overview

This project generates synthetic datasets for training language models on Dyck language completion tasks. The Dyck language consists of properly matched brackets, and the task is to complete a partial sequence by adding the minimal necessary closing brackets.

## Files

### `generator.py`
The main dataset generator script. It contains:

- **`generate_dyck_sequence(max_depth=5, max_length=40)`**: Generates a valid, complete Dyck sequence using a stack-based algorithm
- **`make_partial(sequence, min_cut=1)`**: Creates a partial sequence by randomly cutting a full sequence
- **`compute_stack(sequence)`**: Computes the remaining unmatched opening brackets after scanning a sequence
- **`make_reasoning(stack)`**: Generates step-by-step reasoning text explaining the stack state
- **`build_sample()`**: Creates a single dataset sample with prompt and response
- **`generate_dataset(n_samples, output_file)`**: Generates the full dataset in JSONL format

**Usage:**
```bash
python generator.py
```

This will generate 100,000 samples and save them to `dyck_reasoning_dataset.jsonl`.

### `run.sh`
Bash script wrapper for running the dataset generation. Automatically detects and uses the available Python interpreter (python3 or python).

**Usage:**
```bash
bash run.sh
```

### `dyck_reasoning_dataset.jsonl`
The generated dataset file in JSONL format (JSON Lines). Each line contains a JSON object with:
- `prompt`: The task prompt asking to complete the Dyck sequence
- `response`: The model response containing:
  - `<think>` section with step-by-step reasoning
  - The completed sequence

**Example entry:**
```json
{
  "prompt": "Complete the following Dyck language sequence by adding the minimal necessary closing brackets.\n\nSequence: ({[]})(([(<>{})[][{}<>{}]",
  "response": "<think>\nScan the sequence from left to right.\nPush unmatched opening brackets onto a stack.\nRemaining stack (bottom to top): (, (, [\nClose brackets in reverse order using matching pairs.\n</think>\n({[]})(([(<>{})[][{}<>{}]]))"
}
```

## Dataset Format

The dataset uses JSONL (JSON Lines) format where:
- Each line is a valid JSON object
- Each object contains:
  - `prompt`: Instruction and partial sequence
  - `response`: Reasoning and completed sequence wrapped in `<think>` tags

## Bracket Types

The generator supports 4 bracket types:
- `()` - Parentheses
- `[]` - Square brackets
- `{}` - Curly braces
- `<>` - Angle brackets

## Algorithm

1. **Generate full sequence**: Create a valid Dyck sequence using a stack-based approach
2. **Create partial**: Randomly cut the sequence to create an incomplete input
3. **Compute stack**: Scan the partial sequence to find unmatched opening brackets
4. **Generate reasoning**: Create step-by-step explanation of the stack state
5. **Complete sequence**: Add closing brackets in reverse order of the stack

## Requirements

- Python 3.x
- Standard library only (no external dependencies)

## Customization

You can modify the generator behavior by editing `generator.py`:

- **Number of samples**: Change `n_samples` in the `__main__` block (default: 100,000)
- **Sequence length**: Modify `max_length` in `generate_dyck_sequence()` (default: 40)
- **Nesting depth**: Adjust `max_depth` in `generate_dyck_sequence()` (default: 5)
- **Output file**: Change `output_file` parameter in `generate_dataset()`

## Example Usage

```python
from generator import generate_dataset

# Generate 1000 samples
generate_dataset(n_samples=1000, output_file="my_dataset.jsonl")
```

## License

This project is provided as-is for research and educational purposes.
# Reasoning-Model
