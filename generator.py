# generator.py

import random
import uuid
import sys
import json

class Data:
    """Data object for task challenges"""

    def __init__(self, question: str, answer: str, difficulty: int = 1, metadata: dict = None, **kwargs):
        self.question = question
        self.answer = answer
        self.difficulty = difficulty
        self.metadata = metadata or {}
        self.gpt_response = ""

    def to_json(self):
        """Convert to JSON dict"""
        return {
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
            "gpt_response": self.gpt_response,
        }

    def to_json_str(self):
        """Convert to JSON string"""
        return json.dumps(self.to_json(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_dict):
        """Create from JSON dict"""
        instance = cls(
            question=json_dict.get("question", ""),
            answer=json_dict.get("answer", ""),
            difficulty=json_dict.get("difficulty", 1),
            metadata=json_dict.get("metadata", {})
        )
        if "gpt_response" in json_dict:
            instance.gpt_response = json_dict["gpt_response"]
        return instance

    @classmethod
    def from_json_str(cls, json_str):
        """Create from JSON string"""
        json_data = json.loads(json_str)
        return cls.from_json(json_data)


class DyckLanguageGenerator:
    """Generate Dyck language bracket matching tasks"""

    def __init__(self):
        self.brackets = [
            ("(", ")"),
            ("[", "]"),
            ("{", "}"),
            ("<", ">"),
            ("⟨", "⟩"),
            ("⟦", "⟧"),
            ("⦃", "⦄"),
            ("⦅", "⦆")
        ]

    def generate(
        self,
        seed: int,
        n_types: int = 6,
        total_length: int = 0,
        to_fill_length: int = 0,
        nesting_depth: int = 0,
        max_attempts: int = 1000
    ):
        """
        Generate a single Dyck language task

        Args:
            seed: Seed for reproducibility
            n_types: Number of bracket types (1-8)
            total_length: Total sequence length (0 for random)
            to_fill_length: Length to fill (0 for random)
            nesting_depth: Minimum nesting depth
            max_attempts: Max attempts to generate valid sequence

        Returns:
            Data: Game data object
        """
        # Validate parameters
        if n_types < 1 or n_types > 8:
            raise ValueError("n_types must be between 1 and 8")

        self.used_brackets = self.brackets[:n_types]
        rng = random.Random(seed)

        # Determine lengths
        current_total_length = total_length
        if current_total_length <= 0:
            current_total_length = rng.randint(20, 30) * 2
        elif current_total_length % 2 != 0:
            current_total_length -= 1

        current_fill_length = to_fill_length
        if current_fill_length <= 0:
            current_fill_length = rng.randint(
                max(1, int(current_total_length * 0.2)),
                min(int(current_total_length * 0.5), current_total_length // 2),
            )

        cut_point = current_total_length - current_fill_length

        # Generate sequence
        sequence = self._generate_valid_sequence(
            current_total_length, cut_point, nesting_depth, seed, max_attempts
        )

        question_sequence = sequence[:cut_point]
        closing_sequence = sequence[cut_point:]  # Only the closing brackets

        # Format question
        question = self._format_question(question_sequence)

        # Create Data object
        return Data(
            question=question,
            answer=closing_sequence,
            metadata={
                "seed": seed,
                "trace_id": str(uuid.uuid4()),
                "full_sequence": sequence,
                "question_sequence": question_sequence,
                "closing_sequence": closing_sequence,
                "n_types": n_types,
                "total_length": current_total_length,
                "fill_length": current_fill_length,
                "nesting_depth": nesting_depth,
            }
        )

    def _generate_valid_sequence(self, total_length, cut_point, nesting_depth, seed, max_attempts):
        rng = random.Random(seed) if seed is not None else random.Random()

        for attempt in range(max_attempts):
            try:
                if seed is not None:
                    rng.seed(seed + attempt)

                if total_length % 2 != 0:
                    total_length -= 1

                # Generate prefix (up to cut_point) with some unmatched opening brackets
                prefix = []
                prefix_stack = []
                current_depth = 0
                max_depth = 0

                if nesting_depth > 0:
                    for _ in range(nesting_depth):
                        bracket = rng.choice(self.used_brackets)
                        prefix.append(bracket[0])
                        prefix_stack.append(bracket)
                        current_depth += 1
                        max_depth = max(max_depth, current_depth)

                while len(prefix) < cut_point:
                    remaining_chars = cut_point - len(prefix)

                    if len(prefix_stack) == 0:
                        bracket = rng.choice(self.used_brackets)
                        prefix.append(bracket[0])
                        prefix_stack.append(bracket)
                        current_depth += 1
                        max_depth = max(max_depth, current_depth)
                    elif remaining_chars > len(prefix_stack):
                        if rng.random() < 0.6:
                            bracket = rng.choice(self.used_brackets)
                            prefix.append(bracket[0])
                            prefix_stack.append(bracket)
                            current_depth += 1
                            max_depth = max(max_depth, current_depth)
                        else:
                            bracket = prefix_stack.pop()
                            prefix.append(bracket[1])
                            current_depth -= 1
                    else:
                        bracket = rng.choice(self.used_brackets)
                        prefix.append(bracket[0])
                        prefix_stack.append(bracket)
                        current_depth += 1
                        max_depth = max(max_depth, current_depth)

                suffix = []
                while prefix_stack:
                    bracket = prefix_stack.pop()
                    suffix.append(bracket[1])

                result = "".join(prefix + suffix)

                if len(result) != total_length:
                    raise ValueError(f"Invalid length: {len(result)} != {total_length}")
                if nesting_depth > 0 and max_depth < nesting_depth:
                    raise ValueError("Insufficient nesting depth")
                if len(prefix) != cut_point:
                    raise ValueError(f"Invalid cut point: {len(prefix)} != {cut_point}")

                return result

            except ValueError:
                continue

        raise ValueError("Failed to generate valid sequence")

    def _format_question(self, sequence):
        return f"""Complete the following Dyck language sequence by adding the minimal necessary closing brackets.

Sequence: {sequence}

Rules:
- Add only the closing brackets needed to match all unmatched opening brackets
- Do not add any extra bracket pairs beyond what is required

Provide only the complete valid sequence."""


if __name__ == "__main__":
    # Check command-line argument for seed
    if len(sys.argv) < 2:
        print("Usage: python generator.py <seed>")
        sys.exit(1)

    seed = int(sys.argv[1])

    # Create Dyck Language Generator instance
    generator = DyckLanguageGenerator()

    # Generate the task
    task = generator.generate(
        seed=seed,
        n_types=6,  # Number of bracket types
        total_length=40,  # Total sequence length
        to_fill_length=20,  # Length to fill
        nesting_depth=3,  # Minimum nesting depth
        max_attempts=1000  # Max attempts to generate a valid sequence
    )

    # Print the task as JSON
    print("Generated Dyck Language Task (in JSON format):")
    print(task.to_json_str())