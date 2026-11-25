import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, set_seed


class GPT2TextGenerator:
    name = "GPT-2"

    def __init__(self, model_name: str = "gpt2"):
        """
        Loads the GPT-2 model and BPE tokenizer onto GPU (if available) or CPU.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 30, seed: int = 42) -> str:
        """
        Generates a continuation for the given raw text prompt using fixed
        sampling settings and a fixed seed for reproducibility.
        Returns the prompt plus its continuation as plain text.
        """
        set_seed(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    generator = GPT2TextGenerator("gpt2")
    sample_prompt = "The meaning of"

    # Run twice to verify reproducibility
    out1 = generator.generate(sample_prompt, max_new_tokens=30)
    out2 = generator.generate(sample_prompt, max_new_tokens=30)

    print(f"Model: {generator.name}")
    print(f"Device: {generator.device}")
    print(f"Output 1:\n{out1}\n")
    print(f"Output 2:\n{out2}\n")
    assert out1 == out2, "Outputs should match for fixed seed!"
