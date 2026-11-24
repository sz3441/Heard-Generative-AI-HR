
import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


class SentimentPredictor:

    def __init__(self, adapter_path):

        config = PeftConfig.from_pretrained(adapter_path)
        base_model_name = config.base_model_name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            adapter_path
        )

        self.model.eval()

    def predict(self, aspect, review):

        messages = [
            {
                "role": "system",
                "content": (
                    "Analyze the employee review and determine "
                    "the sentiment for the specified aspect. "
                    "Choose one of: positive, neutral, negative. "
                    "Return only a JSON object."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Aspect: {aspect}\n"
                    f"Review: {review}"
                )
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        prediction = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        try:
            result = json.loads(prediction)

            if result.get("sentiment") not in [
                "positive",
                "neutral",
                "negative"
            ]:
                return {"sentiment": None, "raw_output": prediction}

            return result

        except json.JSONDecodeError:
            return {
                "sentiment": None,
                "raw_output": prediction
            }
