import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast


class TextSummarizer:

    def __init__(self, device='cuda'):

        self._load_model(
            model_type="t5",
            model_dir=
            "./pretrained_models/flan-t5-large-finetuned-openai-summarize_from_feedback",
            device=device)

    def _load_model(self,
                    model_type: str = "t5",
                    model_dir: str = "outputs",
                    device: str = 'cuda'):
        """
        loads a checkpoint for inferencing/prediction
        Args:
            model_type (str, optional): "t5" or "mt5". Defaults to "t5".
            model_dir (str, optional): path to model directory.
                Defaults to "outputs".
            device (str, optional): device to run. Defaults to "cuda".
        """
        if model_type == "t5":
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"{model_dir}")
            self.tokenizer = T5TokenizerFast.from_pretrained(f"{model_dir}")
        else:
            raise NotImplementedError(
                f"model_type {model_type} not implemented")

        self.device = torch.device(device)

        self.model = self.model.to(self.device)

    def _predict(
        self,
        source_text: str,
        max_length: int = 512,
        num_return_sequences: int = 1,
        num_beams: int = 2,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ):
        """
        generates prediction for T5/MT5 model
        Args:
            source_text (str): any text for generating predictions
            max_length (int, optional): max token length of prediction.
                Defaults to 512.
            num_return_sequences (int, optional): number of predictions to be
                returned. Defaults to 1.
            num_beams (int, optional): number of beams. Defaults to 2.
            top_k (int, optional): Defaults to 50.
            top_p (float, optional): Defaults to 0.95.
            do_sample (bool, optional): Defaults to True.
            repetition_penalty (float, optional): Defaults to 2.5.
            length_penalty (float, optional): Defaults to 1.0.
            early_stopping (bool, optional): Defaults to True.
            skip_special_tokens (bool, optional): Defaults to True.
            clean_up_tokenization_spaces (bool, optional): Defaults to True.
        Returns:
            list[str]: returns predictions
        """
        input_ids = self.tokenizer.encode(
            source_text, return_tensors="pt", add_special_tokens=True)
        input_ids = input_ids.to(self.device)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
        )
        preds = [
            self.tokenizer.decode(
                g,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            ) for g in generated_ids
        ]
        return preds

    def __call__(self, source_text):
        generated_text = self._predict(source_text=source_text)
        return generated_text
