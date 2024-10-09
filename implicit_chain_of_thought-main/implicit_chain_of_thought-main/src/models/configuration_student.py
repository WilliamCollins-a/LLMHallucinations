from transformers import PretrainedConfig

class StudentConfig(PretrainedConfig):
    def __init__(
        self,
        base_model='meta-llama/Llama-2-7b-hf',
        tokenizer_name='meta-llama/Llama-2-7b-hf',
        mixture_size=1,
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = tokenizer_name
        self.mixture_size = mixture_size
        super().__init__(**kwargs)
