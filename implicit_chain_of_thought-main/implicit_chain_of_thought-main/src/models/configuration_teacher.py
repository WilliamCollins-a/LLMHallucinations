from transformers import PretrainedConfig

class TeacherConfig(PretrainedConfig):
    def __init__(
        self,
        base_model='meta-llama/Llama-2-7b-hf',
        tokenizer_name='meta-llama/Llama-2-7b-hf',
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = tokenizer_name
        super().__init__(**kwargs)

