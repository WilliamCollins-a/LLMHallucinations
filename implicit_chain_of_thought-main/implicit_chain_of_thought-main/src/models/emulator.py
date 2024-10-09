import os
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .configuration_emulator import EmulatorConfig
import sys
sys.path.append("..")
from utils import get_sep_position
import logging

class Emulator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Change this line to use AutoModelForCausalLM for LLaMA 2
        self.base_model = AutoModelForCausalLM.from_pretrained(config.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        
        
        num_layers = self.base_model.config.num_hidden_layers  # Get the number of layers
        hidden_size = self.base_model.config.hidden_size  # Get the hidden size

        # Update the MLP layers
        self.mlps = nn.ModuleList([nn.Sequential(
             nn.Linear(2 * hidden_size, 4 * hidden_size),
             nn.ReLU(),
             nn.Linear(4 * hidden_size, hidden_size),
             ) for _ in range(num_layers)])

        self.mixture_components = nn.Embedding(config.mixture_size, hidden_size)
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1,
                batch_first=False, dropout=0, bidirectional=False)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size * 2, hidden_size)

    def eval(self):
        self.base_model.eval()

    def forward(self, input_ids, requires_backward=False):
        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        input_ids = input_ids[:, :sep_positions.max() + 1]
        
        outputs = self.base_model.forward(
            input_ids=input_ids,
            return_dict=True  # Ensure that the output is in a dictionary format
        )
        
        emulated_teacher_states = outputs.hidden_states  # Get hidden states from the outputs
        return emulated_teacher_states

    def compute_loss(self, input_ids, teacher_states):
        emulated_teacher_states = self.forward(input_ids=input_ids, requires_backward=True)
        batch_size = input_ids.shape[0]

        loss_fct = nn.MSELoss(reduction='none')
        loss = 0
        for teacher_state, emulated_teacher_state in zip(teacher_states, emulated_teacher_states):
            loss += loss_fct(teacher_state, emulated_teacher_state).sum(-1) / 2
        loss = loss.mean()
        outputs = CausalLMOutputWithCrossAttentions(loss=loss)
        outputs.total_loss = loss * batch_size
        return outputs

    @classmethod
    def from_pretrained(self, pretrained_path):
        config = EmulatorConfig.from_pretrained(pretrained_path)
        model = Emulator(config)
        state_dict = torch.load(os.path.join(pretrained_path, 'state_dict.bin'))
        try:
            model.load_state_dict(state_dict)
        except:
            model.load_state_dict(state_dict, strict=False)
            logging.warn("Some weights of the model Emulator checkpoint not loaded.")
        return model

    def save_pretrained(self, save_directory):
        print(f'Saving to {save_directory}')
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, 'state_dict.bin'))