import torch
from torch import nn
from transformers import AutoModel
from transformers import AutoTokenizer


class MetaphysicalEventDiscriminator(nn.Module):
    def __init__(self, model_name, pretrained_tokenizer_path=None):
        super().__init__()
        if pretrained_tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model_type = self.model.config.model_type

        try:
            self.emb_size = self.model.config.d_model  # bart
        except:
            self.emb_size = self.model.config.hidden_size  # roberta/bert/deberta/electra

        self.nn1 = nn.Linear(self.emb_size, 1024)
        self.nn2 = nn.Linear(1024, 512)
        self.nn3 = nn.Linear(512, 256)
        self.nn4 = nn.Linear(256, 128)
        self.nn5 = nn.Linear(128, 64)
        self.nn6 = nn.Linear(64, 2)

    def get_lm_embedding(self, tokens):
        """
            Input_ids: tensor (num_node, max_length)
            output:
                tensor: (num_node, emb_size)
        """
        outputs = self.model(tokens['input_ids'], attention_mask=tokens['attention_mask'])

        if 'bart' in self.model_type:
            # embedding of [EOS] (</s>) in the decoder
            eos_mask = tokens['input_ids'].eq(self.model.config.eos_token_id)
            if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            sentence_representation = outputs[0][eos_mask, :].view(outputs[0].size(0), -1, outputs[0].size(-1))[
                                      :, -1, :]
        else:
            # embedding of the [CLS] tokens
            sentence_representation = outputs[0][:, 0, :]
        return sentence_representation

    def forward(self, tokens):
        embeddings = self.get_lm_embedding(tokens)  # (batch_size, emb_size)
        x = torch.relu(self.nn1(embeddings))
        x = torch.relu(self.nn2(x))
        x = torch.relu(self.nn3(x))
        x = torch.relu(self.nn4(x))
        x = torch.relu(self.nn5(x))
        x = self.nn6(x)
        return x
