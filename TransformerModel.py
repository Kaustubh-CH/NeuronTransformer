from transformers import AutoModelForAudioClassification
import torch.nn as nn
import torch
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
from transformers import AutoConfig

class TransformerRegression(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # self.transform = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", num_labels=19)
        # self.dropout = nn.Dropout(0.3)
        # self.classifier = nn.Linear(in_features=self.transform.classifier.in_features, out_features=19)

    def forward(self,input, labels=None):
        x = input
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

class Wav2Vec2ForNeuronData(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.regress = TransformerRegression(config)

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            labels=None,
            attention_mask=None, #When using input with different sizes
            output_attentions=None, #to output the attentions
            output_hidden_states=None, 
    ):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )  

        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        output = self.regress(hidden_states)
        loss = None
        loss_fct = nn.MSELoss()
        if(labels!=None):
            loss = loss_fct(output.view(-1, self.num_labels), labels)

        output = (output,) + outputs[2:]
        return ((loss,) + output) if labels is not None else output
    
    



