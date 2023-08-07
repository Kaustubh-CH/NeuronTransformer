from toolbox.Dataloader_H5 import Dataset_h5_neuronInverter
from transformers import  TimeSeriesTransformerModel,Trainer, TrainingArguments
from transformers import AutoFeatureExtractor,AutoModelForAudioClassification
import pandas as pd
import os
from torch.utils.data import  DataLoader
# from datasets import Dataset


from torch import nn
import torch
from transformers import Trainer
from typing import Any, Dict, Union

from torch.cuda.amp import autocast
from transformers import AutoConfig
from TransformerModel import Wav2Vec2ForNeuronData

from torch.utils.data import Dataset, DataLoader

conf={
    'domain':'train'
    
}
conf['world_rank'] = 0
# conf['world_rank']=os.environ['SLURM_PROCID']
# conf['world_size']=int(os.environ['SLURM_NTASKS'])
# print("TASKS",int(os.environ['SLURM_NTASKS']))
conf['world_size']=1
conf['cell_name']="L5_TTPC1cADpyr2"
conf['shuffle']=True
conf['local_batch_size']=16
conf['data_path']='/global/cfs/cdirs/m2043/balewski/neuronBBP3-10kHz_3pr_6stim/dec26_mlPack1/'
conf['h5name']=os.path.join(conf['data_path'],conf['cell_name']+'.mlPack1.h5')
conf['probs_select']=[0]
conf['stims_select']=[0]
conf['max_glob_samples_per_epoch']=150

def gen2(train_data):
    dic2={}
    for idx in range(len(train_data)):
      val, lab = train_data.__getitem__(idx)
      dic={}
      dic["input_values"]=val
      dic["label"]=lab
      yield dic

def gen3(train_data):
    dic2=[]
    for idx in range(len(train_data)):
      val, lab = train_data.__getitem__(idx)
      dic={}
      dic["input_values"]=val
      dic["labels"]=lab
      dic2.append(dic)
    return dic2

class RegressionTrainer(Trainer):
      def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        outputs = self._prepare_inputs(inputs)

        # if self.use_amp:
        #     with autocast():
        #         loss = self.compute_loss(model, inputs)
        # else:
        loss = self.compute_loss(model, outputs)

        # if self.args.gradient_accumulation_steps > 1:
            # loss = loss / self.args.gradient_accumulation_steps

        # if self.use_amp:
        #     self.scaler.scale(loss).backward()
        # elif self.use_apex:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # elif self.deepspeed:
        #     self.deepspeed.backward(loss)
        # else:
        loss.backward()

        return loss.detach()
      
      # def compute_loss(self,
      #                    model,
      #                    inputs,
      #                    return_outputs=False):
            
      #       print("HERERERERaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
      #       labels = inputs.get("labels")
      #       outputs = model(**inputs)
      #       logits = outputs.get('logits')
      #       loss_fct = nn.MSELoss()
      #       loss = loss_fct(logits.squeeze(), labels.squeeze())
      #       return (loss, outputs) if return_outputs else loss

class Training():

  

  def __init__(self):
    conf['domain']='train'
    self.train_data=Dataset_h5_neuronInverter(conf,1)
    # self.train_dataloader = DataLoader(self.train_data[10], batch_size=1, shuffle=True)
    
    # data_list =gen3(self.train_data)
    # len(data_list)
    # self.train_dataset=Dataset.from_pandas(pd.DataFrame(data=self.data_list))
    self.train_dataloader = DataLoader( self.train_data,
                          batch_size=16,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=torch.cuda.is_available())
    # self.train_ds = Dataset
    # for data in gen2(self.train_data):
    #   self.train_ds.add_item(data)
    # print(self.train_dataset)

    # self.hugging_training_data = Dataset.from_pandas(data_files=self.train_data.get_Map())
    # print(self.hugging_training_data)
    # self.train_data=self.train_data.shuffle(seed=42).select(range(1000))
    conf['domain']='valid'
    self.valid_data=Dataset_h5_neuronInverter(conf,1)
    self.valid_dataloader = DataLoader( self.valid_data,
                          batch_size=16,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=torch.cuda.is_available())
    # self.valid_ds = Dataset
    # data_list =gen3(self.valid_data)
    # self.valid_dataset=Dataset.from_pandas(pd.DataFrame(data=data_list))
    # for data in gen2(self.valid_data):
    #   data_list.append()
    # print(self.valid_ds)

    # self.train_data.__getitem__(1)
    # self.valid_data=self.valid_data.shuffle(seed=42).select(range(1000))

  # def pytorch_train(self):
  #    self.train_dataloader
  #    self.valid_dataloader
  #    model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", num_labels=19)
  #    from torch.optim import AdamW
  #    from transformers import get_scheduler
  #    from tqdm.auto import tqdm


  #    optimizer = AdamW(model.parameters(), lr=5e-5)
  #    num_epochs = 3
  #    num_training_steps = num_epochs * len(self.train_dataloader)
  #    lr_scheduler = get_scheduler(
  #    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
  #    )
  #    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  #    model.to(device)

  #    print("TRAINING STARTING")
  #    progress_bar = tqdm(range(num_training_steps))
  #    model.train()
  #    for epoch in range(num_epochs):
  #     for batch in self.train_dataloader:
  #       batch = {k: v.to(device) for k, v in batch.items()}
  #       outputs = model(**batch)
  #       loss = outputs.loss
  #       loss.backward()

  #       optimizer.step()
  #       lr_scheduler.step()
  #       optimizer.zero_grad()
  #       progress_bar.update(1)

     




 
  def train(self):
    print(len(self.train_data))
    
    
    loss_fct=nn.MSELoss()
    training_args = TrainingArguments(
    output_dir='./results/',
    # output_dir='./results/'+str(os.environ['SLURM_JOB_ID']),          # output directory
    num_train_epochs=3,              # total number of training epochs
    evaluation_strategy = "epoch",
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    report_to="wandb",
    )
    
    pooling_mode ="mean"
     
    config = AutoConfig.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=19,
    )
    setattr(config, 'pooling_mode', pooling_mode)

    model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", num_labels=19,ignore_mismatched_sizes=True)
    model = Wav2Vec2ForNeuronData.from_pretrained(
    "facebook/wav2vec2-base",
    config=config
    )
    
    # model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features=19)
    

        
    # TimeSeriesTransformerModel.from_pretrained("huggingface/time-series-transformer-tourism-monthly")
    # from transformers import Wav2Vec2Processor
    # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base",)
    # target_sampling_rate = processor.feature_extractor.sampling_rate
    # abc=[self.train_data[0]['input_values'],self.train_data[1]['input_values']]
    # result = processor(abc, sampling_rate=target_sampling_rate)
    # result['labels']=[list(range(19)),list(range(19))]
    # result=[result]

    trainer = RegressionTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=self.train_data,         # training dataset
    eval_dataset=self.valid_data             # evaluation dataset
    )
    trainer.train()
    trainer.save_model()
    

    


T = Training()
T.train()
