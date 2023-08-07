from transformers import AutoConfig
from TransformerModel import Wav2Vec2ForNeuronData
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from toolbox.Dataloader_H5 import Dataset_h5_neuronInverter

class plots():
    def __init__(self,):
        fig, axs = plt.subplots(5, 4, figsize=(12, 12))
        axs = axs.flatten() 
        self.fig=fig
        self.axs=axs
    def plot(self,arr1,arr2):
        for i in range(19):
            self.axs[i].scatter(arr1[i], arr2[i],color='green')
            self.axs[i].set_title(f"Scatter Plot {i+1}")
            self.axs[i].set_xlabel("Array 1")
            self.axs[i].set_ylabel("Array 2")
    def savePlot(self):
        self.fig.savefig("predictions.png")


config = AutoConfig.from_pretrained(
    "/pscratch/sd/k/ktub1999/NeuronTransformer/results/13284595/checkpoint-6400",
    num_labels=19,
    )
setattr(config, 'pooling_mode', "mean")
model = Wav2Vec2ForNeuronData.from_pretrained(
    "/pscratch/sd/k/ktub1999/NeuronTransformer/results/13284595/checkpoint-6400",
    config=config
    )

conf={
    'domain':'test'
    
}
conf['world_rank'] = 0
# conf['world_rank']=os.environ['SLURM_PROCID']
# conf['world_size']=int(os.environ['SLURM_NTASKS'])
conf['world_size']=1
conf['cell_name']="L5_TTPC1cADpyr0"
conf['shuffle']=True
conf['local_batch_size']=128
conf['data_path']='/pscratch/sd/k/ktub1999/bbp_May_08_19/'
conf['h5name']=os.path.join(conf['data_path'],conf['cell_name']+'.mlPack1.h5')
conf['probs_select']=[0]
conf['stims_select']=[0]
# conf['max_glob_samples_per_epoch']=50
train_data=Dataset_h5_neuronInverter(conf,0)


P = plots()
for data in range(len(train_data)):
    if(data%100==0):
        print("Done",data)
    item = train_data.__getitem__(data)
    val = item["input_values"]
    lab = item["labels"]
    # val.reshape((1,len(val)))
    val=torch.from_numpy(val)
    dataInp=val.float()
    dataInp=torch.reshape(dataInp,(1,len(dataInp)))
    # print("DATA LENGTH",dataInp.shape)
    arr1 = model(dataInp)[0].detach().numpy()
    
    # print(type(arr1))
    arr2=lab
    P.plot(arr1[0],arr2)

P.savePlot()

# a = torch.rand(1,16000,dtype=torch.float)

# a_t=a.float()
# print(model(a_t))





# Show the plot
plt.savefig("testNewTransformer.png")