from transformers import AutoConfig
from TransformerModel import Wav2Vec2ForNeuronData
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from toolbox.Dataloader_H5 import Dataset_h5_neuronInverter
import time
from toolbox.Plotter import Plotter_NeuronInverter
import argparse
from torch.utils.data import DataLoader

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


def get_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--facility", default='corigpu', type=str)
    parser.add_argument('--venue', dest='formatVenue', choices=['prod','poster'], default='prod',help=" output quality/arangement")
    parser.add_argument( "-X","--noXterm", dest='noXterm', action='store_true', default=True, help="disable X-term for batch mode")
    parser.add_argument("-o", "--outPath", default='/pscratch/sd/k/ktub1999/NeuronTransformer/plots/',help="output path for plots and tables")

    args = parser.parse_args()
    args.prjName='nif'
    # args.outPath+'/'
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args



if __name__ == '__main__':
    model_path="/pscratch/sd/k/ktub1999/NeuronTransformer/results/13284595/checkpoint-240000"
    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=19,
        )
    setattr(config, 'pooling_mode', "mean")
    model = Wav2Vec2ForNeuronData.from_pretrained(
        model_path,
        config=config
        )
    device=torch.cuda.current_device()  
    model.to(device)
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
    dataloader = DataLoader(train_data, batch_size= conf['local_batch_size'], shuffle=True)

    P = plots()
    
    num_samp=len(train_data)
    outputSize=19
    criterion =torch.nn.MSELoss().to(device)

    Uall=np.zeros([num_samp,outputSize],dtype=np.float32)
    Zall=np.zeros([num_samp,outputSize],dtype=np.float32)
    test_loss=0
    nEve=0
    nStep=0

    startT=time.time()
    with torch.no_grad():
        for item in dataloader:
            # if(data%100==0):
            #     print("Done",data)
            # item = dataloader.__getitem__(data)
            val = item["input_values"]
            lab = item["labels"]
            data_dev, target_dev = val.to(device), lab.to(device)
            # val.reshape((1,len(val)))
            # val=torch.from_numpy(val)
            # dataInp=val.float()
            # dataInp=torch.reshape(dataInp,(1,len(dataInp)))
            # print("DATA LENGTH",dataInp.shape)
            # arr1 = model(dataInp)[0].detach().numpy()
            # data_dev=data_dev.reshape((1,len(data_dev)))
            out=model(data_dev)[0]
            lossOp=criterion(out, target_dev)
            test_loss += lossOp.item()
            output=out.cpu()
            label = lab.cpu()
            nEve2=nEve+lab.shape[0]
            Uall[nEve:nEve2,:]=label[:]
            Zall[nEve:nEve2,:]=output[:]
            nEve=nEve2
            nStep+=1
            # print(type(arr1))
            arr2=lab
            
    test_loss /= nStep
    predTime=time.time()-startT
    sumRec={}
    sumRec['domain']='test'
    sumRec['jobId']=123
    sumRec['test'+'LossMSE']=float(test_loss)
    sumRec['predTime']=predTime
    sumRec['numSamples']=Uall.shape[0]
    sumRec['lossThrHi']=0.40  # for tagging plots
    sumRec['inpShape']="(4000,1)"
    sumRec['short_name']='Transformers'
    sumRec['modelDesign']='Transformers'
    sumRec['trainTime']=0.0
    sumRec['loss_valid']= 0
    sumRec['train_stims_select']= '[1]'
    sumRec['train_glob_sampl']= 50
    sumRec['pred_stims_select']= '[1]'
    sumRec['residual_mean_std']=1
    inpMD={
    }
    args=get_parser()
    
    inpMD['parName']=['gNaTs2_tbar_NaTs2_t_apical', 'gSKv3_1bar_SKv3_1_apical', 'gImbar_Im_apical',
        'gIhbar_Ih_dend', 'gNaTa_tbar_NaTa_t_axonal', 'gK_Tstbar_K_Tst_axonal', 'gNap_Et2bar_Nap_Et2_axonal',
        'gSK_E2bar_SK_E2_axonal', 'gCa_HVAbar_Ca_HVA_axonal', 'gK_Pstbar_K_Pst_axonal', 'gCa_LVAstbar_Ca_LVAst_axonal',
        'g_pas_axonal', 'cm_axonal', 'gSKv3_1bar_SKv3_1_somatic', 'gNaTs2_tbar_NaTs2_t_somatic',
        'gCa_LVAstbar_Ca_LVAst_somatic', 'g_pas_somatic', 'cm_somatic', 'e_pas_all']
    inpMD['num_phys_par']=19

    plot=Plotter_NeuronInverter(args,inpMD ,sumRec )
    plot.param_residua2D(Uall,Zall)
    figN='test'+'_'+ 'L5TTPC1i=0'+model_path.split('/')[-1]
    plot.display_all(figN, png=1)
# P.plot(arr1[0],arr2)
# P.savePlot()

# a = torch.rand(1,16000,dtype=torch.float)

# a_t=a.float()
# print(model(a_t))





# Show the plot
plt.savefig("testNewTransformer.png")