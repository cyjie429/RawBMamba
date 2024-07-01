import random
import time
import os
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import sys
# sys.path is searched in order.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from mamba_ssm.models.ac_mamba import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
import feature_extract

# eval_protocol
eval_protocol = "/data2/xxx/data/ASVspoof2021LA/CM/trial_metadata.txt"
wav_path = "/data2/xxx/data/ASVspoof2021LA/wav/"


class Dataset_ASVspoof2021_eval(Dataset):
	def __init__(self, list_IDs,allpath):
            '''self.list_IDs	: list of strings (each string: utt key),
               '''
               
            self.list_IDs = list_IDs
            self.allpath =allpath
            self.feature_type = "fft"

	def __len__(self):
            return len(self.list_IDs)


	def __getitem__(self, index):
            utt_id = self.list_IDs[index]
            each_data=self.allpath[utt_id]
            each_data=feature_extract.extract(each_data,self.feature_type)
            each_data=torch.Tensor(each_data)
            return each_data,utt_id  


def genSpoof_list( dir_meta,dir_path,is_train=False,is_eval=False):
    
    d_meta = {}
    label = {}
    seen = {}
    attacks = {}
    file_list=[]
    allpath = {}
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()
    '''
    if (is_train):
        for line in l_meta:

            line= line.strip().split()
            wav_id = line[0]
            wav_path  = line[2]
            label_tmp = line[1]
            allpath[wav_id] = wav_path
            file_list.append(wav_id)
            d_meta[wav_id] = 1 if label_tmp == 'real' else 0
        return d_meta,file_list,allpath
    
    elif(is_eval):'''
    for line in l_meta:
        line= line.strip().split()
        wav_id = line[1]
        
        file_list.append(wav_id)
        allpath[wav_id] = dir_path + line[1] + '.wav'
        label[wav_id] = line[4]
        seen[wav_id] = line[2]
        attacks[wav_id] = line[3]
    return file_list,allpath,label,seen,attacks
    '''else:
        for line in l_meta:
            line= line.strip().split()
            wav_id = line[0]
            wav_path  = line[2]
            label_tmp = line[1]
            allpath[wav_id] = wav_path
            file_list.append(wav_id)
            d_meta[wav_id] = 1 if label_tmp == 'real' else 0
        return d_meta,file_list,allpath'''
def produce_evaluation_file(dataset, model, device, save_path,label,seen,attacks):
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False,
    drop_last=False, pin_memory=True, num_workers=16)
    #data_loader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False, pin_memory=True, num_workers=64)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    
    fname_list = []
    key_list = []
    score_list = []

    s = time.time()
    with torch.no_grad():
        with tqdm(total = len(data_loader),desc = "processing test") as t:
            #t.set_description(desc = "processing testing")
            for batch_x,utt_id in data_loader:
                #fname_list = []
                #score_list = []  
                #batch_size = batch_x.size(0)
                batch_x = batch_x.to(device)
                
                batch_out = model(batch_x)
                batch_out = batch_out[0].to("cpu")
                batch_score = (batch_out[:, 1] - batch_out[:, 0] 
                               ).data.cpu().numpy().ravel() 
                # add outputs
                fname_list.extend(utt_id)
                score_list.extend(batch_score.tolist())
            
                t.update(1)
            with open(save_path, 'w') as fh:
                for f, cm in zip(fname_list,score_list):
                   fh.write('{} {}\n'.format(f, cm))
            fh.close()
    e = time.time()
    total = round(e - s)
    h = total // 3600
    m = (total - h * 3600) // 60
    s = total - 3600*h - 60*m
    print(f"time cost {h}:{m}:{s}")
     
    print('Scores saved to {}'.format(save_path))

def main():
    parser = argparse.ArgumentParser(description = 'ASVspoof2021 evaluation')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=False, default='./models/')
    parser.add_argument("-e", "--eval_output", type=str, help="score file destination", required=False, default='./models/21eval.txt')
    parser.add_argument("--gpu", type=str, help="GPU index", default="0,1")
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--embedding', default=64, type=int,
                        help='embedding dim of transformer encoder')
    parser.add_argument('--heads', default=1, type=int,
                        help='number of heads of each transformer encoder layer')
    parser.add_argument('--posit', default='sine', type=str,
                        help='type of positional embedding')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # print(os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # device = torch.device("cuda:{}".format(args.gpu) if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    device = "cuda"
    dtype = torch.float32

    file_eval,allpath,label,seen,attacks = genSpoof_list( dir_meta =  eval_protocol, dir_path = wav_path, is_train=False,is_eval=True)
    eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval, allpath =allpath)
    #model = se_resnet34(num_classes=2).to(device)
    #model = se_res2net50_v1b(num_classes=2).to(device)
    mamba_config = MambaConfig()
    model = MambaLMHeadModel(config=mamba_config, device=device, dtype=dtype).to(device)
    #model = ResNeXt50(num_classes=2).to(device)
    model.load_state_dict(torch.load(os.path.join(args.out_fold, "rawbmamba_best.pt")))
    produce_evaluation_file(eval_set, model, device, args.eval_output,label=label,seen=seen,attacks=attacks)




if __name__ == "__main__":
    main()
