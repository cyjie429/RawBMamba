import random
import time
import os
import argparse
import torch
from tqdm import tqdm
import sys
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from mamba_ssm.models.ac_mamba import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
import feature_extract

# eavl_protocol path
eval_protocol = "/data2/xxx/data/ASVspoof2021DF1/ASVspoof2021_DF_eval/ASVspoof2021.DF.cm.eval.trl.txt"
wav_path = "/data2/xxx/data/ASVspoof2021DF/"


def set_seed(seed):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        #torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        use_cuda = True


class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
            '''self.list_IDs    : list of strings (each string: utt key),
               '''

            self.list_IDs = list_IDs
            self.base_dir = base_dir


    def __len__(self):
            return len(self.list_IDs)


    def __getitem__(self, index):
            key = self.list_IDs[index]
            x_inp = feature_extract.extract(self.base_dir+key+'.wav', 'fft')
            x_inp = torch.Tensor(x_inp)
            return x_inp,key


def genSpoof_list( dir_meta,is_train=False,is_eval=False):

    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
             _, key,_,_,label = line.strip().split(' ')
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list

    elif(is_eval):
        for line in l_meta:
            key= line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
             _, key,_,_,label = line.strip().split(' ')
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, pin_memory=True, num_workers=16)
    model.eval()
    
    fname_list = []
    score_list = []  
    with torch.no_grad():
        with tqdm(total = len(data_loader),desc = "processing test") as t:
            for batch_x,utt_id in data_loader:
                batch_size = batch_x.size(0)
                batch_x = batch_x.to(device)
                batch_out = model(batch_x)
                batch_out = batch_out[0].to("cpu")
                #batch_score = (batch_out[:, 1]
                #               ).data.cpu().numpy().ravel()
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
    print('Scores saved to {}'.format(save_path))



def main():
    parser = argparse.ArgumentParser(description = 'ASVspoof2021 evaluation')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=False, default='./models')
    parser.add_argument("-e", "--eval_output", type=str, help="score file destination", required=False, default='./21DF_eval.txt')
    parser.add_argument("--gpu", type=str, help="GPU index", default="1")
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--embedding', default=64, type=int,
                        help='embedding dim of transformer encoder')
    parser.add_argument('--heads', default=1, type=int,
                        help='number of heads of each transformer encoder layer')
    parser.add_argument('--posit', default='sine', type=str,
                        help='type of positional embedding')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    set_seed(args.seed)


    file_eval = genSpoof_list(dir_meta = eval_protocol, is_train=False,is_eval=True)
    print('no. of eval trials',len(file_eval))
    eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval, base_dir = wav_path)
    device = "cuda"
    dtype = torch.float32
    mamba_config = MambaConfig()
    model = MambaLMHeadModel(config=mamba_config, device=device, dtype=dtype).to(device)
    #for epoch in range(1, 33):
    #model.load_state_dict(torch.load(os.path.join(args.out_fold, 'checkpoint','senet_epoch_%d.pt' % epoch)))
    model.load_state_dict(torch.load(os.path.join(args.out_fold, 'rawbmamba_best.pt'), map_location='cuda:0'))
    produce_evaluation_file(eval_set, model, device, args.eval_output)





if __name__ == "__main__":
    main()
