# RawBMamba
This repository provides the overall framework for training and evaluating audio anti-spoofing systems proposed in [RawBMamba: End-to-End Bidirectional State Space Model for Audio Deepfake Detection](https://arxiv.org/abs/2406.06086).

### Mamba Installation
[Mamba](https://github.com/state-spaces/mamba)

### Training

To train RawBMamba:
```
python train.py -o ./save_path/
```
### Testing

To test RawBMamba on ASVspoof2019LA:
```
python ./ASVspoof2019LA_eval/19LA_test.py -o ./model/
```

To test RawBMamba on ASVspoof2021LA:
```
python ./ASVspoof2021LA_eval/21LA_test.py -o ./model/ -e ./model/21LA_eval.txt
```

To test RawBMamba on ASVspoof2021DF:
```
python ./ASVspoof2021DF_eval/21DF_test.py -o ./model/ -e ./model/21DF_eval.txt
```

### Result
We found that there is variance in model training, which can sometimes result in better outcomes than those reported in the paper. These are our experimental findings.
<table>
  <tr>
    <th rowspan="2">Models</th>
    <th colspan="2">19LA</th>
    <th colspan="2">21LA</th>
    <th colspan="1">21DF</th>
  </tr>
  <tr>
    <th>EER(%)</th>
    <th>t-DCF</th>
    <th>EER(%)</th>
    <th>t-DCF</th>
    <th>EER(%)</th>
  </tr>
  <tr>
    <td>ours</td>
    <td>1.19</td>
    <td>0.0360</td>
    <td>3.39</td>
    <td>0.2726</td>
    <td>15.85</td>
  </tr>
</table>

### Pre-trained models

We provide pre-trained RawBMamba. Run the following code in the root directory of RawBMamba-main, and remember to modify the file paths accordingly.

To evaluate RawBMamba on ASVspoof2019LA:
```
python ./ASVspoof2019LA_eval/evaluate.py 
```

To evaluate RawBMamba on ASVspoof2021LA:
```
bash ./ASVspoof2021LA_eval/evaluate.sh
```

To evaluate RawBMamba on ASVspoof2021DF:
```
bash ./ASVspoof2021DF_eval/evaluate.sh
```


### References

```bibtex
@inproceedings{liu2023leveraging,
  title={Leveraging positional-related local-global dependency for synthetic speech detection},
  author={Liu, Xiaohui and Liu, Meng and Wang, Longbiao and Lee, Kong Aik and Zhang, Hanyi and Dang, Jianwu},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}

@article{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

```

### Citation

If you use this codebase, or otherwise find our work valuable, please cite RawBMamba.
