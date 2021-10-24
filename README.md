# A ResNet-based model for AF location detection in ECG signals 

This repository is created for the [4th China Physiological Signal Challenge 2021](http://www.icbeb.org/CPSC2021).

The main idea of this method is to use a 5 beat size window to detect the AF event.

The model is based on a modified ResNet. We borrow and adapt code snippets from [this GitHub](https://github.com/lxdv/ecg-classification).

Due to the late entry into the competition, we didn't validate our model very well. But we will continue to improve the model in the future.  

## Dependencies
Dependencies are shown in the [requirements.txt](requirements.txt).

    - Python >= 3.8
    - Pytorch >= 1.7.0
    - wfdb 
    - ...

## Model
We upload our pretrained model link on the [OneDrive](https://portland-my.sharepoint.com/:u:/g/personal/shuomeng2-c_my_cityu_edu_hk/EWLV-8ljcohJh6UT8_s_S8UBkEQsw4_1Bfcm1VbDxklWxw?e=2JyVbd) (only trained 3 epoch 😂) 

## Getting Started!
Training:

1. Download the dataset, code and model.
2. Install all requirements in requirements.txt
3. Generate train data via `data_prepare.py`
4. Change the parameters in `run.py` for dataset path etc. 
5. Run `run.py`

Generate Json File for the Entry of CSPC2001

1. Install all requirements in requirements.txt
2. Download model and put it in the checkpoint directory and change the relevant parameters
3. `python entry_2021.py <test_path> <test_result_path>`

Get results 

1. `python score_2021.py <test_result_path> <result_save_path>`





