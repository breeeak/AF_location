import torch
import os
import json
import numpy as np
import scipy.stats as st
import wfdb
from wfdb import processing
from msmodel import MsModel
import sys



def data_filtering(data_dir):
    record_names = []
    for file in os.listdir(data_dir):
        if file[-4:] == '.dat':
            if not file[:-4] in record_names:
                record_names.append(file[:-4])
    return record_names

def get_json(test_path, results_path, model_path="./checkpoint/af_model_best.pth.tar"):
    min_interval = 5
    mid_peak_inter = 2  # 中间
    features = 2
    test_batch_size = 64


    if not os.path.exists(results_path):
        os.mkdir(results_path)
    record_list = data_filtering(test_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MsModel()
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    for record_name in record_list:
        predicts = []
        try:
            record_path = os.path.join(test_path, record_name)
            sig, fields = wfdb.rdsamp(record_path)
            for i in range(sig.shape[1]):
                sig[:, i] = st.zscore(sig[:, i])
            sig_len = len(sig)
            print("**********", record_name, sig_len)

            qrs_inds = processing.xqrs_detect(sig=sig[:, 1], fs=fields['fs'], verbose=False)
            # if sig_len < max_length:
            #     qrs_inds = processing.xqrs_detect(sig=sig[:, 0], fs=fields['fs'], verbose=False)
            # else:
            #     sigs = []
            #     qrs_inds = []
            #     for m in range(0, sig_len, max_length):
            #         sigs.append(sig[m:min(m + max_length, sig_len)])
            #     for seg_sig in sigs:
            #         sig_len = len(seg_sig)
            #         qrs_inds = len(qrs_inds) + processing.xqrs_detect(sig=seg_sig[:, 0], fs=fields['fs'], verbose=False)

            mean_step = int(np.mean(np.diff(qrs_inds)))
            data_list = []
            slide = mean_step * min_interval

            # 每5个提取数据 固定长度
            left_boudany = qrs_inds[mid_peak_inter]
            for i in range(len(qrs_inds) - mid_peak_inter):
                if i < mid_peak_inter:
                    continue
                start_index = qrs_inds[i] - left_boudany
                end_index = start_index + slide
                if end_index > sig_len:
                    end_index = sig_len
                    start_index = end_index - slide
                data = []
                for j in range(features):
                    data.extend(sig[start_index:end_index,j][:,1])
                data = np.expand_dims(data, axis=0)
                data_list.append(data)

            data_list = np.array(data_list)
            locations = []
            for i in range(0, len(data_list), test_batch_size):
                if i + test_batch_size > len(data_list):
                    inputs = torch.from_numpy(data_list[i:, :, :]).float().to(device)
                else:
                    inputs = torch.from_numpy(data_list[i:(i + test_batch_size), :, :]).float().to(device)
                outputs = model(inputs)
                pred = torch.argmax(outputs, dim=1)
                location = pred.cpu().numpy()
                locations.extend(location)


            if np.sum(locations) > len(locations) - min_interval:
                predicts.append([0, len(sig) - 1])
            elif np.sum(locations) > min_interval:
                # elif np.sum(locations) != 0:

                state_diff = np.diff(locations)
                start_r = np.where(state_diff == 1)[0] + 1
                end_r = np.where(state_diff == -1)[0] + 1

                if locations[0] == 1:
                    start_r = np.insert(start_r, 0, 0)
                if locations[-1] == 1:
                    end_r = np.insert(end_r, len(end_r), len(locations) - 1)
                af_start = []
                af_end = []
                # remove less than min_interval
                for i in range(len(start_r)):
                    last_end = 0

                    if end_r[i] - start_r[i] < min_interval:
                        last_end = end_r[i]
                        continue
                    if end_r[i] - last_end < min_interval:
                        continue
                    last_end = end_r[i]
                    af_start.append(start_r[i])
                    af_end.append(end_r[i])

                if len(af_start) > 0:
                    start_r = np.expand_dims(af_start, -1)
                    end_r = np.expand_dims(af_end, -1)
                    if af_start[0]<len(qrs_inds) and af_end[0]<len(qrs_inds):
                        start_end = np.concatenate((qrs_inds[start_r], qrs_inds[end_r]), axis=-1).tolist()
                        predicts.extend(start_end)
                    else:
                        print("end>lenqrs",start_r,end_r,len(qrs_inds))
        except Exception as e:
            print("record_name{},error{}".format(record_name,e))

        pred_dict = {'predict_endpoints': predicts}

        with open(os.path.join(results_path, record_name + '.json'), 'w') as json_file:
            json.dump(pred_dict, json_file, ensure_ascii=False)
        pass


if __name__ == '__main__':
    TESTSET_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    MODEL_PATH = sys.argv[3]
    get_json(TESTSET_PATH,RESULT_PATH,MODEL_PATH)

    # test_path = "E:\\1_dataset\\CPSC\\test"
    # results_path = "E:\\1_dataset\\CPSC\\test_results_seg"
    # model_path = ".\\checkpoint\\model_best.pth.tar"
    # get_json(test_path,results_path,model_path)



