import math
import os
import random
import pickle
import scipy.stats as st
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from wfdb import processing
from scipy import signal
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import json
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for np types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def data_filtering(data_dir):
    # 可以过滤不参与本次实验的record
    record_names = []
    for file in os.listdir(data_dir):
        if file[-4:] == '.dat':
            if not file[:-4] in record_names:
                record_names.append(file[:-4])
    return record_names


def divide_dataset(data_list, train_percentage=0.7):
    # 随机划分训练集和验证集，然后保存下
    nums = len(data_list)
    random.shuffle(data_list)
    train_len = int(math.ceil(nums * train_percentage))
    train_list = data_list[:train_len]
    test_list = data_list[train_len:]
    print('训练集共有{}个片段，验证集共有{}个片段'.format(train_len, nums - train_len))
    return train_list, test_list


def save_stft_fig(data, save_path, image_size=224, win_sz=100, overlap=50):
    img_size = (image_size, image_size)
    # dpi fix
    fig = plt.figure(frameon=False)
    dpi = fig.dpi
    # fig size / image size
    figsize = (image_size / dpi, image_size / dpi)

    for i in range(len(data)):
        for m in range(features):
            win = signal.windows.hann(win_sz)
            f, t, zxx = signal.stft(data[i][:, m], fs, window=win, nperseg=win_sz, noverlap=overlap, nfft=win_sz,
                                    return_onesided=True, boundary='zeros', padded=True, axis=- 1)

            # plt.figure(figsize=(33,21))
            fig = Figure()
            fig.subplots_adjust(0, 0, 1, 1)
            #         fig.add_axes([0,0,1,1])
            canvas = FigureCanvas(fig)
            ax = fig.gca()
            ax.pcolormesh(t, f, 20 * np.log10(np.abs(zxx)), shading='gouraud')
            ax.axis('off')
            canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            # #         if name%10 == 0:
            c = cv2.resize(image, img_size)
            plt.imsave(os.path.join(save_path, str(record_name) + "_stft_" + str(i) + '_' + str(m) + '.jpg'), c)
            # # 绘制二维图像测试
            # fig = Figure()
            # fig.subplots_adjust(0, 0, 1, 1)
            # #         fig.add_axes([0,0,1,1])
            # canvas = FigureCanvas(fig)
            # ax = fig.gca()
            # ax.plot(data[i][:, m])
            # ax.axis('off')
            # canvas.draw()
            # width, height = fig.get_size_inches() * fig.get_dpi()
            # image2 = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            # # #         if name%10 == 0:
            # c2 = cv2.resize(image2, img_size)
            # plt.imsave(os.path.join(save_path, str(record_name) + "_plt_" + str(i) + '_' + str(m) + '.jpg'), c2)


if __name__ == '__main__':
    data_root = "E:\\1_dataset\\CPSC\\train2"  # 数据路径
    Z_SCORE = True  # 对训练数据是否进行zscore归一化
    SAVE_FIG = True  # 存储训练信号图像100个
    test_percentage = 0.1  # 测试数据比例
    val_percentage = 0.1  # 验证数据比例
    features = 2  # 特征数目，用一条导联就是1，用两条导联就是2
    ann_suffix = 'atr'  # 标注文件的后缀
    db_save = 'E:\\1_dataset\\CPSC\\data_prepare_seg_test'  # 经过处理后保存数据路径
    min_peak_inter = 5  # 最少5次心跳 做一次判断
    mid_peak_inter = 2  # 中间点
    manual_seed = 777
    STEP = 165
    AUG_NUM = 3

    TRAIN = True
    if not os.path.exists(db_save):
        os.mkdir(db_save)
    random.seed(manual_seed)

    data_list = data_filtering(data_root)
    all_list = {}
    all_list["train"], all_list["test"] = divide_dataset(data_list, train_percentage=1 - test_percentage)

    # 首先计算平均心跳 便于取长
    if not STEP:
        means = []
        for record_name in all_list["train"]:
            record_path = os.path.join(data_root, record_name)
            ann_ref = wfdb.rdann(record_path, ann_suffix)
            beat_loc = np.array(ann_ref.sample)
            mean = np.mean(np.diff(beat_loc[1:-1]))
            means.append(mean)
        STEP = int(np.mean(means))
    sql_length = min_peak_inter * STEP

    all_list["train"], all_list["val"] = divide_dataset(all_list["train"], train_percentage=1 - val_percentage)

    for str_name in ["train", "val", "test"]:
        sum_n = 0
        sum_m = 0

        for record_name in tqdm(all_list[str_name]):
            xx = []
            yy = []
            lables = []
            record_path = os.path.join(data_root, record_name)
            sig, fields = wfdb.rdsamp(record_path)
            ann_ref = wfdb.rdann(record_path, ann_suffix)

            fs = fields['fs']
            length = len(sig)
            sample_descrip = fields['comments']

            print('record {}, length {}, type {}'.format(record_name, length, sample_descrip))

            beat_loc = np.array(ann_ref.sample)  # r-peak locations
            ann_note = np.array(ann_ref.aux_note)  # rhythm change flag

            # 将x进行zscore归一化
            if Z_SCORE:
                for i in range(sig.shape[1]):
                    sig[:, i] = st.zscore(sig[:, i])

            af_start_scripts = np.where((ann_note == '(AFIB') | (ann_note == '(AFL'))[0]
            af_end_scripts = np.where(ann_note == '(N')[0]
            if len(af_start_scripts) != len(af_end_scripts):
                print("record {} 标注af起始长度有误，已忽略".format(record_name))
                continue

            # 去掉小于5个心跳的af标注
            af_start = []
            af_end = []
            points = []
            if 'paroxysmal atrial fibrillation' in sample_descrip:
                for i in range(len(af_start_scripts)):
                    last_end = 0
                    if af_end_scripts[i] - af_start_scripts[i] < min_peak_inter:
                        print("record {}，起始点：{}-{} 标注af长度不足规定的心跳，已忽略".format(record_name, af_start_scripts[i],
                                                                             af_end_scripts[i]))
                        continue
                    if af_end_scripts[i] - last_end < min_peak_inter:
                        print(
                            "record {}，起始点：{}-{} 相邻AF之间停顿不足规定的心跳，已忽略".format(record_name, last_end, af_end_scripts[i]))
                        continue
                    last_end = af_end_scripts[i]
                    af_start.append(af_start_scripts[i])
                    af_end.append(af_end_scripts[i])
                    points.append([beat_loc[af_start_scripts[i]], beat_loc[af_end_scripts[i]]])
            elif 'persistent atrial fibrillation' in sample_descrip:
                points.append([0, length - 1])
            label_dict = {"predict_endpoints": points}


            # 每5个提取数据 固定长度
            left_boudany = beat_loc[mid_peak_inter]
            if "non atrial fibrillation" in sample_descrip or 'persistent atrial fibrillation' in sample_descrip:
                for i in range(len(beat_loc) - mid_peak_inter):
                    if i < mid_peak_inter:
                        continue
                    start_index = beat_loc[i] - left_boudany
                    end_index = start_index + sql_length
                    if end_index > length:
                        end_index = length
                        start_index = end_index - sql_length
                    data = sig[start_index:end_index]
                    if 'persistent atrial fibrillation' in sample_descrip:
                        label = 1
                    else:
                        label = 0
                    xx.append(data)
                    yy.append(label)
                #
            elif 'paroxysmal atrial fibrillation' in sample_descrip:
                for i in range(len(beat_loc) - mid_peak_inter):
                    if i < mid_peak_inter:
                        continue
                    for j in range(len(af_start)):
                        if i < af_start[j]:
                            # 落在非处
                            label_new = 0
                            start_index = 0
                        elif af_start[j] <= i <= af_end[j]:
                            #
                            start_index = af_start[j]
                            label_new = 1
                        elif i > af_end[-1]:
                            label_new = 0
                            start_index = af_start[j]
                        for k in range(AUG_NUM):
                            start_index_new = random.randint(start_index, beat_loc[i])
                            end_index_new = start_index_new + sql_length
                            if end_index_new > length:
                                end_index_new = length
                                start_index_new = end_index_new - sql_length
                            data_new = sig[start_index_new:end_index_new]
                            xx.append(data_new)
                            yy.append(label_new)
                        break

            # 准备处理保存数据
            root_path = os.path.join(db_save, str_name)
            if not os.path.exists(root_path):
                os.mkdir(root_path)
            save_path = os.path.join(root_path, record_name)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            # 保存label
            b = json.dumps(label_dict, cls=NumpyEncoder)
            with open(os.path.join(save_path, str(record_name) + '.json'), 'w') as f:
                f.write(b)
            # 保存數據
            xx = np.array(xx)
            yy = np.array(yy)

            if yy[0] ==0:
                sum_n = sum_n + len(xx)
            else:
                sum_m = sum_m + len(xx)
            if len(xx) != len(yy):
                print("err")
            if len(xx[0]) != sql_length:
                print("323223")
            for i in range(len(xx)):
                np.save(os.path.join(save_path, str(record_name) + "_sig_" + str(i) + '.npy'), xx[i])
            np.save(os.path.join(save_path, str(record_name) + '_label.npy'), yy)
            # 生成二维图片 准备训练
            # save_stft_fig(xx, save_path)
        print(str_name, sum_n, sum_m)