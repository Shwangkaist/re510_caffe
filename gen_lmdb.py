import os
from os import walk
import pandas as pd
from os.path import join


def make_path_label_txt(mode, data_root):
    img_count = 0

    if mode == 'train':
        f_train = open('path_label_train.txt', 'w')
        for root, directories, files in os.walk(data_root):
            for file in files:
                if file[-4:] == ".ppm":
                    img_mode = root.split("/")[2]
                    if img_mode == mode:
                        label = root.split("/")[-1].lstrip('0')
                        data = os.path.join(root.replace(data_root,''), file) + " " + str(label) + '\n'
                        f_train.write(data)
                        print ("Train | Saving : ", data)
                        img_count += 1

        f_train.close()
        print("==> TRAIN path saved. Img #: ", img_count)

    elif mode == 'test':
        gt_path = data_root+'/test/GT-final_test.csv'
        df = pd.read_csv(gt_path, delimiter=';')
        print(df.columns)
        f_test = open('path_test.txt', 'w')
        for root, directories, files in os.walk(data_root):
            for file in files:
                    if file[-4:] == '.ppm':
                        img_mode = root.split("/")[2]
                        if img_mode == mode:
                            gt_label = int(df.loc[df['Filename'] == file]['ClassId'])
                            data = os.path.join(root.replace(data_root,''), file) + " " + str(gt_label) + '\n'
                            img_count += 1
                            print ("Test | Saving : ", data)
                            f_test.write(data)
        f_test.close()
        print("==> TEST path saved. Img #: ", img_count)

    elif mode == 'test_kaist':
        f_test = open('path_test_kaist.txt', 'w')
        for root, directories, files in os.walk(data_root):
            for file in files:
                    if file[-4:] == '.jpg':
                        img_mode = root.split("/")[2]
                        if img_mode == mode:
                            data = os.path.join(root.replace(data_root,''), file) + '\n'
                            img_count += 1
                            print ("Test_kaist | Saving : ", data)
                            f_test.write(data)
        f_test.close()
        print("==> TEST path saved. Img #: ", img_count)



def main():
    data_root = './data'
    # make_path_label_txt('train', data_root)
    # make_path_label_txt('test', data_root)
    make_path_label_txt('test_kaist', data_root)


if __name__ == '__main__':
    main()