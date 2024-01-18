import os
import json
import shutil
import random
import argparse
from tqdm import tqdm

# you need to modify keypoint_class and bbox_class to match the yaml file
keypoint_class = ['uwb_tag']

bbox_class = {
    'uwb_device':0  
}

def process_single_json(labelme_path, save_folder):
    with open(labelme_path, 'r', encoding='utf-8') as f:
        labelme = json.load(f)

    img_width = labelme['imageWidth']
    img_height = labelme['imageHeight']

    suffix = labelme_path.split('.')[-2]
    yolo_txt_path = suffix + '.txt'

    with open(yolo_txt_path, 'w', encoding='utf-8') as f:

        for each_ann in labelme['shapes']:

            if each_ann['shape_type'] == 'rectangle':

                yolo_str = ''

                bbox_class_id = bbox_class[each_ann['label']]
                yolo_str += '{} '.format(bbox_class_id)

                bbox_top_left_x = int(min(each_ann['points'][0][0], each_ann['points'][1][0]))
                bbox_bottom_right_x = int(max(each_ann['points'][0][0], each_ann['points'][1][0]))
                bbox_top_left_y = int(min(each_ann['points'][0][1], each_ann['points'][1][1]))
                bbox_bottom_right_y = int(max(each_ann['points'][0][1], each_ann['points'][1][1]))

                bbox_center_x = int((bbox_top_left_x + bbox_bottom_right_x) / 2)
                bbox_center_y = int((bbox_top_left_y + bbox_bottom_right_y) / 2)

                bbox_width = bbox_bottom_right_x - bbox_top_left_x

                bbox_height = bbox_bottom_right_y - bbox_top_left_y

                bbox_center_x_norm = bbox_center_x / img_width
                bbox_center_y_norm = bbox_center_y / img_height

                bbox_width_norm = bbox_width / img_width

                bbox_height_norm = bbox_height / img_height

                yolo_str += '{:.5f} {:.5f} {:.5f} {:.5f} '.format(bbox_center_x_norm, bbox_center_y_norm, bbox_width_norm, bbox_height_norm)

                bbox_keypoints_dict = {}
                for each_ann in labelme['shapes']:
                    if each_ann['shape_type'] == 'point':
                        x = int(each_ann['points'][0][0])
                        y = int(each_ann['points'][0][1])
                        label = each_ann['label']
                        if (x>bbox_top_left_x) & (x<bbox_bottom_right_x) & (y<bbox_bottom_right_y) & (y>bbox_top_left_y):
                            bbox_keypoints_dict[label] = [x, y]

                for each_class in keypoint_class:
                    if each_class in bbox_keypoints_dict:
                        keypoint_x_norm = bbox_keypoints_dict[each_class][0] / img_width
                        keypoint_y_norm = bbox_keypoints_dict[each_class][1] / img_height
                        yolo_str += '{:.5f} {:.5f} {} '.format(keypoint_x_norm, keypoint_y_norm, 2)
                    else:
                        yolo_str += '0 0 0 '
                f.write(yolo_str + '\n')
                
    shutil.move(yolo_txt_path, save_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="labelme annotation to yolo dataset.")

    # input params
    parser.add_argument('data_path', type=str, help='dataset folder path to convert: /home/data_path')
    parser.add_argument('output_path', type=str, help='output folder path to save: /home/save_path')
    args = parser.parse_args()
    dataset_path = args.data_path
    output_path = args.output_path

    print("dataset path: ", dataset_path)
    print("output_path path: ", output_path)

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    if not os.path.isdir(os.path.join(output_path, 'images', 'train')):
        os.makedirs(os.path.join(output_path, 'images', 'train'))
    if not os.path.isdir(os.path.join(output_path, 'images', 'val')):
        os.makedirs(os.path.join(output_path, 'images', 'val'))
    if not os.path.isdir(os.path.join(output_path, 'labels', 'train')):
        os.makedirs(os.path.join(output_path, 'labels', 'train'))
    if not os.path.isdir(os.path.join(output_path, 'labels', 'val')):
        os.makedirs(os.path.join(output_path, 'labels', 'val'))

    # select train and val dataset
    os.chdir(dataset_path)
    
    json_files = list(filter(lambda x: '.json' in x, os.listdir()))
    json_ext = json_files[0].split(".")[-1]
    
    img_files = list(filter(lambda x: '.json' not in x, os.listdir()))
    img_ext = img_files[0].split(".")[-1]
    
    files_without_ext = [file.split(".")[0] for file in json_files]

    frac = 0.2  
    random.seed(123)
    val_num = int(len(files_without_ext) * frac)
    train_files = files_without_ext[val_num:]
    val_files = files_without_ext[:val_num]
    
    print("Total: ", len(files_without_ext))
    print("Train: ", len(train_files))
    print("Value: ", len(val_files))

    # copy to output file
    for train_file in tqdm(train_files):
        img_train_file = train_file + '.' + img_ext
        json_train_file = train_file + '.' + json_ext
        shutil.copy(img_train_file, os.path.join(output_path, 'images', 'train'))
        process_single_json(json_train_file, os.path.join(output_path, 'labels', 'train'))
    
    for val_file in tqdm(val_files):
        img_val_file = val_file + '.' + img_ext
        json_val_file = val_file + '.' + json_ext
        shutil.copy(img_val_file, os.path.join(output_path, 'images', 'val'))
        process_single_json(json_val_file, os.path.join(output_path, 'labels', 'val'))

    print("Successful!")
