import os
import theconf
from theconf import Config as C
import json
from glob import glob
import torch
import numpy as np
from tqdm import tqdm

def preprocessing(flags):

    w_factor = flags.img_w / flags.grid_size
    h_factor = flags.img_h / flags.grid_size
    max_num = 0
    min_num = 100
    num = 0
    cnt = 0
    dirnames = ['train/normal', 'train/tailing', 'test/normal', 'test/tailing']
    if not os.path.exists(flags.out_dir):
        os.makedirs(flags.out_dir)

    for dirname in dirnames:
        raw_dir = os.path.join(flags.raw_data, dirname)
        out_dir = os.path.join(flags.out_dir, dirname)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        file_dirs = glob(os.path.join(raw_dir, '*/'))
        for file in tqdm(file_dirs):
            
            json_files = sorted(glob(os.path.join(file, 'json', '*.json')))

            max_num = max(max_num, len(json_files))
            min_num = min(min_num, len(json_files))
            num += len(json_files)
            cnt += 1
            json_len = len(json_files)

            if 'train' in file:
                out_filename = os.path.join(out_dir, os.path.basename(os.path.dirname(file)) + '.npy')           
                grid = torch.zeros(json_len + flags.blank, flags.grid_size, flags.grid_size)

                for idx, json_file in enumerate(json_files):
                    detection_result = json.load(open(json_file))
                    detected_person = []
                    for i, e in enumerate(detection_result['results'][0]['detection_result']):
                        if e['label'][0]['description'] in ['person']:
                            detected_person.append((int(e['position']['x']/w_factor), int(e['position']['y']/h_factor),
                                                    int(e['position']['w']/w_factor), int(e['position']['h']/h_factor)))
                    for i in range(len(detected_person)):
                        for j in range(detected_person[i][2]):
                            for k in range(detected_person[i][3]):
                                a, b = detected_person[i][1]+k, detected_person[i][0]+j
                                if a >= flags.grid_size:
                                    a = flags.grid_size - 1
                                if b >= flags.grid_size:
                                    b = flags.grid_size - 1
                                grid[idx + 1, a, b] = 127.5
                grid_np = grid.numpy()
                if grid_np.shape[0] >= 16:
                    np.save(out_filename, grid_np)

            elif 'test' in file :
                for start_idx in range(json_len - 16 + 1):
                    filename = os.path.basename(file)
                    out_dir = os.path.join(out_dir, filename)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    out_filename = os.path.join(out_dir, os.path.basename(os.path.dirname(file))+ '_' + str(start_idx).zfill(5) + '.npy')
                    grid = torch.zeros(16, flags.grid_size, flags.grid_size)
                    for idx in range(16):
                        detection_result = json.load(open(json_files[start_idx + idx]))
                        detected_person = []
                        for i, e in enumerate(detection_result['results'][0]['detection_result']):
                            if e['label'][0]['description'] in ['person']:
                                detected_person.append((int(e['position']['x']/w_factor), int(e['position']['y']/h_factor),
                                                        int(e['position']['w']/w_factor), int(e['position']['h']/h_factor)))
                        for i in range(len(detected_person)):
                            for j in range(detected_person[i][2]):
                                for k in range(detected_person[i][3]):
                                    a, b = detected_person[i][1]+k, detected_person[i][0]+j
                                    if a >= flags.grid_size:
                                        a = flags.grid_size - 1
                                    if b >= flags.grid_size:
                                        b = flags.grid_size - 1
                                    grid[idx, a, b] = 127.5
                    grid_np = grid.numpy()
                    if grid_np.shape[0] >= 16:
                        np.save(out_filename, grid_np)
    
    print(f'Max json: {max_num}')
    print(f'Min json: {min_num}')
    print(f'Mean json: {num / cnt}')
  
if __name__ == '__main__':
    parser = theconf.ConfigArgumentParser(conflict_handler='resolve')
    
    flags = parser.parse_args()

    preprocessing(flags)
