import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os
import pickle

def keys_to_output(keys):
    '''
    Convert keys to a multi-hot array
    [A,W,D] boolean values.
    '''
    output = [0,0,0]
    if 'A' in keys:
        output[0] = 1   
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1

    return output

def main(training_data, file_name):
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    #last_time = time.time()
    while(True):
        # 800x600 windowed mode
        screen = grab_screen(region=(0, 40, 800, 640))
        last_time = time.time()
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB) #used to be COLOR_BGR2GRAY
        # resize to something a bit more acceptable for a CNN
        screen = cv2.resize(screen, (60,80))
        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([screen,output])
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        if len(training_data) % 200 == 0:
            print(len(training_data))
            np.save(file_name,training_data)
            # with open(file_name, "wb") as updated_file:
            #     pickle.dump(training_data, updated_file)

if __name__ == '__main__':
    file_name = 'training_data.npy'
    if os.path.isfile(file_name):
        print('File exists, loading previous data!')
        training_data = list(np.load(file_name, allow_pickle = True))
        # with open(file_name, 'rb') as original_file:
        #     pickle_var = pickle.load(original_file, binary = True) #wtf is wrong with this shit
        #     training_data = list(pickle_var)#np.load(file_name, allow_pickle=True))
    else:
        print('File does not exist, starting fresh!')
        training_data = []
    main(training_data, file_name)