import os
import numpy as np
import random


import re
import pandas as pd
from datetime import datetime, timedelta

import re
import pandas as pd
from datetime import datetime, timedelta

class Reader(object):
    def __init__(self, dataset_dir, listfile=None):
        self._dataset_dir = dataset_dir
        self._current_index = 0
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self._data = self._data[1:]

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)



class DecompensationReader(Reader):
    def __init__(self, dataset_dir, listfile=None):
        """ Reader for decompensation prediction task.
        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        # print(self._data[1])
        self._data = [(x, float(t), int(stay_id) ,int(y)) for (x, t, stay_id , y) in self._data]

    def _read_timeseries(self, ts_filename, time_bound):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                t = float(mas[0])
                if t > time_bound + 1e-6:
                    break
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Read the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Directory with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : int (0 or 1)
                Mortality within next 24 hours.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of examples (exclusive).")

        name = self._data[index][0]
        t = self._data[index][1]
        stay_id = self._data[index][2]
        y = self._data[index][3]
        (X, header) = self._read_timeseries(name, t)

        return {"X": X,
                "t": t,
                "stay_id" :  stay_id,
                "y": y,
                "header": header,
                "name": name}

class LengthOfStayReader(Reader):
    def __init__(self, dataset_dir, listfile=None):
        """ Reader for length of stay prediction task.

        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, float(t), int(stay_id) , float(y)) for (x, t, stay_id, y) in self._data]

    def _read_timeseries(self, ts_filename, time_bound):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                t = float(mas[0])
                if t > time_bound + 1e-6:
                    break
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : float
                Remaining time in ICU.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._data[index][1]
        stay_id = self._data[index][2]
        y = self._data[index][3]
        (X, header) = self._read_timeseries(name, t)

        return {"X": X,
                "t": t,
                "stay_id" : stay_id, 
                "y": y,
                "header": header,
                "name": name}

if __name__ == '__main__':

    # the way the data is read, is that all EHR data before the current timestep is included, but only the label for the current prediction time is read
    # so, t is the current time step, x contains everything before that, and name is that name of the file with all timsteps and labels  
    # note that the first part of the name is the subject_id not stay_id
    listfile = '/scratch/fs999/shamoutlab/data/mimic-iv-extracted/decompensation/train_listfile.csv'
    train_decomp_reader = DecompensationReader(dataset_dir='/scratch/fs999/shamoutlab/data/mimic-iv-extracted/decompensation/train', 
                                        listfile='/scratch/fs999/shamoutlab/data/mimic-iv-extracted/decompensation/train_listfile.csv')
    # train_los_reader = LengthOfStayReader(dataset_dir='/scratch/fs999/shamoutlab/data/mimic-iv-extracted/length-of-stay/train', 
    #                                     listfile='/scratch/fs999/shamoutlab/data/mimic-iv-extracted/length-of-stay/train_listfile.csv')

    print("Listfile: ", listfile)
    print("Number of examples: ", train_decomp_reader.get_number_of_examples())
    # print("Number of examples: ", train_los_reader.get_number_of_examples())
    icu_stay_metadata = pd.read_csv('/scratch/fs999/shamoutlab/data/mimic-iv-extracted/root/all_stays.csv')
    # for i in range(1,10):
    #     get_item[i] = train_decomp_reader.read_example(i)
    #     subject_id = re.match(r'\d+', get_item[i]['name']).group()
    #     result = icu_stay_metadata[icu_stay_metadata['subject_id'] == int(subject_id)]
    #     get_item[i]['intime'] = result[i]['intime'].values[0]
    
    total_examples = train_decomp_reader.get_number_of_examples()
    chunk_size = 5000
    
    for index in range(0, total_examples, chunk_size):
        chunk_end = min(index + chunk_size, total_examples)  # Ensure we don't go beyond the total number of examples
        chunk_indices = range(index, chunk_end)
        get_item_data = []
        for i in chunk_indices:
            item_data = train_decomp_reader.read_example(i)
            temp_name = item_data['name']
            subject_id = re.match(r'\d+', temp_name).group()
            item_data['subject_id'] = subject_id
            result = icu_stay_metadata[icu_stay_metadata['subject_id'] == int(subject_id)]
            intime = result.iloc[0]['intime']
            item_data['intime'] = intime
            
            hours = int(item_data['t'])
            date_string =  item_data['intime']
            date_format = "%Y-%m-%d %H:%M:%S"
            date_time = datetime.strptime(date_string, date_format)
            curr_time = date_time + timedelta(hours= hours)
            curr_time = curr_time.strftime(date_format)
            end_time = pd.to_datetime(curr_time)
        
        #     print("original:", date_string)
        #     print("new after adding {} hours:".format(hours), end_time)
            item_data['endtime'] = end_time
            
            get_item_data.append(item_data)
            print ("Done with: " , i)
    
    # Create DataFrame from the list of dictionaries
        get_item = pd.DataFrame(get_item_data)
        get_item.to_csv('updated_decomp_train_listfile.csv', columns= ['name' , 't' , 'stay_id' , 'y', 'intime', 'endtime'] , mode='a', index = False, header = False)  

