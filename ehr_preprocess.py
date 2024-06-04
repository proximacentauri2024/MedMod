#code_dir_medfuse = '/data/nasir/codes/shamoutlab/mml-images-ehr/med_fuse_4/med_fuse/'
import sys
#sys.path.append(f'{code_dir_medfuse}')
from ehr_utils.preprocessing import Discretizer, Normalizer
import numpy as np
import os


# data preprocessor
def read_timeseries(args):
    path = f'{args.ehr_data_root}/{args.task}/train/14991576_episode3_timeseries.csv'
    ret = []
    with open(path, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            ret.append(np.array(mas))
    return np.stack(ret)


def ehr_funcs(args):
    
    discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero',
                          config_path=f'/scratch/se1525/mml-ssl/ehr_utils/resources/discretizer_config.json')


    discretizer_header = discretizer.transform(read_timeseries(args))[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    normalizer_state = args.normalizer_state
    if normalizer_state is None:
        if args.task is 'mortality':
            normalizer_state = 'ihm_ts{}.input_str:previous.start_time:zero.normalizer'.format(args.timestep)
        elif args.task is 'decompensation':
            normalizer_state = 'decomp_ts{}.input_str:previous.n1e5.start_time:zero.normalizer'.format(args.timestep)
        elif args.task is 'length-of-stay':
            normalizer_state = 'los_ts{}.input_str:previous.start_time:zero.n5e4.normalizer'.format(args.timestep)
        else:
            normalizer_state = 'ph_ts{}.input_str:previous.start_time:zero.normalizer'.format(args.timestep)
        normalizer_state = os.path.join('/scratch/se1525/mml-ssl/', normalizer_state)
    normalizer.load_params(normalizer_state)
    
    return discretizer, normalizer
