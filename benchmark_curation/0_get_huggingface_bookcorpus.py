import os

import pandas as pd
from datasets import load_dataset

data = load_dataset("bookcorpus")

for split in ['train']:
    for sample_ratio in [0.1, 0.2, 0.5, 1]:
        data_sample = data[split].select(range(int(len(data[split]) * sample_ratio)))
        if not os.path.exists('../../data/BookCorpus'):
            os.makedirs('../../data/BookCorpus')
            # save data_sample into a csv file
        data_sample.to_csv('../../event_discrimination_data/BookCorpus/bookcorpus_{}_sample{}.csv'.format(split, int(sample_ratio * 100)))

length_min, length_max = 30, 200

if not os.path.exists('../../event_discrimination_data/BookCorpus/bookcorpus_length{}-{}_filter'.format(length_min, length_max)):
    os.makedirs('../../event_discrimination_data/BookCorpus/bookcorpus_length{}-{}_filter'.format(length_min, length_max))

for split in ['train']:
    for sample_ratio in [0.1, 0.2, 0.5, 1]:
        data = pd.read_csv('../../event_discrimination_data/BookCorpus/bookcorpus_{}_sample{}.csv'.format(split, int(sample_ratio * 100)),
                           index_col=None)
        data['length'] = data['text'].apply(lambda x: len(str(x).split()))
        # print distribution of length
        print(data['length'].value_counts())
        data = data[(data['length'] > length_min) & (data['length'] < length_max)]
        data.to_csv(
            '../../event_discrimination_data/BookCorpus/bookcorpus_length{}-{}_filter/bookcorpus_{}_sample{}_length{}-{}.csv'.format(
                length_min,
                length_max,
                split,
                int(sample_ratio * 100),
                length_min,
                length_max),
            index=False)
