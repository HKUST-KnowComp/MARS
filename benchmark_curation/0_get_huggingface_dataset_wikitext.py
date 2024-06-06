import os.path

import pandas as pd
from datasets import load_dataset

data = load_dataset("wikitext", "wikitext-103-raw-v1")

for split in ['train', 'test', 'validation']:
    for sample_ratio in [0.1, 0.2, 0.5, 1]:
        data_sample = data[split].select(range(int(len(data[split]) * sample_ratio)))
        if not os.path.exists('../../data/Wikitext'):
            os.makedirs('../../data/Wikitext')
            # save data_sample into a csv file
        data_sample.to_csv('../../event_discrimination_data/Wikitext/wikitext_{}_sample{}.csv'.format(split, int(sample_ratio * 100)))

length_min, length_max = 30, 200

if not os.path.exists('../../event_discrimination_data/Wikitext/wikitext_length{}-{}_filter'.format(length_min, length_max)):
    os.makedirs('../../event_discrimination_data/Wikitext/wikitext_length{}-{}_filter'.format(length_min, length_max))

for split in ['train', 'test', 'validation']:
    for sample_ratio in [0.1, 0.2, 0.5, 1]:
        data = pd.read_csv('../../event_discrimination_data/Wikitext/wikitext_{}_sample{}.csv'.format(split, int(sample_ratio * 100)),
                           index_col=None)
        # filter out the event_discrimination_data with string contains @
        # drop na
        data = data.dropna()
        data = data[~data['text'].str.contains('@')]
        data['length'] = data['text'].apply(lambda x: len(str(x).split()))
        # print distribution of length
        print(data['length'].value_counts())
        data = data[(data['length'] > length_min) & (data['length'] < length_max)]
        print(split, sample_ratio, len(data))
        data.to_csv(
            '../../event_discrimination_data/Wikitext/wikitext_length{}-{}_filter/wikitext_{}_sample{}_length{}-{}.csv'.format(length_min,
                                                                                                          length_max,
                                                                                                          split,
                                                                                                          int(sample_ratio * 100),
                                                                                                          length_min,
                                                                                                          length_max),
            index=False)
