import json
import os.path

import pandas as pd
from tqdm import trange

CSV_path = [
    '../3_component_abstraction_variation/wikitext_length30-200_filter/wikitext_train_sample50_length30-200/generation.csv',
    '../3_component_abstraction_variation/wikitext_length30-200_filter/wikitext_test_sample100_length30-200/generation.csv',
    '../3_component_abstraction_variation/wikitext_length30-200_filter/wikitext_validation_sample100_length30-200/generation.csv',
    '../3_component_abstraction_variation/bookcorpus_length30-200_filter/bookcorpus_train_sample10_length30-200/generation.csv'
]

for CSV in CSV_path:
    if not os.path.exists('./{}/'.format(CSV.split('/')[2])):
        os.makedirs('./{}/'.format(CSV.split('/')[2]))
    if not os.path.exists('./{}/{}'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0])):
        os.makedirs('./{}/{}'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]))

    data = pd.read_csv(CSV, index_col=None)
    data['generation'] = data['generation'].apply(lambda x: json.loads(x))
    data['key_components'] = data['key_components'].apply(lambda x: json.loads(x))
    data['text_id'] = data.index

    ensemble_data_dict = {
        'text_id': [],
        'text': [],
        'event_id': [],
        'event': [],
        'component_type': [],
        'component_original': [],
        'component_substitution': []
    }
    for i in trange(len(data)):
        text_id = data['text_id'][i]
        text = data['text'][i]
        for ind1, event in enumerate(data['generation'][i]):
            components2variation_dict = data['key_components'][i][event]
            for k in components2variation_dict:
                if components2variation_dict[k] == {}:
                    continue
                else:
                    for original in components2variation_dict[k].keys():
                        for ind2, substitution in enumerate(components2variation_dict[k][original]):
                            if original in event:
                                ensemble_data_dict['text_id'].append('text-{}'.format(text_id + 1))
                                ensemble_data_dict['text'].append(text)
                                ensemble_data_dict['event_id'].append('text-{}_event-{}'.format(text_id + 1, ind1 + 1))
                                ensemble_data_dict['event'].append(event)
                                ensemble_data_dict['component_type'].append(k)
                                ensemble_data_dict['component_original'].append(original)
                                ensemble_data_dict['component_substitution'].append(substitution)
                            else:
                                continue

    ensemble_data = pd.DataFrame(ensemble_data_dict)
    ensemble_data.to_csv('./{}/{}/ensemble.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]), index=False)
    print("length: ", len(ensemble_data),
          "saved to ./{}/{}/ensemble.csv".format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]))
    print("Number of unique event", len(ensemble_data['event'].unique()), "Number of unique text",
          len(ensemble_data['text_id'].unique()))
