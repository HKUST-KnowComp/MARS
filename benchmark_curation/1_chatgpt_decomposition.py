import json
import os
import sys
import warnings

import pandas as pd
from tqdm import tqdm

sys.path.append('../')
sys.path.append('../../')
from generate_with_openai import generate_with_openai

TEST_MODE = False

GENERATION_UPPER_LIMIT = 10000

CSV_PATH = [
    '../../event_discrimination_data/Wikitext/wikitext_length30-200_filter/wikitext_train_sample50_length30-200.csv',
    "../../event_discrimination_data/BookCorpus/bookcorpus_length30-200_filter/bookcorpus_train_sample10_length30-200.csv",
    '../../event_discrimination_data/Wikitext/wikitext_length30-200_filter/wikitext_validation_sample100_length30-200.csv',
    '../../event_discrimination_data/Wikitext/wikitext_length30-200_filter/wikitext_test_sample100_length30-200.csv',
]


def decomposition_event(text, verbose=False):
    prompt = """
    You are required to decompose the given long sentence into several short yet semantically complete events, each describing an action.
    An action event refers to those describing an action or a state change that occurs at a specific time and place.
    The key components of each event should be preserved: including the subject, verb, object, temporal and spatial quantifiers, numerical properties of the subject and objects, and sub-events.
    Generate one event as a whole sentence per line. You can generate as many events as you need. Below are some examples:
    Example:
    Sentence: The University of Colorado created the Department of Medicine and Surgery in September 1883 in the Old Main building on the Boulder campus. The Department of Nursing opened in 1898.
    Event 1: The University of Colorado created the Department of Medicine and Surgery in September 1883.
    Event 2: The Department of Medicine and Surgery was created in the Old Main building on the Boulder campus.
    Event 3: The Department of Nursing opened in 1898.
    
    Example:
    Sentence: In May 1934, following reports of a Japanese spy operating out of Dutch Harbor, the United States Navy dispatched Edwin T. Layton to the Aleutians to investigate the allegations. The result of this investigation was the arrest of the only Japanese man in the region, as well as the town 's only prostitute, a woman accused of conspiring with the Japanese man. During the 1930s, a number of United States governmental committees, boards and reports concluded that air bases in the Aleutians would be for the most part impractical due to the region's inclement weather.
    Event 1: In May 1934, there were reports of a Japanese spy operating out of Dutch Harbor.
    Event 2: The United States Navy dispatched Edwin T. Layton to the Aleutians to investigate the allegations.
    Event 3: As a result of the investigation, the only Japanese man in the region was arrested.
    Event 4: The town's only prostitute, a woman, was also arrested and accused of conspiring with the Japanese man.
    Event 5: During the 1930s, several United States governmental committees, boards, and reports concluded that air bases in the Aleutians would be mostly impractical.
    
    Follow the examples and decompose the sentence below.
    Sentence: {}\n"""
    success, result = generate_with_openai(prompt.format(text), verbose=verbose, max_tokens=300)
    if success:
        generation = result[0]
        cost = result[1]
        events = []
        for i in generation.split('\n'):
            colon_index = i.find(":")
            latter_part = i[colon_index + 1:].strip()
            events.append(latter_part)
        return events, cost
    else:
        print("Generation Failed")
        return None, prompt.format(text)


if TEST_MODE:
    warnings.warn("Test mode is enabled. Only the first 10 rows will be processed.")
for CSV in CSV_PATH:
    if not os.path.exists('./{}/'.format(CSV.split('/')[4])):
        os.makedirs('./{}/'.format(CSV.split('/')[4]))
    if not os.path.exists('./{}/{}'.format(CSV.split('/')[4], CSV.split('/')[-1].split('.')[0])):
        os.makedirs('./{}/{}'.format(CSV.split('/')[4], CSV.split('/')[-1].split('.')[0]))

    data = pd.read_csv(CSV, index_col=None)
    if 'length' in list(data):
        data = data.drop(columns=['length'])
    if TEST_MODE:
        data = data.head(10)
    if len(data) > GENERATION_UPPER_LIMIT:
        sampled_data = data.sample(GENERATION_UPPER_LIMIT, random_state=42)
        unsampled = data.drop(sampled_data.index)
        unsampled.to_csv('./{}/{}/unsampled.csv'.format(CSV.split('/')[4], CSV.split('/')[-1].split('.')[0]),
                         index=False)
        data = sampled_data.reset_index(drop=True)
    data_failed_generation = pd.DataFrame(columns=['text', 'prompt'])
    total_price = 0
    bar = tqdm(total=len(data), desc="Generating {}".format(CSV.split('/')[-1].split('.')[0]))
    for i in range(len(data)):
        sentence = data.loc[i, 'text']
        result, price = decomposition_event(sentence, verbose=TEST_MODE)
        if result:
            data.loc[i, 'generation'] = json.dumps(result)
            total_price += price
        else:
            data_failed_generation.loc[len(data_failed_generation)] = [sentence, price]
        bar.update(1)
        bar.set_postfix_str("Total Price: {}".format(total_price))
        # save for every 1000 generations
        if i % 1000 == 0:
            if not os.path.exists('./{}/'.format(CSV.split('/')[4])):
                os.makedirs('./{}/'.format(CSV.split('/')[4]))
            if not os.path.exists('./{}/{}'.format(CSV.split('/')[4], CSV.split('/')[-1].split('.')[0])):
                os.makedirs('./{}/{}'.format(CSV.split('/')[4], CSV.split('/')[-1].split('.')[0]))
            data.to_csv(
                './{}/{}/generation.csv'.format(CSV.split('/')[4], CSV.split('/')[-1].split('.')[0]), index=False)
            data_failed_generation.to_csv(
                './{}/{}/failed_generation.csv'.format(CSV.split('/')[4], CSV.split('/')[-1].split('.')[0]),
                index=False)
            # print("Total Price: {}".format(total_price))
            open('{}/{}/total_price.txt'.format(CSV.split('/')[4], CSV.split('/')[-1].split('.')[0]), 'w').write(
                "Total price is:" + str(total_price))

    data.to_csv(
        './{}/{}/generation.csv'.format(CSV.split('/')[4], CSV.split('/')[-1].split('.')[0]), index=False)
    data_failed_generation.to_csv(
        './{}/{}/failed_generation.csv'.format(CSV.split('/')[4], CSV.split('/')[-1].split('.')[0]),
        index=False)
    print("Total Price: {}".format(total_price))
    open('{}/{}/total_price.txt'.format(CSV.split('/')[4], CSV.split('/')[-1].split('.')[0]), 'w').write(
        "Total price is:" + str(total_price))
