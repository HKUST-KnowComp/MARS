import json
import os.path
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
    '../1_event_decomposition/bookcorpus_length30-200_filter/bookcorpus_train_sample10_length30-200/generation.csv',
    '../1_event_decomposition/wikitext_length30-200_filter/wikitext_train_sample50_length30-200/generation.csv',
    '../1_event_decomposition/wikitext_length30-200_filter/wikitext_test_sample100_length30-200/generation.csv',
    '../1_event_decomposition/wikitext_length30-200_filter/wikitext_validation_sample100_length30-200/generation.csv',
]


def extract_key_components(sentence, verbose=False):
    prompt = """Given a short event, extract these components:
        1. Subject: The noun that performs the action in the sentence.
        2. Verb: The action word in the sentence.
        3. Object: The noun that receives the action of the verb.
        4. Temporal Quantifier: The time or time period of the event in the sentence.
        5. Spatial Quantifier: The location or spatial extent of the event in the sentence.
        6. Numerical Quantities and Properties: Numerical values describing the number or properties of the subject, object, or sub-events.
        7. Sub-events: Complete events that are part of the main event in the sentence.
        For each component, if there are more than one, separate them with |. If you cannot find one for a component, generate ``None'' only. Below are some examples:
        Event: The University of Colorado created the Department of Medicine and Surgery in September 1883 in the Old Main building on the Boulder campus.
        Subject: University of Colorado
        Verb: created
        Object: Department of Medicine and Surgery
        Temporal Quantifier: September 1883
        Spatial Quantifier: Old Main building on the Boulder campus
        Numerical Quantities and Properties: None
        Sub-events: The University of Colorado created the Department of Medicine and Surgery in September 1883.
        
        Event: After the First Battle of Naktong Bulge, the US Army's 2nd Infantry Division was moved to defend the Naktong River line.
        Subject: US Army's 2nd Infantry Division
        Verb: moved | defend
        Object: None
        Temporal Quantifier: After the First Battle of Naktong Bulge
        Spatial Quantifier: Naktong River line
        Numerical Quantities and Properties: None
        Sub-events: The US Army's 2nd Infantry Division was moved | The US Army's 2nd Infantry Division was moved to defend the Naktong River line.
        
        Event: {}"""

    prompted_event = prompt.format(sentence)
    success, result = generate_with_openai(prompted_event, verbose=verbose)
    if success:
        result_dict = {
            "Subject": [],
            "Verb": [],
            "Object": [],
            "Temporal Quantifier": [],
            "Spatial Quantifier": [],
            "Quantities and Properties of Objects": [],
            "Sub-events": []
        }
        for line in result[0].split("\n"):
            if any([i in line for i in ["Subject", "Verb", "Object", "Temporal Quantifier", "Spatial Quantifier",
                                        "Quantities and Properties of Objects", "Sub-events"]]):
                try:
                    category, content = line.split(": ")
                    gen_list = content.split(" | ") if content != "None" else []
                    result_dict[category.split('.')[-1].strip()] = [i.lower() for i in gen_list if
                                                                    i.lower().strip() != "none"]
                except ValueError as e:
                    print(e, line)
            else:
                continue
        return result_dict, result[1]
    else:
        print("Generation Failed")
        return None, prompted_event


if TEST_MODE:
    warnings.warn("Test mode is enabled. Only the first 3 rows will be processed.")
for CSV in CSV_PATH:
    if not os.path.exists('./{}/'.format(CSV.split('/')[2])):
        os.makedirs('./{}/'.format(CSV.split('/')[2]))
    if not os.path.exists('./{}/{}'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0])):
        os.makedirs('./{}/{}'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]))

    data = pd.read_csv(CSV, index_col=None)
    if 'length' in list(data):
        data = data.drop(columns=['length'])
    if TEST_MODE:
        data = data.head(3)
    # if len(event_discrimination_data) > GENERATION_UPPER_LIMIT:
    #     sampled_data = event_discrimination_data.sample(GENERATION_UPPER_LIMIT, random_state=42)
    #     unsampled = event_discrimination_data.drop(sampled_data.index)
    #     unsampled.to_csv('./{}/{}/unsampled.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]),
    #                      index=False)
    #     event_discrimination_data = sampled_data.reset_index(drop=True)
    data = data.dropna(subset=['generation']).reset_index(drop=True)
    data_failed_generation = pd.DataFrame(columns=['text', 'generation', 'prompt'])
    total_price = 0
    progress_bar = tqdm(total=sum([len(json.loads(data.loc[i, 'generation'])) for i in range(len(data))]),
                        desc="Extracting {}".format(CSV.split('/')[3].split('.')[0]))
    for i in range(len(data)):
        events = json.loads(data.loc[i, 'generation'])
        event_components_dict = {e: None for e in events}
        for e in events:
            result, price = extract_key_components(e, verbose=TEST_MODE)
            if result:
                event_components_dict[e] = result
                total_price += price
            else:
                data_failed_generation.loc[len(data_failed_generation)] = [data.loc[i, 'text'],
                                                                           data.loc[i, 'generation'], price]
            progress_bar.update(1)
            progress_bar.set_postfix({"Total Price (HKD)": total_price,
                                      "Estimated Total Price": total_price / i * len(data) if i != 0 else 0})
        data.loc[i, 'key_components'] = json.dumps(event_components_dict)
        if i % 1000 == 0:
            data.to_csv(
                './{}/{}/generation.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]), index=False)
            data_failed_generation.to_csv(
                './{}/{}/failed_generation.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]),
                index=False)
            open('{}/{}/total_price.txt'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]), 'w').write(
                "Total price is:" + str(total_price))
    data.to_csv(
        './{}/{}/generation.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]), index=False)
    data_failed_generation.to_csv(
        './{}/{}/failed_generation.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]),
        index=False)
    print("Total Price: {}".format(total_price))
    open('{}/{}/total_price.txt'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]), 'w').write(
        "Total price is:" + str(total_price))
