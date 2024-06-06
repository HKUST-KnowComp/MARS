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

GENERATION_UPPER_LIMIT = 5000

CSV_PATH = [
    '../2_event_component_extraction/wikitext_length30-200_filter/wikitext_train_sample50_length30-200/generation.csv',
    '../2_event_component_extraction/bookcorpus_length30-200_filter/bookcorpus_train_sample10_length30-200/generation.csv',
    '../2_event_component_extraction/wikitext_length30-200_filter/wikitext_validation_sample100_length30-200/generation.csv',
    '../2_event_component_extraction/wikitext_length30-200_filter/wikitext_test_sample100_length30-200/generation.csv'
]


def conceptualize_component(event, component_type, component):
    assert component_type in ["Subject", "Verb", "Object", "Temporal Quantifier", "Spatial Quantifier",
                              "Quantities and Properties of Objects", "Sub-events"]
    if component_type == 'Subject':
        prompt = """Given an event and a subject within the event, abstract the given subject in the given sentence into three different concepts. Each concept should be more abstract than the previous one. It doesn't have to fit into the original context. You are encouraged to be creative and generate concepts that even deviates from the given context. For example:
                Event: World's leading scientists announce breakthrough in clean energy technology, revolutionizing global sustainability efforts | Subject: World's leading scientists | Concepts: expert, human, organism
                Event: The cat is sleeping on the couch | Subject: The cat | Concepts: feline, living being, organism
                Event: A driver is speeding down the highway | Subject: A driver | Concepts: person, carbon-based life, entity
                Event: {} | Subject: {} | Concepts:""".format(event, component)
        success, result = generate_with_openai(prompt)
        if success:
            return [i.strip() for i in result[0].split(',')], result[1]
        else:
            return None, prompt
    elif k == "Verb":
        prompt = """Given an event and a verb within the event, abstract the given verb in the given sentence into three different concepts. Each concept should be more abstract than the previous one. It doesn't have to fit into the original context. You are encouraged to be creative and generate concepts that even deviates from the given context. For example:
                Event: World's leading scientists announce breakthrough in clean energy technology, revolutionizing global sustainability efforts | Verb: announce | Concepts: declare, speak, communicate
                Event: The cat is sleeping on the couch | Verb: sleeping | Concepts: resting, acting, behaving
                Event: A driver is speeding down the highway | Verb: speeding down | Concepts: moving, changing, transforming
                Event: {} | Verb: {} | Concepts:""".format(event, component)
        success, result = generate_with_openai(prompt)
        if success:
            return [i.strip() for i in result[0].split(',')], result[1]
        else:
            return None, prompt
    elif k == "Object":
        prompt = """Given an event and an object within the event, abstract the given object in the given sentence into three different concepts. Each concept should be more abstract than the previous one. It doesn't have to fit into the original context. You are encouraged to be creative and generate concepts that even deviates from the given context. For example:
                Event: World's leading scientists announce breakthrough in clean energy technology, revolutionizing global sustainability efforts | Object: breakthrough | Concepts: innovation, advancement, shift
                Event: The cat is sleeping on the couch | Object: couch | Concepts: furniture, object, item
                Event: A driver is speeding down the highway | Object: highway | Concepts: road, infrastructure, transportation
                Event: {} | Object: {} | Concepts:""".format(event, component)
        success, result = generate_with_openai(prompt)
        if success:
            return [i.strip() for i in result[0].split(',')], result[1]
        else:
            return None, prompt
    elif k == "Temporal Quantifier":
        prompt = """Given an event and a temporal quantifier within the event, variate the given temporal quantifier into three different values. One value should be shorter than the original and the other two should be longer. You are encouraged to be creative and generate values that even deviates from the given context. For example:
                    Event: The old farmer has been working on his farm for 10 years | Temporal Quantifier: 10 years | Variations: 1 day, 100 years, eternal life
                    Event: He was a very successful merchant in Tang Dynasty | Temporal Quantifier: in Tang Dynasty | Variations: during the Song Dynasty, in nowadays, for a thousand years
                    Event: On the morning of September 1, the 1st and 2nd Regiments of the NK 9th Division launched their first offensive of the war | Temporal Quantifier: On the morning of September 1 | Variations: At a second in the morning, throughout september, in recent 10 years
                    Event: {} | Temporal Qualifier: {} | Variations:""".format(event, component)
        success, result = generate_with_openai(prompt)
        if success:
            return [i.strip() for i in result[0].split(',')], result[1]
        else:
            return None, prompt
    elif k == "Spatial Quantifier":
        prompt = """Given an event and a spatial quantifier within the event, variate the given spatial quantifier into three different values. One value should be lower than the original and the other two should be higher. You are encouraged to be creative and generate values that even deviates from the given context. For example:
                    Event: The cat is sleeping on the couch | Spatial Quantifier: on the couch | Variations: on the floor, on the roof, in the sky
                    Event: The old farmer has been working on his farm for 10 years | Spatial Quantifier: on his farm | Variations: on the ground, in the water, in the space
                    Event: The children are playing wildly in the park in Seattle | Spatial Quantifier: in the park in Seattle | Variations: in the garden in Seattle, in the city in Seattle, in the country in Seattle
                    Event: {} | Spatial Quantifier: {} | Variations:""".format(event, component)
        success, result = generate_with_openai(prompt)
        if success:
            return [i.strip() for i in result[0].split(',')], result[1]
        else:
            return None, prompt
    elif k == "Quantities and Properties of Objects":
        prompt = """Given an event and a quantity or property of an object within the event, variate the given quantity or property into three different values. One value should be smaller and the other two should be larger. You are encouraged to be creative and generate values that even deviates from the given context. For example:
                    Event: A 10 centimeters worm is chasing a butterfly  | Quantity or Property: 10 centimeters | Variations: 1 millimetre, 10 meters, 1000 kilometers
                    Event: A white woman is chasing the police on the street | Quantity or Property: white | Variations: black, yellow, red
                    Event: An old farmer has strong muscles due to his hardworking | Quantity or Property: strong | Variations: weak, powerful, invincible
                    Event: {} | Quantity or Property: {} | Concepts:""".format(event, component)
        success, result = generate_with_openai(prompt)
        if success:
            return [i.strip() for i in result[0].split(',')], result[1]
        else:
            return None, prompt
    elif k == "Sub-events":
        prompt = """Given an event and a sub-event within the event, abstract the given sub-event into three different concepts. Each concept should be more abstract than the previous one. You are encouraged to be creative and generate very abstract concepts that even deviates from the given context. For example:
                    Event: The couple is hosting their wedding in the hotel and celebrating their love with friends and family | Sub-event: The couple is celebrating their love | Concepts: celebration, love, joyfulness
                    Event: The team is working together to develop a new software application, which is expected to revolutionize the industry | Sub-event: The software is expected to revolutionize the industry | Concepts: revolution, change, expectation
                    Event: The students are participating in a science fair and presenting their innovative projects to judges and visitors | Sub-event: The students are presenting their innovative projects | Concepts: sharing, speaking, expressing
                    Event: {} | Sub-event: {} | Concepts:""".format(event, component)
        success, result = generate_with_openai(prompt)
        if success:
            return [i.strip() for i in result[0].split(',')], result[1]
        else:
            return None, prompt


if TEST_MODE:
    warnings.warn("Test mode is enabled. Only the first 2 rows will be processed.")
for CSV in CSV_PATH:
    if not os.path.exists('./{}/'.format(CSV.split('/')[2])):
        os.makedirs('./{}/'.format(CSV.split('/')[2]))
    if not os.path.exists('./{}/{}'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0])):
        os.makedirs('./{}/{}'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]))

    data = pd.read_csv(CSV, index_col=None)
    if 'length' in list(data):
        data = data.drop(columns=['length'])
    if TEST_MODE:
        data = data.head(2)
    print(len(data))
    if len(data) > GENERATION_UPPER_LIMIT:
        sampled_data = data.sample(GENERATION_UPPER_LIMIT, random_state=42)
        unsampled = data.drop(sampled_data.index)
        unsampled.to_csv('./{}/{}/unsampled.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]),
                         index=False)
        data = sampled_data.reset_index(drop=True)
    data = data.dropna(subset=['key_components']).reset_index(drop=True)
    data_failed_generation = pd.DataFrame(columns=['text', 'generation', 'key_components', 'prompt'])
    total_price, total_data = 0, 0
    for i in range(len(data)):
        key_components = json.loads(data.loc[i, 'key_components'])
        for event, components in key_components.items():
            if not components:
                continue
            for k in components:
                total_data += len(components[k])
    progress_bar = tqdm(total=total_data, desc="Conceptualizing {}".format(CSV.split('/')[3].split('.')[0]))
    total = 0
    for i in range(len(data)):
        events = json.loads(data.loc[i, 'key_components'])
        event_components_dict = {
            e: {s: {} for s in ["Subject", "Verb", "Object", "Temporal Quantifier", "Spatial Quantifier",
                                "Quantities and Properties of Objects", "Sub-events"]} for e in events}
        if not events:
            continue
        for e in events:
            try:
                if not events[e]:
                    continue
            except TypeError:
                continue
            for key in events[e]:
                try:
                    if not events[e][key]:
                        continue
                except TypeError:
                    continue
                for component in events[e][key]:
                    result, price = conceptualize_component(e, key, component)
                    if result:
                        event_components_dict[e][key][component] = result
                        total_price += price
                    else:
                        data_failed_generation.loc[len(data_failed_generation)] = [data.loc[i, 'text'],
                                                                                   data.loc[i, 'generation'],
                                                                                   data.loc[i, 'key_components'],
                                                                                   price]
                    total += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        {"Total Price (HKD)": total_price, "Estimated:": total_price / total * total_data})
        data.loc[i, 'key_components'] = json.dumps(event_components_dict)
        if i % 1000 == 0:
            data.to_csv(
                './{}/{}/generation.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]), index=False)
            data_failed_generation.to_csv(
                './{}/{}/failed_generation.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]),
                index=False)

    data.to_csv(
        './{}/{}/generation.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]), index=False)
    data_failed_generation.to_csv(
        './{}/{}/failed_generation.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]),
        index=False)
    print("Total Price: {}".format(total_price))
    open('{}/{}/total_price.txt'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]), 'w').write(
        "Total price is:" + str(total_price))
