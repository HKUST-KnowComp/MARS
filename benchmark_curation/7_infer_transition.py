import os.path
import sys
import warnings

import pandas as pd
from tqdm import tqdm

sys.path.append('../')
sys.path.append('../../')
from generate_with_openai import generate_with_openai

TEST_MODE = False

CSV_PATH = [
    '../6_metaphysical_inference_generation/wikitext_length30-200_filter/wikitext_validation_sample100_length30-200/generation.csv',
    '../6_metaphysical_inference_generation/wikitext_length30-200_filter/wikitext_test_sample100_length30-200/generation.csv',
    '../6_metaphysical_inference_generation/wikitext_length30-200_filter/wikitext_train_sample50_length30-200/generation.csv',
    '../6_metaphysical_inference_generation/bookcorpus_length30-200_filter/bookcorpus_train_sample10_length30-200/generation.csv',
]

prompt = """You will be given an event and its metaphysical inference, meaning that such an inference is implausible or impossible in real life.
Please generate a transition that would make the inference plausible or possible in real life.
Specifically, you are required to only change a component of the event. The component must be one of the Subject, Verb, Object, Temporal Quantifier, Spatial Quantifier, Numerical Properties of Objects, and Sub-events of the event.
For example:
Event: The citizen announces a new law.
Metaphysical Inference: Other citizens follow the law.
Transition: citizen -> government (Subject)

Event: Sandy always laughs when she sees a clown.
Metaphysical Inference: She is not happy and is crying.
Transition: laughs -> cries (Verb)

Event: The cat is sleeping on the bed.
Metaphysical Inference: The cat is not tired and is playing.
Transition: sleeping -> playing (Verb)

Event: The boss of the company is monitoring the employees.
Metaphysical Inference: The boss feels nervous and is expecting a rise.
Transition: employees -> stocks (Object)

Event: The emperor is ruling the kingdom for 100 years.
Metaphysical Inference: The emperor is not old and is young.
Transition: 100 years -> 10 years (Temporal Quantifier)

Event: A man is walking on the beach.
Metaphysical Inference: There are many cars and buildings around him.
Transition: on the beach -> in the city (Spatial Quantifier)

Event: The man is being chased by a 100 meters butterfly in the forest.
Metaphysical Inference: The man is not scared and is laughing.
Transition: 100 meters -> 10 centimeters (Numerical Properties of Objects)

Event: The family is watching a show where a magician is performing magic tricks.
Metaphysical Inference: The family feels board about the show.
Transition: a magician is performing magic tricks -> a singer is singing old folk songs (Sub-events)

Event: {}
Metaphysical Inference: {}
Transition:"""


def transition_inference_generation(event, metaphysical_inference, verbose=False):
    success, result = generate_with_openai(prompt.format(event, metaphysical_inference), verbose=verbose)
    try:
        generation = result[0].split('Transition:')[-1].strip()
        original_part = generation.split('->')[0].strip()
        modified_part = generation.split('->')[1].strip().split('(')[0].strip()
        modified_type = generation.split('->')[1].strip().split('(')[1].split(')')[0].strip()
    except:
        return None, prompt.format(event, metaphysical_inference)
    if success:
        return (original_part, modified_part, modified_type), result[1]
    else:
        return None, prompt.format(event, metaphysical_inference)


def replace_substrings(string1, string2, string3):
    # Split string2 into words
    words = string2.lower().split()

    # Split string1 and string3 into individual words
    words_to_replace = string1.lower().split()
    replacement_words = string3.lower().split()

    if len(words_to_replace) > 1:
        return string2.replace(string1, string3)
    else:
        return " ".join([i if i != string1 else string3 for i in words])


if TEST_MODE:
    warnings.warn("Test mode is enabled. Only the first 5 rows will be processed.")
for CSV in CSV_PATH:
    if not os.path.exists('./{}/'.format(CSV.split('/')[2])):
        os.makedirs('./{}/'.format(CSV.split('/')[2]))
    if not os.path.exists('./{}/{}'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0])):
        os.makedirs('./{}/{}'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]))

    data = pd.read_csv(CSV, index_col=None)
    if TEST_MODE:
        data = data.sample(2).reset_index(drop=True)
    print("Number of transitions to be generated:", len(data))

    # text_id, text, event_id, event, component_type, component_original, component_substitution
    data_failed_generation = pd.DataFrame(columns=['text_id', 'text', 'event_id', 'event', 'component_type',
                                                   'component_original', 'component_substitution', 'event_type',
                                                   'inference', 'metaphysical_inference', 'prompt'])
    total_price = 0
    progress_bar = tqdm(total=len(data), desc="Generating {}".format(CSV.split('/')[3].split('.')[0]))
    for i in range(len(data)):
        event = data['event'][i]
        instance = data['component_original'][i]
        variation = data['component_substitution'][i]
        event_tokens = event.lower()
        instance_tokens = instance.lower()
        variation_tokens = variation.lower()
        event = replace_substrings(instance_tokens, event_tokens, variation_tokens)

        result, price = transition_inference_generation(event, data['metaphysical_inference'][i], verbose=TEST_MODE)
        if result:
            data.loc[i, 'event_replaced'] = event
            data.loc[i, 'transition_original'] = result[0]
            data.loc[i, 'transition_modified'] = result[1]
            data.loc[i, 'transition_type'] = result[2]
            total_price += price
        else:
            data_failed_generation.loc[len(data_failed_generation)] = [
                data['text_id'][i], data['text'][i],
                data['event_id'][i], data['event'][i],
                data['component_type'][i],
                data['component_original'][i],
                data['component_substitution'][i],
                data['event_type'][i],
                data['inference'][i],
                data['metaphysical_inference'][i],
                prompt.format(event, data['metaphysical_inference'][i])
            ]

        progress_bar.update(1)
        progress_bar.set_postfix(
            {"Total Price (HKD)": total_price, "Estimated:": total_price / (i + 1) * len(data)})

        if i % 1000 == 0:
            data.to_csv(
                './{}/{}/generation.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]), index=False)
            data_failed_generation.to_csv(
                './{}/{}/failed_generation.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]),
                index=False)

    data.dropna().to_csv(
        './{}/{}/generation.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]), index=False)
    data_failed_generation.to_csv(
        './{}/{}/failed_generation.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]),
        index=False)
    print("Total Price: {}".format(total_price))
    open('{}/{}/total_price.txt'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]), 'w').write(
        "Total price is:" + str(total_price))
