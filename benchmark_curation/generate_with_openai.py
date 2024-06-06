import sys
import time

import openai
from openai.lib.azure import AzureOpenAI

sys.path.append('../../MARS/')
from key import OPENAI_KEY

client = AzureOpenAI(
    api_key=OPENAI_KEY,
    api_version="2024-02-01",
    azure_endpoint="https://hkust.azure-api.net"
)


def generate_with_openai(prompt, model="gpt-35-turbo", system_prompt=None, max_tokens=200, temperature=0.2, top_p=1.0,
                         retry_attempt=5, verbose=False):
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [
            {"role": "user", "content": prompt},
        ]
    retry_num = 0
    generation_success = False
    while retry_num < retry_attempt and not generation_success:
        try:
            gen = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            generation_success = True
            input_tokens = gen.usage.prompt_tokens
            output_tokens = gen.usage.completion_tokens

            if verbose:
                print(gen.choices[0].message.content.strip())
                print("Prompt tokens: {}; Completion tokens: {}".format(input_tokens, output_tokens))
        except openai.APIError as e:  # TRIGGERED OpenAI API ERROR, Could be network issue
            if verbose:
                print(e)
            retry_num += 1
            generation_success = False
            time.sleep(10)
        except openai.RateLimitError as e:  # TRIGGERED OpenAI RATE LIMIT
            if verbose:
                print(e)
            retry_num += 1
            generation_success = False
            time.sleep(60)
        except openai.BadRequestError as e:  # TRIGGERED OpenAI CONTENT SAFETY FILTER
            if verbose:
                print(e)
            retry_num += 1
            generation_success = False
            # time.sleep(1)
        except:
            retry_num += 1
            generation_success = False
            time.sleep(20)
    if generation_success:
        try:
            return True, (
                gen.choices[0].message.content.strip(),
                0.002 * 7.83 * (input_tokens + 8) / 1000 + 0.002 * 7.83 * output_tokens / 1000
            )
        except:
            return False, None
    else:
        return False, None
