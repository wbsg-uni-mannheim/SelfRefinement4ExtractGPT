import copy
import json
import random
from datetime import datetime

import click
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

from pieutils import create_pydanctic_models_from_known_attributes
from pieutils.preprocessing import update_task_dict_from_train_set, update_task_dict_from_validation_set


@click.command()
@click.option('--dataset', default='oa-mine', help='Dataset Name')
@click.option('--model', default='gpt-4o-mini-2024-07-18', help='Model name')
@click.option('--verbose', default=True, help='Verbose mode')
@click.option('--description_configuration', default='short', help='Configuration of the attribute descriptions.')
@click.option('--no_example_values', default=5, help='Number of example values to use for self-reflection')
def main(dataset, model, verbose, description_configuration, no_example_values):
    # Load task template
    with open('prompts/task_template.json', 'r') as f:
        task_dict = json.load(f)

    # Assign file name to task_dict
    task_dict['task_name'] = f'03_zs_extraction_no_agent_ex_val_{description_configuration}_desc_{no_example_values}_run_0'

    task_dict['model'] = model
    task_dict['dataset_name'] = dataset
    task_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Copy train/validation task dict
    train_task_dict = copy.deepcopy(task_dict)
    val_task_dict = copy.deepcopy(task_dict)

    # Load examples into task dict
    train_task_dict = update_task_dict_from_train_set(train_task_dict)
    val_task_dict = update_task_dict_from_validation_set(val_task_dict)

    # Read prompt text from file
    with open('finetuning/03_zero_shot_extraction_short_with_na/sys_extraction.txt', 'r') as f:
        system_prompt_txt = f.read()

    train_records = []
    for example in tqdm(train_task_dict['examples']):
        attributes = train_task_dict['known_attributes'][example['category']]

        prompt = ChatPromptTemplate.from_messages([("system", system_prompt_txt),
                                                   ("human", "{input}")])

        formatted_prompt = prompt.invoke({
                                "category": example['category'],
                                "attributes": ', '.join(attributes),
                                "input": example['input']
                                })
        # CONTINUE HERE!
        messages = {"messages": []}
        for message in formatted_prompt.to_messages():
            message_type = message.type if message.type == 'system' else 'user'
            messages['messages'].append({'role': message_type, 'content': message.content})

        targets = {k: list(v.keys())[0] for k,v in example['target_scores'].items()}
        targets = {attribute: targets[attribute] if attribute in targets else "n/a" for attribute in attributes}

        formatted_response = json.dumps(targets, indent=1)
        messages['messages'].append({'role': 'assistant', 'content': f"```json\n{formatted_response}\n```"})
        train_records.append(messages)

    val_records = []
    for example in tqdm(val_task_dict['examples']):
        attributes = train_task_dict['known_attributes'][example['category']]

        prompt = ChatPromptTemplate.from_messages([("system", system_prompt_txt),
                                                   ("human", "{input}")])

        formatted_prompt = prompt.invoke({
                                "category": example['category'],
                                "attributes": ', '.join(attributes),
                                "input": example['input']
                                })
        # CONTINUE HERE!
        messages = {"messages": []}
        for message in formatted_prompt.to_messages():
            message_type = message.type if message.type == 'system' else 'user'
            messages['messages'].append({'role': message_type, 'content': message.content})

        targets = {k: list(v.keys())[0] for k,v in example['target_scores'].items()}
        targets = {attribute: targets[attribute] if attribute in targets else "n/a" for attribute in attributes}

        formatted_response = json.dumps(targets, indent=1)
        messages['messages'].append({'role': 'assistant', 'content': f"```json\n{formatted_response}\n```"})
        val_records.append(messages)

    # Save train and val records line by line
    with open(f'data/finetuning/{task_dict["dataset_name"]}/{task_dict["dataset_name"]}_train_records_short_with_na_prompt.jsonl', 'w') as f:
        for record in train_records:
            f.write(json.dumps(record) + '\n')

    with open(f'data/finetuning/{task_dict["dataset_name"]}/{task_dict["dataset_name"]}_val_records_short_with_na_prompt.jsonl', 'w') as f:
        for record in val_records:
            f.write(json.dumps(record) + '\n')

if __name__ == '__main__':
    load_dotenv()
    random.seed(42)
    main()