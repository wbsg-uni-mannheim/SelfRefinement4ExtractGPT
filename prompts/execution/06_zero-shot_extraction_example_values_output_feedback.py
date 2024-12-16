import json
import random
from datetime import datetime

import click
from dotenv import load_dotenv
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from pieutils import create_pydanctic_models_from_known_attributes, save_populated_task, aggregate_overall_results
from pieutils.evaluation import evaluate_predictions
from pieutils.preprocessing import update_task_dict_from_test_set, load_known_attribute_values
from pieutils.rewrite_description_tools import AttributeDescriptionTool


@click.command()
@click.option('--dataset', default='oa-mine', help='Dataset Name')
@click.option('--model', default='gpt-4o-mini-2024-07-18', help='Model name')
@click.option('--verbose', default=True, help='Verbose mode')
@click.option('--description_configuration', default='short', help='Configuration of the attribute descriptions.')
@click.option('--no_example_values', default=5, help='Number of example values to use for self-reflection')
@click.option('--train_percentage', default=1.0, help='Used percentage of training data')
def main(dataset, model, verbose, description_configuration, no_example_values, train_percentage):
    # Load task template
    with open('prompts/task_template.json', 'r') as f:
        task_dict = json.load(f)

    # Assign file name to task_dict
    task_dict['task_name'] = f'06_zs_extraction_ex_val_{description_configuration}_desc_{no_example_values}_output_feedback_run_0'

    task_dict['model'] = model
    task_dict['dataset_name'] = dataset
    task_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Load examples into task dict
    task_dict = update_task_dict_from_test_set(task_dict)

    known_attribute_values = load_known_attribute_values(task_dict['dataset_name'], n_examples=no_example_values,
                                                         train_percentage=train_percentage)

    # Filter to first 5 examples
    # task_dict['examples'] = task_dict['examples'][:5]

    llm = ChatOpenAI(model_name=task_dict['model'], temperature=0)

    # Initialize attribute descriptions
    tool_attribute_descriptions = AttributeDescriptionTool()


    for category, attributes in task_dict['known_attributes'].items():
        tool_attribute_descriptions.initialize_attribute_descriptions(category=category, attributes=attributes,
                                                                      model=llm,
                                                                      dataset_name=task_dict['dataset_name'],
                                                                      known_values=known_attribute_values[category],
                                                                      description_configuration=description_configuration)
        if 'attribute_descriptions' not in task_dict:
            task_dict['attribute_descriptions'] = {}
        task_dict['attribute_descriptions'][category] = tool_attribute_descriptions.attribute_descriptions[category]

    # Read prompt text from file
    with open('prompts/prompt_templates/06_zero-shot_extraction_example_values_output_feedback/sys_extraction.txt',
              'r') as f:
        system_prompt_txt = f.read()

    with open('prompts/prompt_templates/06_zero-shot_extraction_example_values_output_feedback/sys_review.txt',
              'r') as f:
        system_review_prompt_txt = f.read()

    pydantic_models = create_pydanctic_models_from_known_attributes(task_dict['known_attributes'])

    overall_results = {}
    for i in range(3):
        task_dict['task_name'] = task_dict['task_name'].replace(f'run_{i}', f'run_{i+1}')
        with get_openai_callback() as cb:
            preds = []
            for example in tqdm(task_dict['examples']):
                attributes = task_dict['known_attributes'][example['category']]
                attribute_descriptions = task_dict['attribute_descriptions'][example['category']]
                formatted_attributes = '\n'.join(
                    [f"{attribute}: {attribute_descriptions[attribute]}" for attribute in attributes if
                     attribute in attribute_descriptions and len(attribute_descriptions[attribute]) > 0])

                parser = JsonOutputParser(pydantic_object=pydantic_models[example['category']])

                prompt = ChatPromptTemplate.from_messages([("system", system_prompt_txt),
                                                           ("human", "{input}")])

                llm_chain = prompt | llm
                response = llm_chain.invoke({
                                        "category": example['category'],
                                        "attributes": ', '.join(attributes),
                                        "attribute_descriptions": formatted_attributes,
                                        "input": example['input']
                                        })

                formatted_prompt = prompt.invoke({
                                        "category": example['category'],
                                        "attributes": ', '.join(attributes),
                                        "attribute_descriptions": formatted_attributes,
                                        "input": example['input']
                                        })
                print(f"Formatted Prompt: {formatted_prompt.to_messages()[0].content}")
                print(f"Formatted Input: {formatted_prompt.to_messages()[1].content}")

                try:
                    response = parser.parse(response.content)
                    print(f"Response: {response}")
                except Exception as e:
                    print(f"Error: {e}")
                    response = {}

                # Review output
                prompt = ChatPromptTemplate.from_messages([("system", system_review_prompt_txt),
                                                           ("human", "{input} \n Extracted Values: {response}")])

                llm_chain = prompt | llm
                corrected_response = llm_chain.invoke({
                                        "category": example['category'],
                                        "attributes": ', '.join(attributes),
                                        "attribute_descriptions": formatted_attributes,
                                        "input": example['input'],
                                        "response": response
                                        })

                formatted_prompt = prompt.invoke({
                                        "category": example['category'],
                                        "attributes": ', '.join(attributes),
                                        "attribute_descriptions": formatted_attributes,
                                        "input": example['input'],
                                        "response": response
                                        })
                print(f"Formatted Prompt: {formatted_prompt.to_messages()[0].content}")
                print(f"Formatted Input: {formatted_prompt.to_messages()[1].content}")

                try:
                    if "```json" in corrected_response.content:
                        corrected_response.content = corrected_response.content.split("```json")[1].split("```")[0]

                    corrected_response = json.loads(corrected_response.content)
                    print(f"Corrected Response: {corrected_response}")
                    for attribute in corrected_response.keys():
                        response[attribute] = corrected_response[attribute]
                except Exception as e:
                    print(f"Corrected Response: {corrected_response.content}")
                    print(f"Error: {e}")

                preds.append(json.dumps(response))

            task_dict['prompt_tokens'] = cb.prompt_tokens
            task_dict['completion_tokens'] = cb.completion_tokens
            task_dict['total_tokens'] = cb.total_tokens
            task_dict['total_cost'] = cb.total_cost

            print(f"Total Tokens: {cb.total_tokens}")

        # Calculate recall, precision and f1
        task_dict['results'] = evaluate_predictions(preds, task_dict)
        # Save populated task and make sure that it is saved in the correct folder!
        save_populated_task(task_dict['task_name'], task_dict)

        overall_results[task_dict['task_name']] = task_dict['results']
        overall_results[task_dict['task_name']]['total_tokens'] = task_dict['total_tokens']
        overall_results[task_dict['task_name']]['prompt_tokens'] = task_dict['prompt_tokens']
        overall_results[task_dict['task_name']]['completion_tokens'] = task_dict['completion_tokens']
        overall_results[task_dict['task_name']]['total_cost'] = task_dict['total_cost']

    # Aggregate overall results
    overall_results = aggregate_overall_results(overall_results)

    # Save overall results in last task dict
    task_dict['overall_results'] = overall_results

    # Save populated task and make sure that it is saved in the correct folder!
    save_populated_task(task_dict['task_name'], task_dict)

if __name__ == '__main__':
    load_dotenv()
    random.seed(42)
    main()