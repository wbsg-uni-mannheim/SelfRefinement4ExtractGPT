import copy
import json
import random
from datetime import datetime

import click
from dotenv import load_dotenv
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from pieutils import create_pydanctic_models_from_known_attributes, save_populated_task, aggregate_overall_results
from pieutils.evaluation import evaluate_predictions
from pieutils.preprocessing import update_task_dict_from_test_set, load_known_attribute_values, \
    update_task_dict_from_validation_set, update_task_dict_from_train_set
from pieutils.rewrite_description_tools import AttributeDescriptionTool
from pieutils.search import CategoryAwareSemanticSimilarityExampleSelector


@click.command()
@click.option('--dataset', default='oa-mine', help='Dataset Name')
@click.option('--model', default='gpt-4o-mini-2024-07-18', help='Model name')
@click.option('--verbose', default=True, help='Verbose mode')
@click.option('--description_configuration', default='short', help='Configuration of the attribute descriptions.')
@click.option('--no_example_values', default=5, help='Number of example values to use for self-reflection')
@click.option('--train_percentage', default=1.0, help='Used percentage of training data')
@click.option('--shots', default=10, help='Number of examples to select for in-context learning')
def main(dataset, model, verbose, description_configuration, no_example_values, train_percentage, shots):
    # Load task template
    with open('prompts/task_template.json', 'r') as f:
        task_dict = json.load(f)

    # Assign file name to task_dict
    task_dict['task_name'] = f'09_few_shot_extraction_ex_val_{description_configuration}_desc_{no_example_values}_{model}_default_gpt4o_rewrite_desc_shots_{shots}_run_0'

    task_dict['model'] = model
    task_dict['dataset_name'] = dataset
    task_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    train_task_dict = copy.deepcopy(task_dict)
    validation_task_dict = copy.deepcopy(task_dict)

    # Load examples into task dict
    task_dict = update_task_dict_from_test_set(task_dict)
    validation_task_dict = update_task_dict_from_validation_set(validation_task_dict)
    train_task_dict = update_task_dict_from_train_set(train_task_dict, train_percentage=train_percentage)

    known_attribute_values = load_known_attribute_values(task_dict['dataset_name'], n_examples=no_example_values,
                                                         train_percentage=train_percentage)

    # # Filter to first 5 examples
    # task_dict['examples'] = task_dict['examples'][:5]
    # filtered_categories = list(set([example['category'] for example in task_dict['examples']]))
    # known_attribute_values = {category: known_attribute_values[category] for category in filtered_categories}
    #
    # # Filter validation set for examples from filtered categories
    # validation_task_dict['examples'] = [example for example in validation_task_dict['examples'] if example['category'] in filtered_categories]
    # validation_task_dict['examples'] = validation_task_dict['examples'][:5]
    #
    # # Filter training set for examples from filtered categories
    # train_task_dict['examples'] = [example for example in train_task_dict['examples'] if example['category'] in filtered_categories]
    # train_task_dict['examples'] = train_task_dict['examples'][:5]

    default_llm = ChatOpenAI(model_name=task_dict['model'], temperature=0)
    llm = ChatOpenAI(model_name=task_dict['model'], temperature=0)

    # Initialize attribute descriptions
    tool_attribute_descriptions = AttributeDescriptionTool()
    with get_openai_callback() as cb:
        for category, attributes in task_dict['known_attributes'].items():
            if category not in known_attribute_values:
                continue
            tool_attribute_descriptions.initialize_attribute_descriptions(category=category, attributes=attributes,
                                                                          dataset_name=task_dict['dataset_name'],
                                                                          model=default_llm,
                                                                          known_values=known_attribute_values[category],
                                                                          description_configuration=description_configuration)
            if 'attribute_descriptions' not in task_dict:
                task_dict['attribute_descriptions'] = {}
                task_dict['initial_attribute_descriptions'] = {}

            task_dict['attribute_descriptions'][category] = tool_attribute_descriptions.attribute_descriptions[category]
            task_dict['initial_attribute_descriptions'][category] = copy.deepcopy(tool_attribute_descriptions.attribute_descriptions[category])

        task_dict['attribute_description_prompt_tokens'] = cb.prompt_tokens
        task_dict['attribute_description_completion_tokens'] = cb.completion_tokens
        task_dict['attribute_description_total_tokens'] = cb.total_tokens
        task_dict['attribute_description_total_cost'] = cb.total_cost

    # Read prompt text from file
    with open('prompts/prompt_templates/09_few-shot_extraction_example_values_error_based_rewrite_desc/sys_extraction.txt',
              'r') as f:
        system_prompt_txt = f.read()

    pydantic_models = create_pydanctic_models_from_known_attributes(task_dict['known_attributes'])

    # CategoryAware Semantic Similarity Example Selector for in-context learning
    category_example_selector = CategoryAwareSemanticSimilarityExampleSelector(task_dict['dataset_name'],
                                                                               list(task_dict[
                                                                                        'known_attributes'].keys()),
                                                                               train_percentage=train_percentage,
                                                                               load_from_local=False, k=shots)
    # Define the few-shot prompt.
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        # The input variables select the values to pass to the example_selector
        input_variables=["input", "category"],
        example_selector=category_example_selector,
        example_prompt=ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        ),
    )

    overall_results = {}
    for i in range(3):
        task_dict['task_name'] = task_dict['task_name'].replace(f'run_{i}', f'run_{i+1}')
        with get_openai_callback() as cb:
            for j in range(3):
                train_preds = []
                copy_train_task_dict = copy.deepcopy(train_task_dict)
                # filter train task dict to 5 examples per category
                filtered_train_examples = {}
                for example in random.sample(train_task_dict['examples'], len(train_task_dict['examples'])):
                    if example['category'] not in filtered_train_examples:
                        filtered_train_examples[example['category']] = []
                    if len(filtered_train_examples[example['category']]) < 5: # Increase to 5 after debugging!
                        filtered_train_examples[example['category']].append(example)

                copy_train_task_dict['examples'] = []
                for category, examples in filtered_train_examples.items():
                    copy_train_task_dict['examples'].extend(examples)

                for example in tqdm(copy_train_task_dict['examples']):
                    attributes = train_task_dict['known_attributes'][example['category']]
                    attribute_descriptions = tool_attribute_descriptions.attribute_descriptions[example['category']]
                    formatted_attributes = '\n'.join(
                        [f"{attribute}: {attribute_descriptions[attribute]}" for attribute in attributes if
                         attribute in attribute_descriptions and len(attribute_descriptions[attribute]) > 0])
                    parser = JsonOutputParser(pydantic_object=pydantic_models[example['category']])

                    prompt = ChatPromptTemplate.from_messages([("system", system_prompt_txt),
                                                               few_shot_prompt,
                                                               ("human", "{input}")])
                    llm_chain = prompt | llm
                    response = llm_chain.invoke({
                        "category": example['category'],
                        "attributes": ', '.join(attributes),
                        "attribute_descriptions": formatted_attributes,
                        "input": example['input']
                    })

                    try:
                        response = parser.parse(response.content)
                        print(f"Example: {example['input']}")
                        print(f"Response: {response}")
                    except Exception as e:
                        print(f"Error: {e}")
                        response = {}
                    train_preds.append(json.dumps(response))

                # Evaluate Train Predictions
                chunk_train_results = evaluate_predictions(train_preds, copy_train_task_dict)
                if 'chunk_train_results' not in task_dict:
                    task_dict['chunk_train_results'] = {}
                task_dict['chunk_train_results'][j] = chunk_train_results
                print(f"Chunk Train Results: {chunk_train_results}")

                # Update description based on training set
                prediction_mistakes = {}
                for example in copy_train_task_dict['examples']:
                    if example['category'] not in prediction_mistakes:
                        prediction_mistakes[example['category']] = []

                    prediction_mistake = {'input': example['input']}
                    post_pred = json.loads(example['post_pred'])
                    target = {}
                    for attribute, value in example['target_scores'].items():
                        target[attribute] = list(value.keys())[0]
                    for attribute in post_pred.keys():
                        if attribute in target:
                            if post_pred[attribute] != target[attribute]:
                                if attribute not in prediction_mistake:
                                    prediction_mistake[attribute] = {}
                                prediction_mistake[attribute]['predicted'] = post_pred[attribute]
                                prediction_mistake[attribute]['target'] = target[attribute]
                        elif attribute not in target and post_pred[attribute] != 'n/a':
                            if attribute not in prediction_mistake:
                                prediction_mistake[attribute] = {}
                            prediction_mistake[attribute]['predicted'] = post_pred[attribute]
                            prediction_mistake[attribute]['target'] = 'n/a'

                    if len(prediction_mistake) > 1:
                        prediction_mistakes[example['category']].append(prediction_mistake)

                # Update descriptions based on prediction mistakes
                tool_attribute_descriptions.update_attribute_descriptions(prediction_mistakes, default_llm, single_attribute=True)
                if 'updated_attribute_descriptions' not in task_dict:
                    task_dict['updated_attribute_descriptions'] = {}
                task_dict['updated_attribute_descriptions'][j] = copy.deepcopy(tool_attribute_descriptions.attribute_descriptions)

            task_dict['rewrite_prompt_tokens'] = cb.prompt_tokens
            task_dict['rewrite_completion_tokens'] = cb.completion_tokens
            task_dict['rewrite_total_tokens'] = cb.total_tokens
            task_dict['rewrite_total_cost'] = cb.total_cost

            print(f"Total Tokens: {cb.total_tokens}")

        # Save populated task and make sure that it is saved in the correct folder!
        save_populated_task(task_dict['task_name'], task_dict)

        with get_openai_callback() as cb:
            # Run final predictions on test set
            preds = []
            for example in tqdm(task_dict['examples']):
                attributes = validation_task_dict['known_attributes'][example['category']]
                attribute_descriptions = task_dict['attribute_descriptions'][example['category']]
                formatted_attributes = '\n'.join(
                    [f"{attribute}: {attribute_descriptions[attribute]}" for attribute in attributes if
                     attribute in attribute_descriptions and len(attribute_descriptions[attribute]) > 0])
                parser = JsonOutputParser(pydantic_object=pydantic_models[example['category']])

                prompt = ChatPromptTemplate.from_messages([("system", system_prompt_txt),
                                                           few_shot_prompt,
                                                           ("human", "{input}")])
                llm_chain = prompt | llm
                response = llm_chain.invoke({
                    "category": example['category'],
                    "attributes": ', '.join(attributes),
                    "attribute_descriptions": formatted_attributes,
                    "input": example['input']
                })

                try:
                    response = parser.parse(response.content)
                    print(f"Example: {example['input']}")
                    print(f"Response: {response}")
                except Exception as e:
                    print(f"Error: {e}")
                    response = {}
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