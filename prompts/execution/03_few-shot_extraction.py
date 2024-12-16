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
from pieutils.preprocessing import update_task_dict_from_test_set
from pieutils.search import CategoryAwareSemanticSimilarityExampleSelector


@click.command()
@click.option('--dataset', default='oa-mine', help='Dataset Name')
@click.option('--model', default='gpt-4o-mini-2024-07-18', help='Model name')
@click.option('--verbose', default=True, help='Verbose mode')
@click.option('--shots', default=10, help='Number of examples to select for in-context learning')
@click.option('--train_percentage', default=1.0, help='Used percentage of training data')
def main(dataset, model, verbose, shots, train_percentage):
    # Load task template
    with open('prompts/task_template.json', 'r') as f:
        task_dict = json.load(f)

    # Assign file name to task_dict
    task_dict['task_name'] = f'03_fs_extraction_shots_{shots}_run_0'

    task_dict['model'] = model
    task_dict['dataset_name'] = dataset
    task_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Load examples into task dict
    task_dict = update_task_dict_from_test_set(task_dict)

    # Filter to first 5 examples
    # task_dict['examples'] = task_dict['examples'][:5]

    llm = ChatOpenAI(model_name=task_dict['model'], temperature=0)

    # Initialize demonstrations

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

    # Read prompt text from file
    with open('prompts/prompt_templates/03_few-shot_extraction/sys_extraction.txt',
              'r') as f:
        system_prompt_txt = f.read()

    pydantic_models = create_pydanctic_models_from_known_attributes(task_dict['known_attributes'])

    overall_results = {}
    for i in range(3):
        task_dict['task_name'] = task_dict['task_name'].replace(f'run_{i}', f'run_{i+1}')
        with get_openai_callback() as cb:
            preds = []
            for example in tqdm(task_dict['examples']):
                attributes = task_dict['known_attributes'][example['category']]
                parser = JsonOutputParser(pydantic_object=pydantic_models[example['category']])

                prompt = ChatPromptTemplate.from_messages([("system", system_prompt_txt),
                                                           few_shot_prompt,
                                                           ("human", "{input}")])

                llm_chain = prompt | llm
                response = llm_chain.invoke({
                                        "category": example['category'],
                                        "attributes": ', '.join(attributes),
                                        "input": example['input']
                                        })

                formatted_prompt = prompt.invoke({
                                        "category": example['category'],
                                        "attributes": ', '.join(attributes),
                                        "input": example['input']
                                        })
                print(f"Formatted Prompt: {formatted_prompt.to_messages()[0].content}")
                print(f"Formatted Input: {formatted_prompt.to_messages()[1].content}")

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