import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class AttributeDescriptionInput(BaseModel):
    """Input for the attribute description tool."""
    category: str = Field(..., title="Category", description="The category of the product offer.")


class AttributeDescriptionTool:
    """Tool to provide attribute descriptions."""

    def __init__(self):
        self.name = "AttributeDescription"
        self.attribute_descriptions = {}

    def initialize_attribute_descriptions(self, category: str, attributes: list, dataset_name, model, known_values=None, description_configuration='short'):
        """Initialize the attribute descriptions."""
        print(f"Category: {category}")
        print(f"Attributes: {attributes}")
        #print(f"Known Values: {known_values}")
        print(f"Description Configuration: {description_configuration}")

        # Generate Descriptions
        if description_configuration == 'only_example_values' and known_values is not None:
            self.attribute_descriptions[category] = known_values
        elif description_configuration == 'no_description':
            self.attribute_descriptions[category] = {attribute: '' for attribute in attributes}
        elif description_configuration == 'single_attribute':
            sys_prompt = ("You are a world-class product catalogue expert. "
                          "Write a detailed attribute description for the provided attribute."
                          "Please consider all example values of the attribute for the description."
                          "The description is used to extract attribute values from product descriptions."
                          "Respond with a JSON object with the following format: { attribute: 'attribute_name', description: 'attribute_description'}.")
            self.attribute_descriptions[category] = {}

            for attribute in attributes:
                json_known_values = json.dumps(known_values[attribute], indent=4)
                result = model.invoke([SystemMessage(content=sys_prompt),
                                       HumanMessage(content=f"Product Category: {category} \nAttribute and Example Values: \n{json_known_values}")])
                print("Attribute Description for " + attribute)
                print(result.content)
                if "```json" in result.content:
                    result.content = result.content.split("```json")[1].split("```")[0]
                self.attribute_descriptions[category][attribute] = json.loads(result.content)['description']
        elif description_configuration == 'single_attribute_no_attribute_name':
            sys_prompt = ("You are a world-class product catalogue expert. \n"
                          "The provided attribute values belong to the same attribute.\n"
                          "Write a attribute description that helps to identify values belonging to the same attribute.\n"
                          "The example values should be mentioned in brackets and should naturally occur in the attribute description.\n"
                          #"Please avoid the words 'category', 'attribute', and 'example values'.\n"
                          "Respond with a JSON object with the following format: { attribute: 'attribute_name', description: 'attribute_description'}.")
            self.attribute_descriptions[category] = {}

            for attribute in attributes:
                json_known_values = json.dumps(known_values[attribute], indent=4)
                result = model.invoke([SystemMessage(content=sys_prompt),
                                       HumanMessage(content=f"Product Category: {category} \nAttribute Values: \n{json_known_values}")])
                print("Attribute Description for " + attribute)
                print(result.content)
                if "```json" in result.content:
                    result.content = result.content.split("```json")[1].split("```")[0]
                self.attribute_descriptions[category][attribute] = json.loads(result.content)['description']
        else:
            if known_values is None:
                sys_prompt = ("You are a world-class product catalogue expert. "
                              "Write a one sentence attribute descriptions for the provided attributes."
                              "The descriptions are used to extract attribute values from product descriptions."
                              "The output should be a JSON object with the following format: {attribute: description}.")

                result = model.invoke([SystemMessage(content=sys_prompt),
                                       HumanMessage(content=f"Category: {category} \nAttributes: {attributes}")])
            else:
                if description_configuration == 'short':
                    sys_prompt = ("Write attribute descriptions for the provided attributes. \n"
                                  "The descriptions guide the extraction of attribute values from product descriptions.\n"
                                  "Attribute values should be mentioned in \" \" and should naturally occur in the attribute description.\n"
                                  "The output should be a JSON object with the following format: {attribute: description with example values}.")
                elif description_configuration == 'detailed':
                    sys_prompt = ("You are a world-class product catalogue expert. \n"
                                  "You are provided with a list of attributes and example values for these attributes that have been extract from product descriptions. \n"
                                  "Write detailed attribute descriptions for the provided attributes.\n"
                                  "The descriptions should help a human annotator to extract relevant attribute values from product descriptions. \n"
                                  "Please add all example values to the respective attribute description.\n"
                                  "Attribute values should be mentioned in \" \" and should naturally occur in the attribute description. \n"
                                  "The output should be a JSON object with the following format: {attribute: description with example values}.")
                elif description_configuration == 'detailed+difference':
                    sys_prompt = ("You are a world-class product catalogue expert. "
                                  "Write detailed attribute descriptions for the provided attributes."
                                  "The descriptions are used to extract attribute values from product descriptions."
                                  "If necessary, explain the difference between attributes."
                                  "Please add all example values to the respective attribute description."
                                  "The output should be a JSON object with the following format: {attribute: description with example values}.")
                else:
                    raise ValueError(f"Invalid description configuration {description_configuration}. Please use 'short' or 'long'.")

                json_known_values = json.dumps(known_values, indent=4)
                result = model.invoke([SystemMessage(content=sys_prompt),
                                       HumanMessage(
                                           content=f"Attribute Example Values: \n{json_known_values}")])
            print(result.content)
            if "```json" in result.content:
                result.content = result.content.split("```json")[1].split("```")[0]
            self.attribute_descriptions[category] = json.loads(result.content)

    def update_attribute_descriptions(self, prediction_mistakes: dict, model, single_attribute=False):
        """Update the attribute descriptions based on prediction mistakes."""

        for category in self.attribute_descriptions:
            if category not in prediction_mistakes or len(prediction_mistakes[category]) == 0:
                print(f"No prediction mistakes for category {category}.")
            elif single_attribute:
                sys_prompt = ( "You are provided with an attribute description "
                               "that is used to extract attribute values from product descriptions and "
                               "a list product descriptions with incorrectly and correctly extract attribute values. "
                               "Improve the attribute description based on the incorrect and correct attribute values. "
                               "Attribute values should be mentioned in \" \" and should naturally occur in the "
                               "attribute description. Respond with JSON object in the format: "
                               "attribute: attribute description with example values.")

                # Collect mistakes per attribute
                attribute_mistakes = {}
                for prediction_mistake in prediction_mistakes[category]:
                    for attribute in prediction_mistake:
                        if attribute in ['input']:
                            continue
                        if attribute not in attribute_mistakes:
                            attribute_mistakes[attribute] = []
                        attribute_mistakes[attribute].append(prediction_mistake)

                for attribute in attribute_mistakes:
                    # Format mistakes
                    task_input = f"{attribute}:{self.attribute_descriptions[category][attribute]}\n"
                    for prediction_mistake in attribute_mistakes[attribute]:
                        task_input += f"Product Offer: {prediction_mistake['input']}\n"
                        task_input += f"Incorrect Value: {prediction_mistake[attribute]['predicted']}\n"
                        task_input += f"Correct Value: {prediction_mistake[attribute]['target']}\n\n"

                    messages = [SystemMessage(content=sys_prompt), HumanMessage(content=task_input)]

                    for message in messages:
                        print(message.content)

                    result = model.invoke(messages)

                    if "```json" in result.content:
                        result.content = result.content.split("```json")[1].split("```")[0]
                    try:
                        updated_description = json.loads(result.content)
                        for attribute in updated_description:
                            self.attribute_descriptions[category][attribute] = updated_description[attribute]
                        print(result.content)
                    except Exception as e:
                        print(f"Error: {e}")

                    #if attribute == 'Size' and category == 'Toothbrush':
                    #    print('Wait!') # Debugging
            else:
                sys_prompt = (  "You are a world-class product catalogue expert.\n"
                                "You are provided with a list of attribute descriptions that are used to extract attribute values from product descriptions. \n"
                                "Additionally, you are provided with a list of correctly and incorrectly extract attribute values. \n"
                                "Reflect on and improve the attribute descriptions based on the correct and incorrect attribute values. \n"
                                "Attribute values should be mentioned in \" \" and should naturally occur in the attribute description. \n"
                                "The output format is supposed to be a JSON object with the following format: {attribute: attribute description with example values}")
                formatted_mistakes = ""
                for prediction_mistake in prediction_mistakes[category]:
                    formatted_mistakes += f"Product Offer: {prediction_mistake['input']}\n"
                    for attribute in prediction_mistake:
                        if attribute in ['input']:
                            continue
                        formatted_mistakes += f"Attribute: {attribute}\n"
                        formatted_mistakes += f"Incorrect Value: {prediction_mistake[attribute]['predicted']}\n"
                        formatted_mistakes += f"Correct Value: {prediction_mistake[attribute]['target']}\n\n"
                formatted_attributes = '\n'.join([f"{attribute}: {self.attribute_descriptions[category][attribute]}" for attribute in self.attribute_descriptions[category]])

                messages = [SystemMessage(content=sys_prompt),
                                         HumanMessage(content=f"Attribute Descriptions: {formatted_attributes} \nPrediction Mistakes: \n{formatted_mistakes}")]

                for message in messages:
                    print(message.content)

                result = model.invoke(messages)

                if "```json" in result.content:
                    result.content = result.content.split("```json")[1].split("```")[0]
                try:
                    updated_descriptions = json.loads(result.content)
                    for attribute in updated_descriptions:
                        self.attribute_descriptions[category][attribute] = updated_descriptions[attribute]
                    print(f"Updated attribute descriptions for category {category}.")
                    print(result.content)
                except Exception as e:
                    print(f"Error: {e}")

    def ensemble_attribute_descriptions(self, tools_attribute_description, validation_results):
        """Ensemble attribute descriptions from multiple tools."""
        results_per_category_attribute = {}
        attribute_category_combinations = {}
        for description_configuration in validation_results:
            print(f"Description Configuration: {description_configuration}")
            if len(attribute_category_combinations) == 0:
                for category in tools_attribute_description[description_configuration].attribute_descriptions:
                    for attribute in tools_attribute_description[description_configuration].attribute_descriptions[category]:
                        attribute_category_combinations[f"{attribute}_{category}"] = (category, attribute)

                # Initialize attribute descriptions
                self.attribute_descriptions = tools_attribute_description[description_configuration].attribute_descriptions

            for attribute_category in validation_results[description_configuration]:
                if attribute_category in ['micro', 'macro']:
                    continue
                if attribute_category not in attribute_category_combinations:
                    category = attribute_category.split('_')[1]
                    attribute = attribute_category.split('_')[0]
                else:
                    category, attribute = attribute_category_combinations[attribute_category]

                if category not in results_per_category_attribute:
                    results_per_category_attribute[category] = {}
                if attribute not in results_per_category_attribute[category]:
                    results_per_category_attribute[category][attribute] = {}

                # Select description based on F1-score
                results_per_category_attribute[category][attribute][description_configuration] = validation_results[description_configuration][attribute_category]['f1']

        for category in results_per_category_attribute:
            if category not in self.attribute_descriptions:
                self.attribute_descriptions[category] = {}

            for attribute in results_per_category_attribute[category]:
                best_description_configuration = max(results_per_category_attribute[category][attribute],
                                                     key=results_per_category_attribute[category][attribute].get)

                if category in tools_attribute_description[best_description_configuration].attribute_descriptions \
                    and attribute in tools_attribute_description[best_description_configuration].attribute_descriptions[category]:
                    self.attribute_descriptions[category][attribute] = tools_attribute_description[best_description_configuration].attribute_descriptions[category][attribute]
                else:
                    self.attribute_descriptions[category][attribute] = ""

    def invoke(self, category: str):
        return json.dumps(self.attribute_descriptions.get(category, {}))
