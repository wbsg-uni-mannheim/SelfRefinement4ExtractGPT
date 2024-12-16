# Self-Refinement Strategies for LLM-based Product Attribute Value Extraction
This repository contains code and data for the paper "Self-Refinement Strategies for LLM-based Product Attribute Value Extraction".

### Requirements

We evaluate OpenAI's LLM GPT-4on the task of product attribute value extraction using self-refinement strategies.
For the hosted LLMs an OpenAI access tokens needs to be placed in a `.env` file at the root of the repository.
To obtain this OpenAI access token, users must [sign up](https://platform.openai.com/signup) for an OpenAI account.

### Installation

The codebase requires python 3.9 To install dependencies we suggest to use a conda virtual environment:

```
conda create -n SelfRefinement python=3.9
conda activate SelfRefinement
pip install -r requirements.txt
pip install .
```

### Execution
You can run all experiments using the following script:

```
scripts/01_self-refinement_experiments.sh
```

## Self-Refinement Strategies for LLM-based Product Attribute Value Extraction

This paper critically evaluates two self-refinement techniques for extracting attribute values from product descriptions: 
_Error-based Prompt Rewriting_ and _Self-Correction_.

## Prompting Techniques
All prompting techniques and the code to execute the prompts are defined in the folder `prompts`.
Each task contains two sub-folders:
- `execution` contains the code to execute the prompting technique.
- `prompt_templates` contains the prompt templates for the different prompting techniques.

### Zero-Shot Prompting
``` 
System: 
    Extract the attribute values for 'Brand', 'Supplement Type', 'Dosage', 'Net Content' into a JSON object. 
    Return 'n/a' if the attribute is not found.

User: 
    NOW Supplements, Vitamin A (Fish Liver Oil) 25,000 IU, Essential Nutrition, 250 Softgels
```
Expected Output:
```
{
    "Brand": "NOW Supplements",
    "Supplement Type": "Vitamin A (Fish Liver Oil)",
    "Dosage": "25,000 IU",
    "Net Content": "250"
}
```

### Error-based Prompt Rewriting
Error-based Prompt Rewriting uses training data to improve the attribute definitions in the prompts. 
Error-based Prompt Rewriting assumes that better attribute definitions improve the product attribute value extraction.
```
System:
    You are provided with an attribute definition that is used to extract 
    attribute values from product descriptions and a list of product
    descriptions with incorrectly and correctly extract attribute values. Improve
    the attribute definition based on the incorrect and correct attribute values.
    Attribute values should be mentioned in "" and should naturally occur in
    the attribute definition. Respond with JSON object in the format:
    attribute: attribute definition with example values.

User:
    Type: The 'Type' attribute describes the style or category of the shoe, such 
    as 'Oxford', or 'Slide Mule'. It provides insight into the shoe's design.
    
    Product Offer: adidas Men's Superstar 2G Ultra Basketball Shoe,White/Black,12.5 D US
    Incorrect value: Basketball Shoe
    Correct value: n/a
```

Expected Output:
```
{
    "Type": "The ’Type’ attribute describes the specific style or category of the shoe, 
            focusing on traditional and widely recognized categories like 'Oxford' or 'Slide Mule'. 
            It should not include general terms like ’Walking Shoe’ or ’Basketball Shoe’."
}
```


### Self-Correction
Self-Correction reviews and updates the initial output of an LLM if it spots a wrongly extracted value.
A first prompt instructs the LLM to extract attribute-value pairs from the input. 
The first prompt can be a zero-shot or few-shot in-context learning prompt.
The output of the first prompt is sent to the same LLM again with a request to reflect on and correct erroneously extracted attribute values. 

``` 
System: 
    Extract the attribute values for 'Brand', 'Supplement Type', 'Dosage', 'Net Content' into a JSON object. 
    Return 'n/a' if the attribute is not found.

User: 
    NOW Supplements, Vitamin A (Fish Liver Oil) 25,000 IU, Essential Nutrition, 250 Softgels
    Extracted Attribute Values: 
        {
        "Brand": "NOW Supplements",
        "Supplement Type": "Vitamin A (Fish Liver Oil)",
        "Dosage": "25,000 IU",
        "Net Content": "n/a" // Incorrect value
        }
```
Expected Output:
```
{
    "Brand": "NOW Supplements",
    "Supplement Type": "Vitamin A (Fish Liver Oil)",
    "Dosage": "25,000 IU",
    "Net Content": "250" // Corrected value
}
```


## Dataset

For this work we use subsets of three datasets from related work: [OA-Mine](https://github.com/xinyangz/OAMine/)and  [AE-110k](https://github.com/cubenlp/ACL19_Scaling_Up_Open_Tagging/).
Each subset is split into a train and a test set.
OA-Mine contains 115 attributes and AE-110k contains 101 attributes.
WDC PAVE comes in two variants. The first variant contains extracted attribute values (WDC PAVE) and the second variant contains extracted and normalized attribute values (WDC PAVE-normalized).
Further statistics and information about the subsets can be found in the table below and in the paper.
Here is the updated table with the WDC PAVE dataset removed:

|                     | OA-Mine Train  | OA-Mine Test       | AE-110K Train     | AE-110K Test       |
|---------------------|----------------|--------------------|--------------------|--------------------|
| Attribute/Value Pairs| 1,467         | 2,451              | 859               | 1,482              |
| Unique Attribute Values | 1,120      | 1,749              | 302               | 454                |
| Product Offers      | 286            | 491                | 311               | 524                |

The dataset subsets of OA-Mine and AE-110k are available in the folder `data\processed_datasets`.
