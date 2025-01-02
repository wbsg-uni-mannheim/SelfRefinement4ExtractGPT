# Automated Self-Refinement and Self-Correction for LLM-based Product Attribute Value Extraction
This repository contains code and data for the paper "Automated Self-Refinement and Self-Correction for LLM-based Product Attribute Value Extraction".

### Requirements

We evaluate OpenAI's LLM GPT-4on the task of product attribute value extraction using self-refinement strategies.
For the hosted LLMs an OpenAI access tokens needs to be placed in a `.env` file at the root of the repository.
To obtain this OpenAI access token, users must [sign up](https://platform.openai.com/signup) for an OpenAI account.

### Installation

The codebase requires python 3.9 to install dependencies we suggest to use a conda virtual environment:

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

We critically evaluate two self-refinement techniques for extracting attribute values from product descriptions: 
_Error-based Prompt Rewriting_ and _Self-Correction_, across zero-shot, few-shot in-context learning,
and fine-tuning settings.

## Prompting Techniques
All prompting techniques and the code to execute the prompts are implemented in the folder `prompts`.
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
It assumes that better attribute definitions improve the product attribute value extraction.
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

|                     | OA-Mine Train | OA-Mine Test       | AE-110K Train | AE-110K Test       |
|---------------------|---------------|--------------------|---------------|--------------------|
| Attribute/Value Pairs| 3,626         | 2,451              | 2,170         | 1,482              |
| Unique Attribute Values | 2,400         | 1,749              | 587           | 454                |
| Product Offers      | 715           | 491                | 785           | 524                |

The dataset subsets of OA-Mine and AE-110k are available in the folder `data\processed_datasets`.

### Example Product Offers for testing

We provide a list of example product offers (Task Input) and the expected output (Task Output).
These examples can be used to quickly test the LLMs and the prompts.

``` 
User: Dr. Brown's Infant-to-Toddler Toothbrush Set, 1.4 Ounce, Blue
Expected Task Output (Answer by the LLM): 
{
    "Brand": "Dr. Brown's",
    "Color": "Blue",
    "Material": "n/a"
    "Age": "Infant-to-Toddler"
    "Size": "n/a"
}

``` 

``` 
User: SJP by Sarah Jessica Parker Women's Fawn Pointed Toe Dress Pump, Candy, 6.5 Medium US
Expected Task Output (Response of the LLM): 
{
  "Brand": "SJP by Sarah Jessica Parker",
  "Gender": "Women's",
  "Model name": "Fawn",
  "Shoe type": "Pointed Toe Dress Pump",
  "Color": "Candy",
  "Size": "6.5 Medium US"
  "Sport": "n/a"
}

```

``` 
User: Bigelow Green Tea with Wild Blueberry & Acai, 20 Count Box (Pack of 6), Caffeinated Green Tea, 120 Teabags Total
Expected Task Output (Response of the LLM): 
{
  "Brand": "Bigelow",
  "Tea variety": "Green Tea",
  "Flavor": "Wild Blueberry & Acai",
  "Net content": 120,
  "Pack size": "Pack of 6",
  "Caffeine content": "Caffeinated",
  "Item form": "Teabags"
}

```

``` 
User: YTQHXY Crank Fishing Lure 60mm 6.5g Artificial Hard Bait Wobblers 3D Eyes Plastic Japan Bait Bass Perch Fishing Peche YE-512
Expected Task Output (Response of the LLM): 
{
  "Model Number": "YE-512",
  "Brand Name": "YTQHXY",
  "Length": "60mm",
  "Weight": "6.5g"
}

``` 

``` 
User: POLISI Children Kids Winter Skate Snowmobile Glasses Snowboard Ski Snow Goggles UV400 Anti-Fog Lens Skiing Eyewear
Expected Task Output (Response of the LLM): 
{
  "Sport Type": "Skiing",
  "Brand Name": "POLISI"
  "Lenses Optical Attribute": "UV400 Anti-Fog Lens",
  "Frame Material": "n/a"
}

``` 
