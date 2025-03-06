# SRC

## DATA

- Better handling for the different options of data status (avoid redowloading if its already dowloaded, ...)

### CIC-IDS2017


- Dealing with missing values and infinite values. Needs solving
- Visualization of missing values
- nan class? -> doing the mapping of attack might be a good option

- Data analysis
    - Outliers analysis?

- Extended dataset? Avoided bacause it needs futher processing before applying PCA

- Check if new data pipeline works (analysis not tested)

## MODELS

- Implementation of a base class for all classification models, so adding new model is easier and they all share a common api

## PIPELINE

- Implement pipeline for using CLI to clasify a datset using Mistral and Gemini

## IMPLEMENTATION / Notebooks?

- Preparing data for models
    - Binarizing