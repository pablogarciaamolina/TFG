# SRC

## DATA

- Better handling for the different options of data status (avoid redowloading if its already dowloaded, ...)

- balanced_dataset and train_test_split wrapped !

### CIC-IDS2017

- Problem: If the dataset is already loaded with a configuration, loading another with a different configuration (without deleting the previous) will end in an error, since it tries to concat also the data in the already processed on (previous configuration)

--------------

- Dealing with missing values and infinite values. Needs solving
- Visualization of missing values
- nan class? -> doing the mapping of attack might be a good option

- Data analysis
    - Outliers analysis?

## MODELS

- Implement Data Augmentation for TAbNet Model:
    - The augmentation will be done to the classes that present less data (set in the configuration by a threshold with respect to the maximum)
    - The each augmentation will duplicate the number of examples for a given class, this will be done a limited amout of times (set in the configuration)

## PIPELINE

# Other

- Update README!