# Neural Network for Breast Cancer Classification

This project involves building a neural network to classify breast cancer tumors as malignant (M) or benign (B) based on features extracted from cell nuclei. The dataset is provided in a CSV file containing 32 columns, where the "diagnosis" column serves as the label (M for malignant, B for benign). The other columns represent various characteristics of the cell nucleus obtained through fine-needle aspiration.


Steps:

## Dataset Splitting:
The first step involves splitting the dataset into two parts: one for training and one for validation. This ensures that the model is evaluated on unseen data during the training process.
> to launch it make split &&  ./bin/split [-d || --dataset] [path dataset] 

## Training the Neural Network:

In this step, we will build and train the neural network on the training portion of the data. The model will learn to predict whether a tumor is malignant or benign based on the input features.

## Testing:
After the model is trained, we will evaluate its performance on the validation set to assess how well it generalizes to new, unseen data.

The dataset is located in the resources folder. It includes 32 columns, where "diagnosis" is the target label, and the other features describe the cell nucleus characteristics of breast masses.


## Unit Testing:

Unit tests have been implemented to verify the correctness of the key functions within the project. These tests ensure that the functions behave as expected and return the appropriate outputs.
> make test_all


