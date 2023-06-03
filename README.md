# Chest X-ray Classification

This project focuses on classifying chest X-ray images into two categories: normal and pneumonia. It utilizes machine learning techniques and various classifiers to achieve accurate classification results.

## Dataset

The dataset used in this project consists of chest X-ray images. The images are divided into a training set and a test set. The training set is used to train the models, while the test set is used to evaluate their performance.

## Setup and Dependencies

To run the code, you need to have Python installed on your system. Additionally, make sure to install the following dependencies:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- OpenCV (cv2)
- imblearn
- yellowbrick
- tensorflow

You can install these dependencies by running the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python imbalanced-learn yellowbrick tensorflow
```


## Usage

1. Clone this repository or download the code files.
2. Ensure that the dataset is located in the correct directories (`train` and `test` folders).
3. Run the code in your preferred Python environment.
4. The code will preprocess the data, train the classifiers, and generate classification reports and confusion matrices.
5. You can modify the code as needed or experiment with different classifiers, preprocessing techniques, or hyperparameter tuning.

## Results

The code will provide accuracy scores, confusion matrices, and classification reports for each classifier used. Additionally, it includes data visualization to understand the distribution of labels and the impact of preprocessing techniques.

## Model Saving

The best performing model (K-Nearest Neighbors) is saved as a serialized file (`finalized_model.sav`) using the `pickle` library. You can load this model later for inference or further analysis.

## Contributing

Contributions to this project are welcome. If you find any issues or have ideas for improvements, please feel free to submit a pull request or open an issue.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use and modify the code for your own purposes.

## Acknowledgments

This project is inspired by the need for accurate classification of chest X-ray images for medical diagnosis. It utilizes the power of machine learning and open-source libraries to achieve reliable results.
