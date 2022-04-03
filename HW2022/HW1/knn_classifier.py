import cv2, sys, os
from glob import glob
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns


class KNN_Classifier_HHD():
    def __init__(self,folder):

        """
        Ctor for Classifier
        """
        # ----------------------- 1 preprocessing
        self.df_data_image = self.__preprocessing(folder)
       
        # ----------------------- 1 spliting data
        self.all_train, self.all_val, self.all_test = self.__split_dataset()
        self.x_train, self.y_train = self.__dataset_to_npArray(self.all_train)
        self.x_val, self.y_val = self.__dataset_to_npArray(self.all_val)
        self.x_test, self.y_test = self.__dataset_to_npArray(self.all_test)
       
        # ----------------------- 2 training
        # Finding best fit k value
        self.k = self.__get_best_fit_k_value_acc_val()

        # ----------------------- 3 evaluation knn
        # Applay KNeighbors classifier
        self.pred = self.__KNeighbors_classifier()
        self.__model_evaluation_to_file()
    # ------------------------------------------------------------------------------ 1 preprocessing
    def __add_white_padding(self, img, color):
        """
        b. A method that adds white padding to the image so that its size is square.
            - If the image width is small from a height, Padding to the right and left should be added.
            - Otherwise, if a width is greater than height, Padding should be added from top to bottom.
        :param img: Image from the dataset
        :param color: The color of the padding to be added
        :return: Image after adding padding
        """
        h, w = img.shape[:2]
        sh, sw = (max(h, w), max(h, w))
      
        # interpolation method
        if h > sh or w > sw:  # shrinking image
            interp = cv2.INTER_AREA
        else:  # stretching image
            interp = cv2.INTER_CUBIC
        # aspect ratio of image
        aspect, saspect = float(w) / h, float(sw) / sh

        if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image (width < height)
            new_h = sh
            new_w = np.round(new_h * aspect).astype(int)
            pad_horz = float(sw - new_w) / 2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0

        elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image (width > height)
            new_w = sw
            new_h = np.round(float(new_w) / aspect).astype(int)
            pad_vert = float(sh - new_h) / 2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
        # set pad color
        padColor = color

        # scale and pad
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                        borderType=cv2.BORDER_CONSTANT, value=padColor)

        return scaled_img

    def __preprocessing(self, folder):
        """
        A method that converts all letters to a uniform size.
            a. Convert the image to grayscale.
            b. Add white padding to the image so that its size is square
            c. Move the image to a uniform size (32,32).
        :param folder: Path to folder with image database
        :return: df_data_image - A data frame containing three columns:
                    'img_data': Vector of the image after pre-processing.
                    'img_labels': Image Label (Classification Class 0-26).
                    'image_name': Name of the image received as input.
        """
        # Defining new data frames
        df_data_image = pd.DataFrame(columns=['img_data', 'img_labels', 'dir_path'])
        dir_path = glob(folder + "/**/*.png", recursive=True)
        for img in dir_path:

            # Load image
            input_img = cv2.imread(img)

            # a- Convert image to grayscale
            input_img_cvtColor = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

            # b. Add white padding to the image so that its size is square
            color = [255, 255, 255]  # white color

            dst = self.__add_white_padding(input_img_cvtColor, color)
            
            # c- Resize the image to a uniform size (32,32)
            input_img_resize = cv2.resize(dst, (32, 32))

            # # Append all the data to dataframe
            df_data_image = df_data_image.append(
                {'img_data': input_img_resize, 'img_labels': int(img.split("\\")[-2]), 'dir_path': img},
                ignore_index=True)

        return df_data_image
    # ------------------------------------------------------------------------------ 1 spliting data
    def __split_dataset(self):
        """
        A method that randomly divides the data set into three groups: 80% training, 10% validation and  10% testing sets.
            Images of each letter should appear in each group in% 10:% ​​10:% ​​80 ratio.
        :return: all_train, all_val, all_test - Three data frames for training, validation and testing.
                    Each data frame contains:
                    'img_data': Vector of the image after pre-processing.
                    'img_labels': Image Label (Classification Class 0-26).
                    'image_name': Name of the image received as input.
        """
        # Defining new data frames for all train, val, test
        all_train = pd.DataFrame()
        all_val, all_test = all_train, all_train

        # A loop that runs on all types of labels (0-26) and divides each label to 80% train, 10% val and 10% test
        for label in set(self.df_data_image['img_labels']):

            # Take 80% of the data on a particular label to train
            train = self.df_data_image[self.df_data_image['img_labels'] == label].sample(frac=0.8)

            # not_train - Contains 20% data that are not train
            not_train = self.df_data_image[self.df_data_image['img_labels'] == label].drop(train.index)

            # Of the remaining 20%, half go to val and half to test
            test = not_train.sample(frac=0.5)
            val = not_train.drop(test.index)

            # Append all the data to dataframes
            all_train, all_val, all_test = all_train.append(train), all_val.append(val), all_test.append(test)
        print("The division of the data into train, validation and test was successful!")
        return all_train, all_val, all_test

    def __dataset_to_npArray(self, df):
        """
        A method that replaces the data with a new format
        :param df: Data frame of training, validation or testing
        :return: X - Vector of the image after reshape to size (img_data.shape[0], 32*32) and conversion to np array.
                 y - Image label (0-26 classification) after conversion to np array.
        """
        # Convert a dataset to np.array
        img_data = np.array(df['img_data'].values.tolist())

        # Size after reshape is (img_data.shape[0], 32*32)
        # (40000, 32, 32)
        X = img_data.reshape(img_data.shape[0], -1)
        y = np.array(df['img_labels'].values.tolist())
        return X, y

    #------------------------------------------------------------------------------ 2 training
    def __plot_relationship_k_vs_val_acc(self,df_results_k):
        """
        A method that creates a diagram of the relationship between k and the accuracy of validation and training set.
        :param df_results_k:
            data frame containing the following columns: k, val_acc, train_acc, error_rate.
        :return: Plotting the relationship between k and the accuracy of validation and training.
        """
        train_accuracy, val_accuracy = df_results_k['train_acc'].values.tolist(), df_results_k['val_acc'].values.tolist()
        neighbors = np.arange(1, 16, 2)

        # Set Main Title
        plt.title('KNN Neighbors')

        # Set X-Axis Label
        plt.xlabel('Neighbors\n(#)')

        # Set Y-Axis Label
        plt.ylabel('Accuracy\n(%)', rotation=0, labelpad=35)

        # Place Testing Accuracy
        plt.plot(neighbors, val_accuracy, label='Val Accuracy')

        # Place Training Accuracy
        plt.plot(neighbors, train_accuracy, label='Training Accuracy')

        # Append Labels on Testing Accuracy
        for a, b in zip(neighbors, val_accuracy):
            plt.text(a, b, str(round(b, 2)))

        # Add Legend
        plt.legend()

        # Generate Plot
        plt.show()

    def __plot_error_rate_vs_k(self, df_results_k):
        """
        A method that creates a chart that shows the error rate values vs k for validation set.
        :param df_results_k:
            data frame containing the following columns: k, val_acc, train_acc, error_rate.
        :return: Plotting the error rate vs k graph.
        """
        error_rate = df_results_k['error_rate'].values.tolist()
        plt.figure(figsize=(12, 6))
        plt.plot(range(1,16,2), error_rate, marker="o", markerfacecolor="green",
                 linestyle="dashed", color="red", markersize=15)
        plt.title("Error rate vs k value", fontsize=20)
        plt.xlabel("k- values", fontsize=20)
        plt.ylabel("error rate", fontsize=20)
        plt.xticks(range(1, 16,2))
        plt.show()

    def __get_k_with_beat_val_acc(self,df_results_k):
        """
        :param df_results_k:
            data frame containing the following columns: k, val_acc, train_acc, error_rate.
        :return: k - Value K with the  highest accuracy on the validation set.
        """
        val_accuracy = df_results_k['val_acc'].values.tolist()

        # Max index
        max_index = int(np.argmax(val_accuracy))

        # list of k types
        k_index = list(range(1, 16, 2))

        # Get the name of the k
        k = k_index[max_index]
        print("Best K value = ", k)
        return k

    def __get_results_for_specific_k(self,k):
        """
        A method that performs training of the k-NN classifier on a specific value of k.
        :param k:  K value for KNN classifier
        :return: List of results for a specific K value containing a column: k, val_acc, train_acc, error_rate.
        """
        # Try KNeighbors with each of 'n' neighbors
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')

        # Fit the k-nearest neighbors classifier from the training dataset.
        knn.fit(self.x_train, self.y_train)
        
        # Val - Return the mean accuracy on the given test data and labels.
        test_acc = knn.score(self.x_val, self.y_val)

        # Training - Return the mean accuracy on the given test data and labels.
        train_acc = knn.score(self.x_train, self.y_train)

        # Predict the class labels for the provided data.
        predict_i = knn.predict(self.x_val)

        # Error Rate
        error_rate = np.mean(predict_i != self.y_val)
        print('Results of value K =', k, ' were obtained!')
        return [k, test_acc, train_acc, error_rate]

    def __get_best_fit_k_value_acc_val(self):
        """
        A method that performs the training of a k-NN classifier on different values ​​of k.
        Performs an evaluation of the results on a validation set for each k value,
        and selects the best k value that gives the highest accuracy on the validation set.
            - The classifier should be trained on the values ​​of k between 1 and 15 in steps of 2.
            - Euclidean distance should be used as the distance function.
        :return: k - Value K with the  highest accuracy on the validation set.
        """
        # Create a data frame with K values ​​in the range (1, 16, 2)
        df_results_k = pd.DataFrame(range(1, 16, 2), columns=['k'])

        # Accuracy results using range on 'n' values for KNN Classifier
        df_results_k = df_results_k['k'].apply(lambda r: pd.Series(self.__get_results_for_specific_k(r),
                                                                             index=['k', 'val_acc', 'train_acc', 'error_rate']))
      
        # Plotting the error rate vs k graph
        self.__plot_error_rate_vs_k(df_results_k)

        # Plotting the relationship batween k and the val accuracy
        self.__plot_relationship_k_vs_val_acc(df_results_k)

        # Get the value K that has the best val accuracy
        k = self.__get_k_with_beat_val_acc(df_results_k)

        return k

    #-------------------------------------------------------------------------------- 3 evaluation knn

    def __KNeighbors_classifier(self):
        """
        A method that performs a run of the knn classifier on the results of K with the optimal value.
        Performs knn results evaluation on testing set.
        :return: pred - A list containing the prediction on the testing data.
        """
        # Create KNN Object
        clf = KNeighborsClassifier(n_neighbors=self.k, weights='distance', metric='euclidean')

        # Training the model
        clf.fit(self.x_train, self.y_train)

        # Predict testing set
        pred = clf.predict(self.x_test)
        return pred

    def __wrire_cm_to_csv(self, con):
        """
        A method that creates a csv file to evaluate the model
        :param con: confusion_matrix - Confusion matrix of knn run results on testing set.
        :return: File named “confusion_matrix.csv”
        """
        pd.DataFrame(con).to_csv("confusion_matrix.csv")

    def __write_to_result_txt(self, con):
        """
        A method that creates a text file to evaluate the model.
        :param con: confusion_matrix - Confusion matrix of knn run results on testing set.
        :return: File named “results.txt” that contains :
                a. A k value that gives the highest accuracy in the format k =…
                b. Accuracy reached by the classifier for each of the letters (27 different letters) in the format
                    Letter Accuracy
                    0   …
                    1	…
                    …
                    26  …
        """
        # Now the normalize the diagonal entries
        cm = con.astype('float') / con.sum(axis=1)[:, np.newaxis]

        # The diagonal entries are the accuracies of each class
        letter, accuracy = list(set(self.df_data_image['img_labels'])), cm.diagonal()
        accuracy_letter_dict = dict(zip(letter, accuracy))
        with open('results.txt', 'w') as file:

            # Write the data
            file.writelines("k = "+ str(self.k)+ '\n\n')
            file.writelines('Letter'+'     '+ 'Accuracy'+ '\n\n')
            for key, value in accuracy_letter_dict.items():
                file.write(str(key) + '     '+ str(value)+ '\n\n' )
        file.close()

    def __plot_confusion_matrix(self, con):
        """
        :param con: confusion_matrix - Confusion matrix of knn run results on testing set.
        :return: Plotting of Confusion matrix.
        """
        plt.figure(figsize=(6, 6))
        sns.heatmap(con, annot=True, cbar=True)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def __model_evaluation_to_file(self):
        """
        A method that reports the results:Includes the confusion matrix, classification report
        And saving the results in files.
        :return: Save model evaluation results to files.
        """
        con = confusion_matrix(self.y_test, self.pred)
        self.__plot_confusion_matrix(con)

        # Printing precision,recall,accuracy score etc
        print(classification_report(self.y_test, self.pred))

        # Writing results into files
        self.__write_to_result_txt(con)
        self.__wrire_cm_to_csv(con)
    #----------------------------------------------------------------------------------------

if __name__ == "__main__":

    path = sys.argv[1]
    
    # Check If Path Exists
    if os.path.exists(path):
        print("path exists")
        KNN_Classifier_HHD(path)
    else:
        print("path not exists")

