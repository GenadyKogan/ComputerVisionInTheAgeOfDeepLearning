import cv2, sys, os
from glob import glob
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import  Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report
# Set a seed value
from numpy.random import seed
seed(1)
tf.random.set_seed(1)

class NN_Classifier_HHD():
    def __init__(self,folder):

        """
        Ctor for Classifier
        """
        # ----------------------- 1 preprocessing
        self.df_data_image = self.__preprocessing(folder)
       
        # ----------------------- 1 spliting data
        self.all_train, self.all_val, self.all_test = self.__split_dataset()
      
        self.x_train, self.y_train = self.__dataset_to_npArray_and_OneHotEncoder(self.all_train)
        self.x_val, self.y_val = self.__dataset_to_npArray_and_OneHotEncoder(self.all_val)
        self.x_test, self.y_test = self.__dataset_to_npArray_and_OneHotEncoder(self.all_test)
        
        # ----------------------- 2 training
        os.mkdir('data')
        name_history, name_model = 'hhd_NN_history_5_1', 'hhd_NN_5_1'

        # Get a configuration type of the model
        model = self.__get_configuration_type(5_1)

        # train the model with model.fit()
        history = model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), epochs=50, batch_size=256)

        # Save model
        self.__save_model(history, model, name_history, name_model)
        
        # ----------------------- 3 evaluation nnn
        # Load model
        model, history = self.__load_models(name_history, name_model)
        print(model.summary())
        self.__plot_loss_and_accurecy_model(history, 'accuracy', 'Accuracy_'+name_model)
        self.__plot_loss_and_accurecy_model(history, 'loss', 'Loss_'+ name_model)
        self.__model_evaluation_to_file(model, name_model)
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

        # print(img.shape[:2])
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

        # print(scaled_img.shape[:2])
        return scaled_img

    def __preprocessing(self, folder):
        """
        A method that converts all letters to a uniform size.
            a. Convert the image to grayscale.
            b. Add white padding to the image so that its size is square.
            c. Move the image to a uniform size (32,32).
            d. Turning an image into a negative.
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
            
            # color = [0,0,0]  # black color
            dst = self.__add_white_padding(input_img_cvtColor, color)

            # c- Resize the image to a uniform size (32,32)
            input_img_resize = cv2.resize(dst, (32, 32))

            # d- Turning an image into a negative
            input_img_negative = 255 - input_img_resize

            # # Append all the data to dataframe
            df_data_image = df_data_image.append(
                {'img_data': input_img_negative, 'img_labels': int(img.split("\\")[-2]), 'dir_path': img},
                ignore_index=True)
        print(df_data_image)
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
            train = self.df_data_image[self.df_data_image['img_labels'] == label].sample(frac=0.8, random_state = 42)

            # not_train - Contains 20% data that are not train
            not_train = self.df_data_image[self.df_data_image['img_labels'] == label].drop(train.index)

            # Of the remaining 20%, half go to val and half to test
            test = not_train.sample(frac=0.5, random_state = 42)
            
            val = not_train.drop(test.index)

            # Append all the data to dataframes
            all_train, all_val, all_test = all_train.append(train), all_val.append(val), all_test.append(test)
        print("The division of the data into train, validation and test was successful!")
        return all_train, all_val, all_test

    def __dataset_to_npArray_and_OneHotEncoder(self, df):
        """
        A method that replaces the data with a new format
        :param df: Data frame of training, validation or testing
        :return: X - Vector of the image after reshape to size (img_data.shape[0], 32*32), conversion to np array
                    and normalize by pixels.
                 y - Image label (0-26 classification) after conversion to np array and one hot.
        """
        # Convert a dataset images and labels to np.array
        x = np.array(df['img_data'].values.tolist())
        y = np.array(df['img_labels'].values.tolist())

        # Size after reshape is (img_data.shape[0], 32*32)
        x = x.reshape(x.shape[0], -1)

        # Convert data to float and normalize by pixels
        x = (x.astype(np.float16))/255.

        #  Transform labels to one hot encoding
        y = tf.keras.utils.to_categorical(y, num_classes=len(set(df['img_labels'])))
        return x, y
    #------------------------------------------------------------------------------ 2 training
    def __save_model(self, history, model, name_history, name_model):
        np.save('data/'+ name_history + '.npy', history.history)
        model.save('data/'+ name_model)

    def __load_models(self, name_history, name_model):
        model = load_model('data/'+ name_model)
        history = np.load('data/'+ name_history + '.npy', allow_pickle='TRUE').item()
        return model, history

    def __plot_loss_and_accurecy_model(self, history, name, save_nane ):
        """
        :param history: The history results of the selected model.
        :param name: Name of the curve we want to display: Loss or Accurecy.
        :param save_nane: Name to save the results as an image.
        :return: Plotting of Loss or Accurecy.
        """
        plt.plot(history[name],'b',label='train_'+name)
        plt.plot(history['val_'+name],'r',label='val_'+name)
        plt.title(name + ' Curves')
        plt.ylabel(name)
        plt.xlabel('Epochs')
        plt.savefig('data/'+ save_nane + '.png')

        plt.legend(['Training ' + name, 'Validation '+ name], loc='upper right')
        plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25, wspace=0.35)
        plt.show()
        plt.close()

    def __get_model_without_regulation(self):
        '''
        A method that builds a neural network without regulation.
        :return: model - Type of model built.
        '''
        # Define Sequential model
        model = keras.Sequential()

        # Input layer
        model.add(Dense(1024, activation='relu')) 

        # Hidden layer 1
        model.add(Dense(512)) 

        # Hidden layer 2
        model.add(Dense(512)) 

        # Output layer
        model.add(Dense(27, activation='softmax')) 

        # config the model with losses and metrics with model.compile()
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def __get_model_with_regularization(self, type_reg, lambda_value):
        """
        A method that adds regularization to each layer except the output layer.
        :param type_reg: Gets the type of regularization of the model: L1 or L2.
        :param lambda_value: Accepts the Lambda value.
        :return: model - Type of model built.
        """
        # Define Sequential model
        model = keras.Sequential()

        # Input layer
        model.add(Dense(1024, activation='relu', kernel_regularizer=type_reg(lambda_value))) 

        # Hidden layer 1
        model.add(Dense(512, kernel_regularizer=type_reg(lambda_value))) 

        # Hidden layer 2
        model.add(Dense(512, kernel_regularizer=type_reg(lambda_value))) 

        # Output layer
        model.add(Dense(27, activation='softmax')) 

        # config the model with losses and metrics with model.compile()
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def __get_model_with_dropout(self):
        """
        A method that adds a dropout with p = 0.5 in each layer except the output layer.
        :return: model - Type of model built.
        """
        # Define Sequential model
        model = keras.Sequential()

        # Input layer
        model.add(Dense(1024, activation='relu')) 

        # Dropout layer
        model.add(Dropout(0.5)) 

        # Hidden layer 1
        model.add(Dense(512)) 
        model.add(Dropout(0.5))
        
        # Hidden layer 2
        model.add(Dense(512)) 
        model.add(Dropout(0.5))

        # Output layer
        model.add(Dense(27, activation='softmax'))

        # config the model with losses and metrics with model.compile()
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def __get_model_with_dropout_and_reg(self, lambda_value):
        """
        A method that adds a dropout with p = 0.5 and type L2 regulation in each layer except the output layer.
        :param lambda_value: Accepts the Lambda value
        :return: model - Type of model built
        """
        # Define Sequential model
        model = keras.Sequential()

        # Input layer
        model.add(Dense(1024, activation='relu', kernel_regularizer=l2(lambda_value)))

        # Dropout layer
        model.add(Dropout(0.5)) 

        # Hidden layer 1
        model.add(Dense(512, kernel_regularizer=l2(lambda_value))) 
        model.add(Dropout(0.5))

        # Hidden layer 2
        model.add(Dense(512, kernel_regularizer=l2(lambda_value))) 
        model.add(Dropout(0.5))

        # Output layer
        model.add(Dense(27, activation='softmax')) 
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def __get_configuration_type(self, choice):
        """
        A method that aims to return a model according to the type of configuration.
        :param choice: An integer representing the type of configuration you want to run.
        :return: model - Type of model built.
        """
        if choice == 1:
            return self.__get_model_without_regulation()
        elif choice == 2_1:
            return self.__get_model_with_regularization(l1, 0.001)
        elif choice == 2_2:
            return self.__get_model_with_regularization(l1, 0.01)
        elif choice == 3_1:
            return self.__get_model_with_regularization(l2, 0.001)
        elif choice == 3_2:
            return self.__get_model_with_regularization(l2, 0.01)
        elif choice == 4:
            return self.__get_model_with_dropout()
        elif choice == 5_1:
            return self.__get_model_with_dropout_and_reg(0.001)
        else: 
            return self.__get_model_with_dropout_and_reg(0.01)

        # -------------------------------------------------------------------------------- 3 evaluation nn
    def __plot_confusion_matrix(self, con):
        """
        :param con: confusion_matrix - Confusion matrix of nn run results on testing set.
        :return: Plotting of Confusion matrix.
        """
        plt.figure(figsize=(6, 6))
        sns.heatmap(con, annot=True, cbar=True)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()


    def __wrire_cm_to_csv(self, con):
        """
        A method that creates a csv file to evaluate the model
        :param con: confusion_matrix - Confusion matrix of knn run results on testing set.
        :return: File named “confusion_matrix.csv”
        """
        pd.DataFrame(con).to_csv("confusion_matrix.csv")

    def __write_to_result_txt(self, con, name_configuration_final_model):
        """
        A method that creates a text file to evaluate the model.
        :param con: confusion_matrix - Confusion matrix of knn run results on testing set.
        :param name_configuration_final_model: Configuration name of the final model.
        :return: File named “results.txt” that contains :
                a. Configuration name of the final model =…
                b. Image with loss curve on loss of training and validation for the final model.
                c. Accuracy reached by the classifier for each of the letters (27 different letters) in the format
                    Letter Accuracy
                    0   …
                    1   …
                    …
                    26  …
                    _________
                    avg ...

        """

        # Now the normalize the diagonal entries
        cm = con.astype('float') / con.sum(axis=1)[:, np.newaxis]

        # The diagonal entries are the accuracies of each class
        letter, accuracy = list(set(self.df_data_image['img_labels'])), cm.diagonal()
        accuracy_letter_dict = dict(zip(letter, accuracy))
        avg = str(sum(accuracy_letter_dict.values()) / len(accuracy_letter_dict.keys()))
        with open('results.txt', 'w') as file:

            # Write the data
            # ------------- a
            file.writelines("Configuration of the final model: " + str(name_configuration_final_model) + '\n\n')

            # ------------- b
            # to do
            # ------------- c
            file.writelines('\n\nLetter' + '     ' + 'Accuracy' + '\n\n')

            for key, value in accuracy_letter_dict.items():
                file.write(str(key) + '     ' + str(value) + '\n\n')
            file.writelines('_______________________________________' + '\n\n')
            file.writelines('avg' + '     ' + str(avg) + '\n\n')
        file.close()

    def __model_evaluation_to_file(self, model, name_configuration_final_model):
        """
        A method that reports the results:Includes the confusion matrix, classification report
        And saving the results in files.
        :return: Save model evaluation results to files.
        """
        preds=model.predict(self.x_test)
        pred_list=[ int(np.argmax(pred)) for pred in preds]
        y_test_list=[ int(np.argmax(y)) for y in self.y_test]
        con = confusion_matrix(y_test_list, pred_list)
        self.__plot_confusion_matrix(con)

        # Printing precision,recall,accuracy score etc
        print(classification_report(y_test_list, pred_list))

        # Writing results into files
        self.__write_to_result_txt(con, name_configuration_final_model)
        self.__wrire_cm_to_csv(con)

if __name__ == "__main__":

    path = sys.argv[1]

    # Check If Path Exists
    if os.path.exists(path):
        print("path exists")
        NN_Classifier_HHD(path)
    else:
        print("path not exists")