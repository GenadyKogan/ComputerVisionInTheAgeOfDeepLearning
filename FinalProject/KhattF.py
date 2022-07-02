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
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.layers import Flatten, Dense, MaxPooling2D, Conv2D, Dropout, Input
from tensorflow.keras.applications import vgg16, ResNet50, Xception, EfficientNetB0
# Set a seed value
from numpy.random import seed
seed(1)
tf.random.set_seed(1)

class Khatt():
    def __init__(self):
        """
        Ctor for Classifier
        """
        # ----------------------- Load_DataSet
        self.df_test = self.load_dataSet_to_df('test')
        self.df_train = self.load_dataSet_to_df('train')

        # ----------------------- Experiment 1 and 2
        # data/cnn.h5' name of save cnn model : 32 number input shape ->(32, 32)
        self.model_dict={'data/cnn.h5':400, "data/vgg16_model.h5":400, "data/resNet50.h5":400,
                         "data/Xception.h5":400, "data/EfficientNetB0.h5":400}
        self.age_class = ['16-25', '26-50']
        choice = 0

        #os.mkdir('data')
        # index 0 to CNN, 1 to VGG16, 2 to ResNet50, 3 to Xception, 4 to EfficientNetB0 model
        model = self.run_model_type(choice)

        # ----------------------- Results
        #self.write_all_the_results_to_txt()

    # ----------------------- Experimental settings
    def load_dataSet_to_df(self, name_split):

        # Load images data from a folder named "KHATT" to dataframe
        df = pd.DataFrame({'image_path': glob('KHATT/*/' + name_split + '/*', recursive=True)})
        df['group_name'] = df['image_path'].apply(lambda x: x.split('\\')[1]).astype(int)
        return df

    def y_preprocessing(self,df):

        # Convert a dataset label to binary (2 to 0, 3 to 1)
        df['group_name'] = df['group_name'].replace(2, 0).replace(3, 1)
        
        # Convert a dataset images and labels to np.array
        y = np.array(df['group_name'].values.tolist())
        return y

    def get_split_window_for_specific_image(self, image_name):

        # read the image
        image = cv2.imread(image_name)  # your image path
        tmp = image  # for drawing a rectangle
        # define the stepSize and window size (width,height)
        stepSize = 200

        # define the window size (width,height)
        (w_width, w_height) = (400, 400)
        for y in range(0, image.shape[0] -w_height , stepSize):
            for x in range(0, image.shape[1]- w_width, stepSize):

                # draw window on image
                cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (0,0,0))
        return np.array(tmp)

    def x_preprocessing(self, df, size):
        df_data_image = pd.DataFrame(columns=['img_data'])
        for image_name in df['image_path']:

            # Extract the patches by moving a 400 × 400 pixel sliding window in a step of 200 pixels in vertical and horizontal directions.
            tmp = self.get_split_window_for_specific_image(image_name)
            gray = tmp[:, :, 0]

            # Resize the image to a uniform size (400,400)
            input_img_resize = cv2.resize(gray, (size, size))

            # Size after reshape is (32, 32, 3)
            input_img_reshape = np.reshape(input_img_resize, (size, size, 1))

            # print(input_img_reshape)
            # Append all the data to dataframe
            df_data_image = df_data_image.append({'img_data': input_img_reshape}, ignore_index=True)

        # Convert a dataset images and labels to np.array
        x = np.array(df_data_image['img_data'].values.tolist())

        # Convert data to float and normalize by pixels
        x = (x.astype(np.float16)) / 255
        return x

    def get_train_and_test(self, choice):

        # Get the name of a specific model
        name_model_file = list(self.model_dict.keys())[choice]

        # Get the size of a specific model
        size=self.model_dict.get(name_model_file)

        # Get preprossing of the train set
        y_train = self.y_preprocessing(self.df_train)
        x_train = self.x_preprocessing(self.df_train, size)

        # Get preprossing of the test set
        y_test = self.y_preprocessing(self.df_test)
        x_test = self.x_preprocessing(self.df_test,size)
        return y_train, x_train, y_test, x_test

    #--------------------------------------------------------------------------------- Experiment 1 and 2
    def save_model(self, history, model, name_history, name_model):
        np.save(name_history + '.npy', history.history)
        model.save(name_model)

    def load_models(self, name_history, name_model):
        model = load_model(name_model)
        history = np.load(name_history + '.npy', allow_pickle='TRUE').item()
        return history, model

    def get_cnn_model(self,size):
        '''
            :param size: An integer representing the size of the input layer.
            A method that builds a CNN.
            The architecture of the cnn model is : INPUT=>[CONV=>RELU=>CONV=>RELU=>POOL=>DO]*3=>FC=>RELU=>DO=>FC
            :return: model - CNN model
        '''
        # Define Sequential model
        model = keras.Sequential()

        # First iteration: CONV=>RELU=>CONV=>RELU=>POOL=>DO
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(size,size, 1)))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Second iteration: CONV=>RELU=>CONV=>RELU=>POOL=>DO
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Third iteration: CONV=>RELU=>CONV=>RELU=>POOL=>DO
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Layer FC => RELU => DO
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))

        # The last FC layer with 2 neurons
        model.add(Dense(1, activation='sigmoid'))

        # config the model with losses and metrics with model.compile()
        model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=["accuracy"])
        print(model.summary())
        return model

    def run_transfer_learning(self, preTrained_conv, name_save):
        # ------------------------ b) Freeze the required layers:
        # Freeze all the layers
        for layer in preTrained_conv.layers[:]:
            layer.trainable = False

        # Check the trainable status of the individual layers
        # for layer in preTrained_conv.layers:
        # print(layer, layer.trainable)

        # ------------------- c) Add a classifier on top of the convolutional base - add a fully connected layer followed by a sigmoid with one outputs
        # Create the model
        model = keras.Sequential()
        # Add the vgg convolutional base model
        model.add(preTrained_conv)

        # Add new layers
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid', name='output'))

        # Show a summary of the model. Check the number of trainable parameters
        model.summary()

        # Configure the model for training
        model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=['accuracy'])
        print(model.summary())

        # use the train feature vectors
        model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=20, batch_size=150 )

        # ---------------------- e) UnFreeze all the layers
        for layer in preTrained_conv.layers[:]:
            layer.trainable = True

        # Check the trainable status of the individual layers
        # for layer in preTrained_conv.layers:
        # print(layer, layer.trainable)

        # ------------------ e) train the model for additional 20 epochs
        model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=20, batch_size=150)

        # save model
        model.save(name_save)
        return model

    def run_model_type(self, choice):
        """
        :param choice: An integer representing the type of model you want to run.
        :return: model: CNN/ VGG16/ ResNet50/ Xception/EfficientNetB0 model
        """
        # Get the name of a specific model
        name_model_file = list(self.model_dict.keys())[choice]

        # Get the size of a specific model
        size=self.model_dict.get(name_model_file)
        self.y_train, self.x_train, self.y_test, self.x_test =self.get_train_and_test(choice)
        if choice == 0:

            #------------ CNN model
            model = self.get_cnn_model(size)
            history = model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=50, batch_size=150)

            # Save model
            self.save_model(history, model, 'data/history_cnn', name_model_file)

            # Load model
            history, model = self.load_models('data/history_cnn', name_model_file)
            self.plot_loss_and_accurecy_model(history, 'accuracy', 'Accuracy_cnn')
            self.plot_loss_and_accurecy_model(history, 'loss', 'Loss_cnn')
        elif choice>0 and choice<5:

            # List that cointains all name of transfer learning
            transfer_learning_type_list=[vgg16.VGG16, ResNet50, Xception, EfficientNetB0 ]
            
            # Get a specific transfer learning name
            transfer_learning_name = transfer_learning_type_list[choice-1]
            print(transfer_learning_name)

            # ------------------------ a) Load the pre-trained model: CNN/ VGG16/ ResNet50/ Xception/EfficientNetB0
            img_input = Input(shape=(size, size, 1))
            img_conc = tf.keras.layers.Concatenate()([img_input, img_input, img_input])
            conv = transfer_learning_name(weights='imagenet', include_top=False, input_tensor=img_conc)
            model = self.run_transfer_learning(conv, name_model_file)

        return model

    #--------------------------------------------------------------------------------- Results
    def plot_loss_and_accurecy_model(self, history, name, save_nane ):
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

        plt.legend(['Training ' + name, 'test '+ name], loc='upper right')
        plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25, wspace=0.35)
        plt.show()
        plt.close()

    def plot_confusion_matrix(self, con, model_name):
        """
                :param con: confusion_matrix - Confusion matrix of nn run results on testing set.
                :param model_name: Name of the model
                :return: Plotting of Confusion matrix.
                """
        group_counts = ["{0:0.0f}".format(value) for value in con.flatten()]
        group_counts = np.asarray(group_counts).reshape(2, 2)
        ax = sns.heatmap(con, annot=group_counts, fmt='', xticklabels=self.age_class, yticklabels=self.age_class,
                         cmap='Blues')
        ax.set_title(model_name);
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label');

        ## Display the visualization of the Confusion Matrix.
        plt.show()

    def write_results_to_file(self,df):
        """
        A method that creates a text file to evaluate the model.
        :param con: confusion_matrix - Confusion matrix of model.
        :param name_type_model: name of the model.
        :param age_class : list of class type ['16-25', '26-50']
        :param acc_score : Model accuracy
        :return: File named “results.txt” that contains :
                a. name of the model =…
                b. Summarization for each model
                c. Overall accuracy for each model
                d. Accuracy reached by the classifier for each of the letters (2 different ages) in the format
                    Letter Accuracy
                    0   …
                    1   …
                e. Confusion_matrix
        """
        file= open('results.txt', 'w')
        for row in df.iloc:
            
            # Loud model
            model = keras.models.load_model(str(row['name_model_file']))
            con= row['confusion_matrix']

            # Now the normalize the diagonal entries
            cm = con.astype('float') / con.sum(axis=1)[:, np.newaxis]

            # The diagonal entries are the accuracies of each class
            letter, accuracy = self.age_class, cm.diagonal()
            accuracy_letter_dict = dict(zip(letter, accuracy))

            # Write the data
            # ------------- a
            file.write("Name of the model: " + str(row['name_model_file']) + '\n\n')

            # ------------- b
            model.summary(print_fn=lambda x: file.write(x + '\n'))

            # ------------- c
            file.write("Overall accuracy of the model: " + str(row['acc_score']) + '\n\n')

            # ------------- d
            file.write("The accuracy for each group: " + '\n\n')
            file.write('Letter' + '     ' + 'Accuracy' + '\n\n')
            for key, value in accuracy_letter_dict.items():
                file.write(str(key) + '     ' + str(value) + '\n\n')

            #-------------- e
            file.write('Confusion matrix of model: ' + '\n\n')
            file.write(str(con)  + '\n\n')

        file.close()

    def model_evaluation_to_file(self, name_model_file):
        """
        :param name_model_file: List with model names
        A method that reports the results:Includes the name_model_file, confusion_matrix, acc_score.
        And saving the results in files.
        :return: list contains columns named: 'name_model_file', 'confusion_matrix', 'acc_score'
        """
        # Loud model
        model = keras.models.load_model(name_model_file)

        # Get the size of a specific model
        size = self.model_dict.get(name_model_file)

        # Get preprossing of the test set
        self.y_test = self.y_preprocessing(self.df_test)
        self.x_test = self.x_preprocessing(self.df_test, size)

        # Get prediction of model
        prediction = model.predict(self.x_test)

        # Convert the prediction vector to a binary list
        binary_predictions = [0 if i[0] <= 0.5 else 1 for i in prediction]

        # Get confusion matrix of model
        con = confusion_matrix(self.y_test, binary_predictions)

        # Printing precision,recall,accuracy score etc
        print(classification_report(self.y_test, binary_predictions, target_names = self.age_class))

        # Overall accuracy
        acc_score = accuracy_score(self.y_test, binary_predictions)
        self.plot_confusion_matrix(con, name_model_file)
        return [name_model_file, con, acc_score]

    def write_all_the_results_to_txt(self):
        '''
        :return: df_result_model - Data frame contains columns named: 'name_model_file', 'confusion_matrix', 'acc_score'
        '''
        name_model_file = list(self.model_dict.keys())

        # Convert a list to a data frame with model names
        df_result_model = pd.DataFrame({'name_model_file': name_model_file})
        
        # For each model name return the results: name_model_file, confusion_matrix , acc_score
        df_result_model = df_result_model['name_model_file'].apply(lambda r: pd.Series(self.model_evaluation_to_file(r),
                                                                             index=['name_model_file', 'confusion_matrix', 'acc_score']))
        print(df_result_model)

        # Writing results into files
        self.write_results_to_file(df_result_model)

if __name__ == "__main__":
    Khatt()