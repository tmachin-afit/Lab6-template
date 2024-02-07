# Lab 6
This lab will once again be a lab from scratch which means we will have to do the seven-step process again. Our problem this time will be to determine the fandom of fan fictions. We will use the text of the fan fiction (the story) to determine which fandom it belongs to. We will use some flavor of RNNs to treat the words as a sequence instead of doing statistics on the sentence. 

## ACE Environment Setup
- CPU: 2
- Memory: 16 Gi
- GPU: 1 exclusive

## Step 1 Get Some Data
The instructor will provide the fan fiction data scraped from the site [Archive of Our Own](https://archiveofourown.org/) (AO3). This data was obtained with permission from the site to use for academic and educational purposes. However, if you plan to publish text from any specific fan fiction you must obtain consent from the author. Also, if you did not already know fan fictions can get weird so read the actual fan fictions at your own risk.  

The data is available on Canvas under the lab assignment. Use the small dataset `website_txt_splits_small.zip` for this lab (the large dataset is indeed large and takes a lot longer to process, but you can try it if you wish). After unzipping, there will be a train and test folder. The train and test folders will have directories for each fandom. Each fandom folder will contain `.txt` files with the actual fan fiction story. You can expect only english language but potentially odd unicode characters or special symbols. 

Put the data into the '/opt/data' directory, as this will be the same between student and professor environments, removing the necessity for excessive custom path objects. 

## Step 2 Measures of Success
Our problem this time is of multi class classification thus we are going to use `accuracy` as our metric. A note on the data, fan fictions are known for "crossovers" which have more than one fandom in the same story. When the data was created, if a fan fiction was in both fandoms it was copied to both folders for the respective fandoms. Thus, some stories will actually belong to two fandoms but are labelled as only one. Trying to determine if stories actually belong to both fandoms is beyond the scope of this lab.

Our baseline will be a model made by the instructor that did not use sequential data. This is a sensible baseline since this is the type of model that would be used if we did not know of RNNs. The baseline was a "linear regression" (technically a multivariate linear regression) model trained on word counts in the story. For reference, this baseline model achieved an accuracy of about 85% on the given test set. 

## Step 3 Prepare Data
Prepare you text data so that it can be ingested by an RNN. This means you need to make a tensor with three dimensions corresponding to the sample, time step (or more generally position in sequence), and finally the data. Remember there are many representations for words (tokens) such as sparse (one hot), or dense (an embedding). Also, which words are included in the vocabulary is up to the student. 

I suggest using the function `text_dataset_from_directory` from `tensorflow.keras.preprocessing`. This can generally make working with a large number of text files much easier. This [site](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text_dataset_from_directory) is the docs for `text_dataset_from_directory` which may prove useful. Also, the keras layer `TextVectorization` with docs [here](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/TextVectorization) may also be useful. 

## Step 4 Evaluation Method
The dataset contains a separate training and test set. You may use the training set for your validation splits as you see necessary. 

Remember to report your final accuracy on the test set which you only did after you decided your model had trained well enough.

## Step 5 Develop a model
In this section make a model that uses recurrent layers in some fashion. At this point you just want to get the model to compile. 

Separate this code from the next section with comments or if statements when you get a compiling model.

## Step 6 Overfit Model
Add more layers or other techniques to overfit the model to your training data. Note that RNNs can be much slower to train compared to CNNs or dense ANNs. Using GPU implementations of LSTM will be much faster and is automatically done if under certain LSTM arguments, see the [docs](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM) This problem is not complex enough to require training for a very long time and thus your final model should be able to train in < 10 minutes. This is not a hard requirement, but it will help you to try out more models and save the instructor time in grading. 

Again separate this code from the earlier code and later code by comments or if statements.

## Step 7 Regularize Model
Once the model overfits, you can then add regularization to achieve acceptable performance on your validation data. 

Again separate this model from the rest. If this model is the exact same as the one before it then note that in your code.

## Test Performance and Visualization
After you have a satisfactory model evaluate it against the test set and note the performance in a comment in the code. Also use the visualization code in the template to show the confusion matrices for the different classes. 

# Deliverables
Students should have commented code, including the breakdown for the 7-step process, model performance in steaps 5-7, and performance for the final test set. 
Please include the figure of your confusion matrix, and your trained model. DO NOT include the data. 
