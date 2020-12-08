# image_caption_generator

Image caption generator based on Tensorflow 2.3
This aim of this project to build a deep learning model capable of generating a predicted caption given an image input. The dataset used is the Flickr8k dataset, which contains 6000 images with 5 associated captions each. A merged model of an InceptionV3 feature extractor and a LSTM recurrent neural network sequence processor is used to generate caption predictions for images.

The file 'caption_generator.ipynb' contains the code of the project where as the file 'model.png' illustrates the structure and design of the merged model. The model was also subsequently deployed as a webapp via Streamlit, with the code used in 'webapp.py'.

The model is successful in generating a caption that makes sense to the reader, however the descriptions are not entirely accurate due to overfitting of the small dataset. The model would be greatly improved given a much larger dataset and longer timeframe to train, for example, using the MS-COCO dataset, which contains over 300k images.
