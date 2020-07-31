# YOLOV2
After reading the YOLOV2 paper, try to use pytorch to build a YOLOV2 model
## Train
1.Modify the class_name_to_index.json file to construct the correspondence between the category and its index.  
2.Modify the conf.json file. img_file_path is equivalent to the JPEGImages folder in voc2012, txt_file_path is equivalent to the \ImageSets\Main folder in voc2012, of which only train.txt and val.txt are needed, and xml_file_path is equivalent to the Annotations folder in voc2012.  
3.Modify the configuration file conf.json, you can modify the learning rate, epoch, batch_size, and training depth image size, etc. Above "test_img_path" are training configuration options.  
4.Run kmeans.py to cluster the bounding box, and the cluster_anchor_boxes.npy file will be generated in the project folder.  
5.Run train.py.  
## predict  
1.Modify the options under "test_img_path" in the conf.json file, test image storage path test_img_path, image size predict_model_input_img_size used in the test, etc.  
2.Run predict.py.
## predict result  
In order to test the model, I found some data on the Internet, the link is https://github.com/cosmicad/dataset, the prediction results are as follows：  
![pic1](https://github.com/1991yuyang/YOLOV2/blob/master/predict_result/BloodImage_00197.jpg?raw=true)
