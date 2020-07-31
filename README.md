# YOLOV2
After reading the YOLOV2 paper, try to use pytorch to build a YOLOV2 model
## Train
1.Modify the class_name_to_index.json file to construct the correspondence between the category and its index. 
2.Modify the conf.json file. img_file_path is equivalent to the JPEGImages folder in voc2012, txt_file_path is equivalent to the \ImageSets\Main folder in voc2012, of which only train.txt and val.txt are needed, and xml_file_path is equivalent to the Annotations folder in voc2012.

