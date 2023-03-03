#ifndef DETECTION_H
#define DETECTION_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;

class ssd_detector
{
private:
    const size_t inWidth = 300;
    const size_t inHeight = 300;
    const double inScaleFactor = 1.0/127.5;
    const float confidenceThreshold = 0.7;
    const std::string configFile = "/media/rahul/a079ceb2-fd12-43c5-b844-a832f31d5a39/kitti-360/Project/model/ssd_mobilenet_v2_coco_2018_03_29.pbtxt";
    const std::string modelFile = "/media/rahul/a079ceb2-fd12-43c5-b844-a832f31d5a39/kitti-360/Project/model/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb";
    const std::string classFile = "/media/rahul/a079ceb2-fd12-43c5-b844-a832f31d5a39/kitti-360/Project/model/coco_class_labels.txt";
    std::vector<std::string> classes;
    cv::dnn::Net net = cv::dnn::readNetFromTensorflow(modelFile, configFile);
    void loadClassFile();
    void display_text(cv::Mat& img, std::string text, int x, int y);
public:
    ssd_detector()
    {
        loadClassFile();
    };
    cv::Mat detect_objects(cv::Mat);
    void display_objects(cv::Mat&, cv::Mat, cv::Mat&, float);
    ~ssd_detector()
    {
        std::cout<<"detector destroyed properly"<<std::endl;
    }
};


#endif