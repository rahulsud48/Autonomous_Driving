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

#include <torch/torch.h>
#include <torch/script.h>
#include <cmath> 

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

class DataEncoder
{
private:
    int img_w = 300; 
    int img_h = 300;
    int num_classes;
    int num_fms = 5;
    float cls_threshold=0.7f;
    float nms_threshold=0.3f;
    std::vector<float> anchor_areas;
    std::vector<float> aspect_ratios{0.5,1,2};
    std::vector<float> scales;
    std::vector<int> fm_sizes;
    std::map<std::string, int> img_size;
    torch::Tensor anchor_boxes_tensor;
    void createParameters();
    torch::Tensor generate_anchor_grid(int img_w, int img_h, int fm_size, torch::Tensor anchors);
    torch::Tensor generate_anchors(float anchor_area, std::vector<float> aspect_ratios, std::vector<float> scales);
    torch::Tensor convert_2dvec_to_torch_tensor(std::vector<std::vector<float>> array_2d);

public:
    DataEncoder();
    torch::Tensor decode_boxes(const torch::Tensor& deltas, const torch::Tensor& anchors);
    torch::Tensor compute_nms(const torch::Tensor& boxes, const torch::Tensor& conf, float threshold = 0.5);

    void decode(
        torch::Tensor loc_pred, 
        torch::Tensor cls_pred, 
        std::vector<std::map<int, std::vector<std::vector<float>>>>& output_boxes, 
        std::vector<std::map<int, std::vector<int>>>& output_classes, 
        int batch_size
    );
};


class ssd_detector_torch
{
private:
    std::string model_path = "/media/rahul/a079ceb2-fd12-43c5-b844-a832f31d5a39/Projects/autonomous_cars/Object_Detector_for_road/SSD_Detector_for_road_training/ssd_libTorch/build/tiny_model.pt";
    torch::jit::script::Module ssd_detector;
    void load_model();
public:
    ssd_detector_torch()
    {
        load_model();
    };
    void transform_image(cv::Mat* img, torch::Tensor* img_tensor);
    void detect(torch::Tensor img, torch::Tensor boxes, torch::Tensor classes);
    void display_objects(
        cv::Mat& img, 
        std::vector<std::map<int, std::vector<std::vector<float>>>> output_boxes, 
        std::vector<std::map<int, std::vector<int>>> output_classes,
        int batch_size,
        cv::Mat& depth_map
    );
    void display_text(cv::Mat& img, std::string text, int x, int y);
    ~ssd_detector_torch()
    {
        std::cout<<"detector destroyed properly"<<std::endl;
    };
};

#endif