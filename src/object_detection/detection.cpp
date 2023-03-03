#include "../object_detection/detection.h"

void ssd_detector::loadClassFile()
{
    std::ifstream ifs(classFile.c_str());
    std::string line;
    while (getline(ifs, line))
    {
        classes.push_back(line);
    }
    
};

cv::Mat ssd_detector::detect_objects(cv::Mat frame)
{
    const cv::Scalar meanVal(127.5, 127.5, 127.5);
    cv::Mat inputBlob = cv::dnn::blobFromImage(frame, inScaleFactor, cv::Size(inWidth, inHeight),
                                               meanVal, true, false);
    this->net.setInput(inputBlob);
    cv::Mat detection = net.forward("detection_out");
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    return detectionMat;
}

void ssd_detector::display_text(cv::Mat& img, std::string text, int x, int y)
{
    // Get text size
    int baseLine;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, &baseLine);
    // Use text size to create a black rectangle
    rectangle(img, Point(x,y-textSize.height-baseLine), Point(x+textSize.width,y+baseLine),
             Scalar(0,0,0),-1);
    // Display text inside the rectangle
    putText(img, text, Point(x,y-5), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,255), 1, LINE_AA);
}

void ssd_detector::display_objects(Mat& img, Mat objects, cv::Mat& depth_map ,float threshold = 0.25)
{
    // For every detected object
    for (int i = 0; i < objects.rows; i++){
        int classId = objects.at<float>(i, 1);
        float score = objects.at<float>(i, 2);
        // Recover original cordinates from normalized coordinates
        int x = static_cast<int>(objects.at<float>(i, 3) * img.cols);
        int y = static_cast<int>(objects.at<float>(i, 4) * img.rows);
        int w = static_cast<int>(objects.at<float>(i, 5) * img.cols - x);
        int h = static_cast<int>(objects.at<float>(i, 6) * img.rows - y);

        int x_mid = x + (w/2);
        int y_mid = y + (h/2);
        float depth = depth_map.at<float>(x_mid, y_mid);
        // Check if the detection is of good quality
        if (score > threshold){
            // display_text(img, classes[classId].c_str(), x, y);
            std::cout<<"the depth of: "<< classes[classId].c_str() << "is: "<< depth<<std::endl;
            display_text(img, classes[classId].c_str(), x_mid, y_mid);
            rectangle(img, Point(x,y), Point(x+w, y+h), Scalar(255,255,255), 2);
        }
    }
}