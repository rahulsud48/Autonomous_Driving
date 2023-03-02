/* \author Aaron Brown */
// Create simple 3d highway enviroment using PCL
// for exploring self-driving car sensors

#include "sensors/lidar.h"
#include "render/render.h"
#include "processPointClouds.h"
// using templates for processPointClouds so also include .cpp to help linker
#include "processPointClouds.cpp"

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "opencv2/imgcodecs.hpp"

struct cameraCalibrations
{
public:
    std::vector<std::vector<float>> k_left{{788.629315, 0.000000, 687.158398}, 
                                            {0.000000, 786.382230, 317.752196}, 
                                            {0.000000, 0.000000, 0.000000}};

    std::vector<std::vector<float>> k_right{{785.134093, 0.000000, 686.437073}, 
                                            {0.000000, 782.346289, 321.352788}, 
                                            {0.000000, 0.000000, 0.000000}};

    std::vector<float> t_left{0.000000, 0.000000, 0.000000};
    std::vector<float> t_right{-0.594052, 0.007198, -0.010233};

    std::vector<std::vector<float>> r_left{{1.000000, 0.000000, 0.000000}, 
                                            {0.000000, 1.000000, 0.000000}, 
                                            {0.000000, 0.000000, 1.000000}};

    std::vector<std::vector<float>> r_right{{0.999837, 0.004862, -0.017390}, 
                                            {-0.004974, 0.999967, -0.006389}, 
                                            {0.017358, 0.006474, 0.999828}};
                    
};


void cityBlock(pcl::visualization::PCLVisualizer::Ptr& viewer, ProcessPointClouds<pcl::PointXYZI>* pointProcessorI, const pcl::PointCloud<pcl::PointXYZI>::Ptr& inputCloud)
{
    // ----------------------------------------------------
    // -----Open 3D viewer and display City Block     -----
    // ----------------------------------------------------

    // ProcessPointClouds<pcl::PointXYZI>* pointProcessorI = new ProcessPointClouds<pcl::PointXYZI>();
    // pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloud = pointProcessorI->loadPcd("../src/sensors/data/pcd/data_1/0000000000.pcd");
    // renderPointCloud(viewer,inputCloud,"inputCloud");

    bool render_box = true;
    constexpr float X{ 30.0 }, Y{ 6.5 }, Z{ 2.5 };
    pcl::PointCloud<pcl::PointXYZI>::Ptr filterCloud = pointProcessorI->FilterCloud(inputCloud, 0.1f, Eigen::Vector4f(-(X / 2), -Y, -Z, 1), Eigen::Vector4f(X, Y, Z, 1));
    // renderPointCloud(viewer,filterCloud,"filterCloud");
    std::pair<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr> segmentCloud = pointProcessorI->SegmentPlane(filterCloud, 25, 0.3);
    // renderPointCloud(viewer,segmentCloud.first,"obstCloud",Color(1,0,0));
    renderPointCloud(viewer,segmentCloud.second,"planeCloud",Color(0,1,0));

    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloudClusters = pointProcessorI->Clustering(segmentCloud.first, .5, 100, 1000);
    int clusterId = 0;
    std::vector<Color> colors = {Color(1,0,0), Color(0,1,0), Color(0,0,1)};
    for(pcl::PointCloud<pcl::PointXYZI>::Ptr cluster: cloudClusters){
        std::cout << "cluster size ";
        pointProcessorI->numPoints(cluster);
        renderPointCloud(viewer, cluster, "obstCloud"+std::to_string(clusterId), Color(0,0,1));//colors[clusterId%colors.size()]);
        
        if (render_box){
            Box box = pointProcessorI->BoundingBox(cluster);
            renderBox(viewer, box, clusterId, Color(1,0,0));
        }

        clusterId++;
    }

}


//setAngle: SWITCH CAMERA ANGLE {XY, TopDown, Side, FPS}
void initCamera(CameraAngle setAngle, pcl::visualization::PCLVisualizer::Ptr& viewer)
{

    viewer->setBackgroundColor (0, 0, 0);
    // set camera position and angle
    viewer->initCameraParameters();
    // distance away in meters
    int distance = 16;
    switch(setAngle)
    {
        case XY : viewer->setCameraPosition(-distance, -distance, distance, 1, 1, 0); break;
        case TopDown : viewer->setCameraPosition(0, 0, distance, 1, 0, 1); break;
        case Side : viewer->setCameraPosition(0, -distance, 0, 0, 0, 1); break;
        case FPS : viewer->setCameraPosition(-10, 0, 0, 0, 0, 1);
    }
    if(setAngle!=FPS)
        viewer->addCoordinateSystem (1.0);
}

std::string get_filename(std::string path)
{
    std::string s = path.substr(path.find_last_of("/\\") + 1);
    std::string delimiter = ".";
    std::string filename = s.substr(0, s.find(delimiter)); 
    return filename;
}

cv::Mat getDepthMap(cv::Mat& disp_left)
{
    cameraCalibrations calib;
    // disp_left.setTo(0.1, disp_left == 0);
    // disp_left.setTo(0.1, disp_left == -1);
    
    float focal = calib.k_left[0][0];
    float b = (calib.t_left[0] - calib.t_right[0]);

    cv::Size s = disp_left.size();


    cv::Mat depth_map = cv::Mat::ones(s.width, s.height, CV_32F);

    depth_map = focal*b/disp_left;
    // std::cout << "depth_map = " << std::endl << " "  << depth_map << std::endl << std::endl;
    std::cout<<(b)<<std::endl;
    return depth_map;
    
}

cv::Mat getDisparityMap(cv::Mat& img_left, cv::Mat& img_right)
{
    // Creating an object of StereoSGBM algorithm
    // cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();
    // stereo->setNumDisparities(6*16);
    // stereo->setBlockSize(11);
    
    cv::Mat img_left_gray, img_right_gray, disp, disparity;
    cv::cvtColor(img_left, img_left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_right, img_right_gray, cv::COLOR_BGR2GRAY);

    // stereo->compute(img_left_gray,img_right_gray,disp);
    // double Min,Max;
    // cv::minMaxLoc(img_left,&Min,&Max);
    // disp -= Min;
    // disp.convertTo(disparity,CV_8U,(Max-Min));
    // // disparity = (disparity/16.0f - (float)0)/((float)6*16);

    // stereo->compute(img_left_gray,img_right_gray,disp);
    // disp.convertTo(disparity,CV_32F, 1.0);
    // disparity = (disparity/16.0f);// - (float)0)/((float)6*16);

    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();

    int min_disparity = 0;
    int num_disparities = 6*16;
    int block_size = 11;
    int window_size = 6;

    stereo->setMinDisparity(min_disparity);
    stereo->setNumDisparities(num_disparities);
    stereo->setBlockSize(block_size);
    stereo->setP1(8 * 3 * window_size^2);
    stereo->setP2(32 * 3 * window_size^2);
    stereo->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
    double Min,Max;
    cv::minMaxLoc(img_left,&Min,&Max);

    stereo->compute(img_left_gray,img_right_gray,disp);
    disp.convertTo(disparity,CV_32F);
    disparity = (disparity/16.0f);// - (float)0)/((float)6*16);

    return disparity;

}


void view_images(std::string bin_path)
{
    std::string filename = get_filename(bin_path);
    std::string root_left = "/media/rahul/a079ceb2-fd12-43c5-b844-a832f31d5a39/kitti-360/download_2d_perspective/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/";
    std::string root_right = "/media/rahul/a079ceb2-fd12-43c5-b844-a832f31d5a39/kitti-360/download_2d_perspective/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_01/data_rect/";
    std::string path_left = root_left+filename+".png";
    std::string path_right = root_right+filename+".png";
    cv::Mat img_left = cv::imread(path_left, cv::IMREAD_COLOR);
    cv::Mat img_right = cv::imread(path_right, cv::IMREAD_COLOR);
    
    if(img_left.empty())
    {
        std::cout << "Could not read the image: " << path_left << std::endl;
    }

    if(img_right.empty())
    {
        std::cout << "Could not read the image: " << img_right << std::endl;
    }
    cv::Mat disparity_left = getDisparityMap(img_left, img_right);
    cv::imshow( "img_left", img_left );
    cv::imshow( "img_right", img_right );
    // cv::imshow( "disparity_left", disparity_left );
    cv::imwrite( "disparity_left.jpg", disparity_left );

    cv::Mat depth_map = getDepthMap(disparity_left);
    cv::imwrite( "depth_map.jpg", depth_map );
    
    int k = cv::waitKey(); // Wait for a keystroke in the window
    // cv::imwrite("temp.jpg", img);
}

void print(cameraCalibrations cam_calib)
{
    
    for (int i=0; i<cam_calib.k_left.size(); i++)
    {
        for (int j=0; j < cam_calib.k_left[i].size(); j++)
        {
            std::cout<<cam_calib.k_left[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
}




int main (int argc, char** argv)
{
    std::cout << "starting enviroment" << std::endl;
    // viewer is a pointer in heap memory
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
    // initializing the environment parameters
    CameraAngle setAngle = FPS;
    initCamera(setAngle, viewer);

    // load camera calibrations
    cameraCalibrations cam_calibs;
    // print(cam_calibs);

    // renders the scenes of the viewer
    ProcessPointClouds<pcl::PointXYZI>* pointProcessorI = new ProcessPointClouds<pcl::PointXYZI>();
    // std::vector<boost::filesystem::path> stream = pointProcessorI->streamPcd("../src/sensors/data/pcd/data_4");
    std::vector<boost::filesystem::path> stream = pointProcessorI->streamPcd("/media/rahul/a079ceb2-fd12-43c5-b844-a832f31d5a39/kitti-360/download_2d_perspective/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data");
    auto streamIterator = stream.begin();
    pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloudI;
    // cityBlock(viewer);

    while (!viewer->wasStopped ())
    {

        // Clear viewer
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();

        // Load pcd and run obstacle detection process
        // inputCloudI = pointProcessorI->loadPcd((*streamIterator).string());
        inputCloudI = pointProcessorI->loadBIN((*streamIterator).string());
        view_images((*streamIterator).string());
        cityBlock(viewer, pointProcessorI, inputCloudI);
        std::cout<<"The file being processed is: "<<get_filename((*streamIterator).string())<<std::endl;
        // std::string path_pcd = "../src/PCD/" + get_filename((*streamIterator).string()) + ".pcd";
        // pointProcessorI->savePcd(inputCloudI, path_pcd);

        streamIterator++;
        if(streamIterator == stream.end())
            streamIterator = stream.begin();

        viewer->spinOnce ();
    }
    
}