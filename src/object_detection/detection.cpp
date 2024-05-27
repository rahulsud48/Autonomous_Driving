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
    putText(img, text, Point(x,y-5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1, LINE_AA);
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
        // try {
        // // Block of code to try
        //     depth = depth_map.at<float>(x_mid, y_mid);
        //     throw 505;
        //     // throw exception; // Throw an exception when a problem arise
        // }
        // catch (...){
        // // Block of code to handle errors
        //     depth = -1;
        // }
        // Check if the detection is of good quality
        if (score > threshold){
            int x_mid = static_cast<int>(x + (w/2));
            int y_mid = static_cast<int>(y + (h/2));
            // float depth;
            cv::Rect myROI(x_mid, y_mid, 10, 10);
            cv::Mat depth_object = depth_map(myROI);
            double Min,Max;
            cv::minMaxLoc(depth_object,&Min,&Max);
            // display_text(img, classes[classId].c_str(), x, y);
            std::cout<<"the depth of: "<< classes[classId].c_str() << " is: "<< Min<<std::endl;

            std::ostringstream ss;
            ss << Min;
            std::string s(ss.str());


            // display_text(img, classes[classId].c_str(), x, y);
            display_text(img, s, x_mid, y_mid);
            rectangle(img, Point(x,y), Point(x+w, y+h), Scalar(255,255,255), 2);
        }
    }
}
//////////////////////////////////////////////

void DataEncoder::createParameters()
{
    // creating areas for anchor boxes
    for (int i=3; i<8; i++)
    {
        anchor_areas.push_back(std::pow(std::pow(2,i),2));
    }
    // creating scales
    for (int i=0; i<3; i++)
    {
        float power = (float)((float)i/3);
        scales.push_back(std::pow(2.,(float)((float)i/3)));
    }
    // creating feature maps sizes
    for (int i=0; i<5; i++)
    {
        fm_sizes.push_back(std::ceil(300/std::pow(2.0, i+3)));
    }
    int i = 0;
    std::vector<torch::Tensor> anchor_boxes;
    for (const auto& fm_size : fm_sizes)
    {
        torch::Tensor anchors = generate_anchors(anchor_areas[i], aspect_ratios, scales);
        torch::Tensor anchor_grid = generate_anchor_grid(img_w, img_h, fm_size, anchors);
        anchor_boxes.push_back(anchor_grid);
        i++;
    }
    anchor_boxes_tensor = torch::cat(anchor_boxes, 0);
}

torch::Tensor DataEncoder::generate_anchor_grid(int img_w, int img_h, int fm_size, torch::Tensor anchors)
{
    float grid_size = (float)img_w/(float)fm_size;
    std::vector<torch::Tensor> meshgrid = torch::meshgrid({torch::arange(0, fm_size) * grid_size, torch::arange(0, fm_size) * grid_size});
    
    anchors = anchors.view({-1, 1, 1, 4});
    torch::Tensor xyxy = torch::stack({meshgrid[0], meshgrid[1], meshgrid[0], meshgrid[1]}, 2).to(torch::kFloat);
    auto boxes = (xyxy + anchors).permute({2, 1, 0, 3}).contiguous().view({-1, 4});
    // Clamp the coordinates to the input size
    boxes.index({torch::indexing::Slice(), torch::indexing::Slice(0)}).clamp_(0, img_w);
    boxes.index({torch::indexing::Slice(), torch::indexing::Slice(2)}).clamp_(0, img_w);
    boxes.index({torch::indexing::Slice(), torch::indexing::Slice(1)}).clamp_(0, img_h);
    boxes.index({torch::indexing::Slice(), torch::indexing::Slice(3)}).clamp_(0, img_h);
    return boxes;

}

torch::Tensor DataEncoder::generate_anchors(float anchor_area, std::vector<float> aspect_ratios, std::vector<float> scales)
{
    std::vector<std::vector<float>> anchors;
    for (const auto& scale : scales )
    {
        for (const auto& ratio : aspect_ratios)
        {
            float h = std::round(std::pow(anchor_area,0.5)/ratio);
            float w = std::round(ratio*h);
            float x1 = (std::pow(anchor_area,0.5) - scale * w) * 0.5f;
            float x2 = (std::pow(anchor_area,0.5) + scale * w) * 0.5f;
            float y1 = (std::pow(anchor_area,0.5) - scale * h) * 0.5f;
            float y2 = (std::pow(anchor_area,0.5) + scale * h) * 0.5f;
            anchors.push_back({x1,y1,x2,y2});
        }
    }
    torch::Tensor anchors_tensor = convert_2dvec_to_torch_tensor(anchors);
    return anchors_tensor;
}

torch::Tensor DataEncoder::convert_2dvec_to_torch_tensor(std::vector<std::vector<float>> array_2d)
{
    // use template to handle other data structures
    std::vector<float> flat_array;
    for (const auto& inner_vec : array_2d) 
    {
        flat_array.insert(flat_array.end(), inner_vec.begin(), inner_vec.end());
    }
    auto tensor = torch::tensor(flat_array, torch::kFloat);
    // Reshape the tensor to match the original 2D structure
    int rows = array_2d.size();  // Number of outer vectors
    int cols = array_2d[0].size();  // Assumes all inner vectors have the same size
    torch::Tensor reshaped_tensor = tensor.view({rows, cols});
    return reshaped_tensor;
}

DataEncoder::DataEncoder()
{
    createParameters();
}

torch::Tensor DataEncoder::decode_boxes(const torch::Tensor& deltas, const torch::Tensor& anchors) {
    // Calculate the width and height of the anchors
    auto anchors_wh = anchors.slice(1, 2, 4) - anchors.slice(1, 0, 2) + 1;

    // Calculate the centers of the anchors
    auto anchors_ctr = anchors.slice(1, 0, 2) + 0.5 * anchors_wh;

    // Calculate the centers of the predicted boxes
    auto pred_ctr = deltas.slice(1, 0, 2) * anchors_wh + anchors_ctr;

    // Calculate the width and height of the predicted boxes
    auto pred_wh = torch::exp(deltas.slice(1, 2, 4)) * anchors_wh;

    // Calculate the top-left and bottom-right coordinates
    auto top_left = pred_ctr - 0.5 * pred_wh;
    auto bottom_right = pred_ctr + 0.5 * pred_wh - 1;

    // Concatenate the top-left and bottom-right coordinates
    auto result = torch::cat({top_left, bottom_right}, 1);

    return result;
}

torch::Tensor DataEncoder::compute_nms(const torch::Tensor& boxes, const torch::Tensor& conf, float threshold) {
    // Extract box coordinates
    auto x1 = boxes.index({torch::indexing::Slice(), 0});
    auto y1 = boxes.index({torch::indexing::Slice(), 1});
    auto x2 = boxes.index({torch::indexing::Slice(), 2});
    auto y2 = boxes.index({torch::indexing::Slice(), 3});

    // Calculate areas of the boxes
    auto areas = (x2 - x1 + 1) * (y2 - y1 + 1);

    // Sort confidence in descending order and get the sorted indices
    auto sorted_result = conf.sort(0, /*descending=*/true);
    auto order = std::get<1>(sorted_result);

    // List to keep indices of boxes that pass NMS
    std::vector<int64_t> keep;

    while (order.numel() > 0) {
        int64_t i = order[0].item<int64_t>();
        keep.push_back(i);

        if (order.numel() == 1) {
            break;
        }

        // Get the indices except the first one (rest)
        auto rest = order.index({torch::indexing::Slice(1)});
        
        // Calculate overlap with the rest of the boxes
        auto xx1 = torch::maximum(x1.index({rest}), x1[i]);
        auto yy1 = torch::maximum(y1.index({rest}), y1[i]);
        auto xx2 = torch::minimum(x2.index({rest}), x2[i]);
        auto yy2 = torch::minimum(y2.index({rest}), y2[i]);
        // auto yy1 = y1.index({rest}).clamp_min(y1[i]);
        // auto xx2 = x2.index({rest}).clamp_max(x2[i]);
        // auto yy2 = y2.index({rest}).clamp_max(y2[i]);

        auto w = (xx2 - xx1 + 1).clamp_min(0);
        auto h = (yy2 - yy1 + 1).clamp_min(0);

        auto inter = w * h;
        auto ovr = inter / (areas[i] + areas.index({rest}) - inter);

        // Get indices where overlap is less than or equal to the threshold
        auto valid_indices = (ovr <= threshold).nonzero().squeeze();

        if (valid_indices.numel() == 0) {
            break;
        }

        // Convert valid indices to a vector of indices
        auto valid_indices_vec = valid_indices.to(torch::kInt64).contiguous().data_ptr<int64_t>();

        // Convert valid_indices to a std::vector
        std::vector<int64_t> valid_indices_std(valid_indices_vec, valid_indices_vec + valid_indices.numel());

        // Use the converted indices to index the rest
        order = rest.index_select(0, torch::tensor(valid_indices_std, torch::dtype(torch::kLong)));
    }

    return torch::tensor(keep, torch::dtype(torch::kLong));
}

void DataEncoder::decode(
    torch::Tensor loc_pred, 
    torch::Tensor cls_pred, 
    std::vector<std::map<int, std::vector<std::vector<float>>>>& output_boxes, 
    std::vector<std::map<int, std::vector<int>>>& output_classes, 
    int batch_size
)
{
    for (int i=0; i<batch_size; i++)
    {
        std::map<int, std::vector<std::vector<float>>> out_boxes;
        std::map<int, std::vector<int>> out_cls;
        torch::Tensor boxes = decode_boxes(loc_pred[i], anchor_boxes_tensor);
        
        torch::Tensor conf = cls_pred[i].softmax(1);
        for (int j=1; j<num_classes;j++)
        {
            torch::Tensor class_conf = conf.index({torch::indexing::Slice(), j});
            // Find indices where class_conf exceeds cls_threshold
            auto ids_tensor = (class_conf > cls_threshold).nonzero();
            

            // Squeeze the tensor to remove extra dimensions
            auto ids_squeezed = ids_tensor.squeeze();

            // Convert to a list of indices
            std::vector<int64_t> ids;
            ids_squeezed = ids_squeezed.view(-1);  // Ensure tensor is 1D

            ids.assign(ids_squeezed.data_ptr<int64_t>(), ids_squeezed.data_ptr<int64_t>() + ids_squeezed.numel());
            torch::Tensor ids_map = torch::tensor(ids);


            torch::Tensor keep = compute_nms(boxes.index_select(0,ids_map), class_conf.index_select(0,ids_map), 0.5);


            torch::Tensor boxes_out = boxes.index_select(0,ids_map).index_select(0,keep);
            torch::Tensor conf_out = class_conf.index_select(0,ids_map).index_select(0,keep);

            // out_boxes[j] = boxes_out;
            // out_cls[j] = conf_out;

            torch::IntArrayRef dimensions = boxes_out.sizes();
            std::vector<int64_t> tensor_shape(dimensions.begin(), dimensions.end());

            std::vector<std::vector<float>> float_boxes;
            std::vector<int> float_cls;
            for (int64_t k = 0; k < tensor_shape[0]; k++)
            {
                std::vector<float> float_box;

                // // Extract the coordinates from the tensor
                // for (int64_t l=0; l < tensor_shape[1]; l++)
                // {
                //     float_box.push_back(boxes_out[k][l].item<float>());
                // }
                float_box.push_back(boxes_out[k][0].item<float>() * 6.4);
                float_box.push_back(boxes_out[k][1].item<float>() * 4);
                float_box.push_back(boxes_out[k][2].item<float>() * 6.4);
                float_box.push_back(boxes_out[k][3].item<float>() * 4);

                float_cls.push_back(conf_out[k].item<int>());
                float_boxes.push_back(float_box);
            }

            out_boxes[j] = float_boxes;
            out_cls[j] = float_cls;

        }
        output_boxes.push_back(out_boxes);
        output_classes.push_back(out_cls);
    }
    
}

///////////////////////////////////////////////
void ssd_detector_torch::load_model()
{
    ssd_detector = torch::jit::load(this->model_path);
}

void ssd_detector_torch::transform_image(cv::Mat* img, torch::Tensor* img_tensor)
{
    // Resize to a new fixed size (e.g., 100x100 pixels)
    cv::Size new_size(300, 300);
    cv::resize(*img, *img, new_size);

    // Convert cv::Mat to a PyTorch tensor
    *img_tensor = torch::from_blob(
        img->data, {img->rows, img->cols, img->channels()}, torch::kByte);


    // Convert to float and scale pixel values from [0, 255] to [0.0, 1.0]
    *img_tensor = img_tensor->to(torch::kFloat).div(255.0);

    // Convert from BGR to RGB by reversing the last dimension
    *img_tensor = img_tensor->permute({2, 0, 1});  // [channels, height, width]

    // Add a batch dimension (batch size = 1)
    *img_tensor = img_tensor->unsqueeze(0);  // [1, channels, height, width]
}

void ssd_detector_torch::detect(torch::Tensor& img_tensor, torch::Tensor& boxes, torch::Tensor& classes)
{
    std::vector<torch::jit::IValue> jit_input;
    jit_input.push_back(img_tensor);

    auto outputs = ssd_detector.forward(jit_input).toTuple();
    boxes = outputs->elements()[0].toTensor();
    classes = outputs->elements()[1].toTensor();

}

void ssd_detector_torch::display_text(cv::Mat& img, std::string text, int x, int y)
{
    // Get text size
    int baseLine;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, &baseLine);
    // Use text size to create a black rectangle
    rectangle(img, Point(x,y-textSize.height-baseLine), Point(x+textSize.width,y+baseLine),
             Scalar(0,0,0),-1);
    // Display text inside the rectangle
    putText(img, text, Point(x,y-5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1, LINE_AA);
}

void ssd_detector_torch::display_objects(
    cv::Mat& img, 
    std::vector<std::map<int, std::vector<std::vector<float>>>> output_boxes, 
    std::vector<std::map<int, std::vector<int>>> output_classes,
    int batch_size,
    cv::Mat& depth_map

)
{
    for (int i=0; i<batch_size; i++)
    {
        for (int j=1; j<12;j++)
        {
            std::vector<std::vector<float>> boxes = output_boxes[i][j];
            int instance = boxes.size();
            for (int64_t k = 0; k < instance; k++)
            {
                std::vector<float> box = boxes[k];
                int x1 = static_cast<int>(boxes[k][0]);
                int y1 = static_cast<int>(boxes[k][1]);
                int x2 = static_cast<int>(boxes[k][2]);
                int y2 = static_cast<int>(boxes[k][3]);

                int x_mid = (x1+x2)/2;
                int y_mid = (y1+y2)/2;

                cv::Rect myROI(x_mid, y_mid, 10, 10);
                cv::Mat depth_object = depth_map(myROI);
                double Min,Max;
                cv::minMaxLoc(depth_object,&Min,&Max);

                std::cout<<"the depth of the object is: "<< Min<<std::endl;

                std::ostringstream ss;
                ss << Min;
                std::string s(ss.str());


                // display_text(img, classes[classId].c_str(), x, y);
                display_text(img, s, x_mid, y_mid);
                rectangle(img, Point(x1,y2), Point(y1, y2), Scalar(255,255,255), 2);
            }
        }
    }
    
}