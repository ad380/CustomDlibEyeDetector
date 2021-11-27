#include <iostream>
#include <chrono>
#include <optional>
#include <stdlib.h>
 
#include <asio/ts/internet.hpp>
 
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/dnn/dnn.hpp>
 
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>
 
using namespace cv;
using namespace dnn;
using namespace dlib;
using namespace std;
 
class Detector {       // The class
public:             // Access specifier
int count;
int detectCount; 
int frameCount;
std::vector<cv::Point2f> prevShape;// Attribute (int variable)
Net net;
shape_predictor predictor;
VideoWriter result;
Size s = Size(360, 480);
float confidenceThreshold;
 
Detector() {
count = 0;
detectCount = 0;
frameCount = 0;
 
//initialize caffeeDNN face detector
String prototxtPath = "face_detector/deploy.prototxt";
String weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel";
net = readNet(weightsPath, prototxtPath);
 
//initialize dlib shape predictor
deserialize("eye_predictor2.dat") >> predictor;
 
//initialize video writer
result = VideoWriter("cpptest.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 20, s);
 
confidenceThreshold = 0.4;
}
 
std::vector<cv::Point2f> detectEyesLeft(Mat frame);
std::vector<cv::Point2f> detectEyesRight(Mat frame);
 
void dlib_point2cv_Point(full_object_detection& S, std::vector<Point2f>& L, double& scale)
{
for (unsigned int i = 0; i < S.num_parts();++i)
{
L.push_back(Point2f(S.part(i).x() * (1 / scale), S.part(i).y() * (1 / scale)));
}
}
};
 
std::vector<cv::Point2f> Detector::detectEyesLeft(Mat frame) {
frameCount++;
 
//convert to grey scale
double scale = 1;
Mat gray;
cvtColor(frame,gray, COLOR_BGR2GRAY);
//resize(gray, resized, Size(), scale, scale);
//resize(frame, resized, Size(), scale, scale);
 
// construct a blob from the image
Mat blob = blobFromImage(frame, 1.0, Size(300, 300), (104.0, 177.0, 123.0));

//detect face
net.setInput(blob);
Mat detection = net.forward();
Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
 
float confidence = detectionMat.at<float>(0, 2);
 
if (confidence > confidenceThreshold) {
int idx = static_cast<int>(detectionMat.at<float>(0, 1));
int xLeftBottom = static_cast<int>(detectionMat.at<float>(0, 3) * frame.cols);
int yLeftBottom = static_cast<int>(detectionMat.at<float>(0, 4) * frame.rows);
int xRightTop = static_cast<int>(detectionMat.at<float>(0, 5) * frame.cols);
int yRightTop = static_cast<int>(detectionMat.at<float>(0, 6) * frame.rows);
 
dlib::rectangle rect = dlib::rectangle((int)(xLeftBottom - 15), (int)yLeftBottom, (int)(xRightTop - xLeftBottom - 5), (int)(yRightTop - yLeftBottom - 10));
 
full_object_detection eyes = predictor(gray, rect);
 
if (count == 0) {
dlib_point2cv_Point(eyes, prevShape, scale);
count++;
}
else {
std::vector<cv::Point2f> shape;
dlib_point2cv_Point(eyes, shape, scale);

std::vector<float> x_error_sum = {};
std::vector<float>y_error_sum = {};
for (int i = 0; i < shape.size();i++) {
x_error_sum.push_back(abs(shape[i].x- prevShape[i].y));
y_error_sum.push_back(abs(shape[i].y - prevShape[i].y));
}
 
float x_error = *max_element(x_error_sum.begin(), x_error_sum.end());
float y_error = *max_element(y_error_sum.begin(), y_error_sum.end());
 
if (frameCount <= 3 || detectCount > 4) {
prevShape = shape;
detectCount = 0;
}

if (x_error < 10 && y_error < 10 && frameCount > 3) {
prevShape = shape;
return shape;
}
else {
detectCount++;
}		
}
}
 
return std::vector<cv::Point2f>();
}
int main() {
for (int i = 0; i < 650; i++) {
String path = "RPM Mask Test/Image_Feed0_Frame" + std::to_string(i) + ".png";
Mat frame = imread(path);
Detector myDetect;
std::vector<cv::Point2f> landmarks = myDetect.detectEyesLeft(frame);
 
for (Point2f p: landmarks){
circle(frame, p, 1, (0, 0, 255), -1);
}
imshow("Frame", frame);
 
int key = cv::waitKey(30) & 255; // key is an integer here
if (key == 27) break;
}
destroyAllWindows();
return 0;
//rectangle(img, object, Scalar(0, 255, 0), 2);
}
 
std::vector<cv::Point2f> Detector::detectEyesRight(Mat frame)
{
return std::vector<cv::Point2f>();
}

