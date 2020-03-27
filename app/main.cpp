

#include <iostream>
#include <vector>

#include "SemanticsUtil.h"

#include <opencv2/core.hpp>
#include <opencv/highgui.h>
#include <opencv2/highgui.hpp>


using namespace std;
using namespace cv;
using namespace DynaSeg;

int main()
{
    char* filepath = "/home/huan/catkin_ws/dyna_field_nav/data/";
    char* savepath = "/home/huan/catkin_ws/dyna_field_nav/labels/";
    char buf[255];
    int fNum = 162;

    DynaSeg::SemanticsUtil *sem = new DynaSeg::SemanticsUtil();
    for(int i = 0 ; i < fNum ; i += 5){
        sprintf(buf, "%s/rgb_%d.png" , filepath , i);
        cv::Mat img = cv::imread(buf , 1);

        cv::Mat labels;
        vector<cv::Rect> bbox;
        sem->detectHardLabel(img , labels , bbox);

        cv::Mat sImg = sem->visualizeSemantic(labels);
        sprintf(buf, "%s/labels_s_%d.png", savepath , i);
        cv::imwrite(buf , sImg);
        sprintf(buf, "%s/labels_%d.png" , savepath , i);
        cv::imwrite(buf, labels);
    }

    return 0;
}