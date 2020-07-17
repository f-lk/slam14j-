#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

/***************************************************
 * 本节演示了如何根据data/目录下的十张图训练字典
 * ************************************************/

int main( int argc, char** argv )
{
    // read the image 
    cout<<"reading images... "<<endl;
    vector<Mat> images; 
    for ( int i=0; i<10; i++ )
    {
        string path = "./data/"+to_string(i+1)+".png";
        images.push_back( imread(path) );
    }
    cout<<"number of image "<<images.size()<<endl;
    // detect ORB features
    cout<<"detecting ORB features ... "<<endl;
    Ptr< Feature2D > detector = ORB::create();  //默认构造函数，每张图500个特征点
    vector<Mat> descriptors;
    for ( Mat& image:images )
    {
        vector<KeyPoint> keypoints; 
        Mat descriptor;
        detector->detectAndCompute( image, Mat(), keypoints, descriptor );
        descriptors.push_back( descriptor );
    }
//    cout <<"descriptors"<<descriptors[0]<<endl;
//    cout<<descriptors[0].cols<<endl<<descriptors[0].rows<<endl;

    // create vocabulary 
    cout<<"creating vocabulary ... "<<endl;
    DBoW3::Vocabulary vocab;
    vocab.create( descriptors );
    cout<<"vocabulary info: "<<vocab<<endl;
    vocab.save( "vocabulary.yml.gz" );
    cout<<"done"<<endl;
    
    return 0;
}