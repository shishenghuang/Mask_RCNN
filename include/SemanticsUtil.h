

#ifndef _SEMANTICS_UTIL_H
#define _SEMANTICS_UTIL_H

#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#if 0
#include "KeyFrame.h"
#endif

//#include <torch/script.h>
//#include <torch/torch.h>
#include <Python.h>

//#include "MaskNet.h"

using namespace std;
using namespace cv;

namespace DynaSeg{


    class SemanticsUtil{

            bool mbFileBased;
            std::vector<double> mFileTimestamps;
            std::string mFileBaseSemFolder;
            std::string mFileBaseInstFolder;

            //std::shared_ptr<torch::jit::script::Module> mNeuralModel;

            PyObject* mpPythonModule;
            PyObject* mpPythonExecute;
            void* initializeMaskRCNN();
            inline PyObject* getPyObject(const char* name);
            cv::Mat extractImage();
            void extractClassIDs(std::vector<int>* result);
            void extractBoundingBoxes(std::vector<cv::Rect>* result);

            //DeepVoxel::SegmentObject* MaskNet;
            // Visualize Color.
            std::vector<cv::Vec3b> mClsColor;
//            void showVisualization();

        public:
            // Use MaskRCNN
            //SemanticsUtil(const std::string& nnModelPath);
            SemanticsUtil();
            // Use GroundTruth Label
            SemanticsUtil(const std::string& assocPath, const std::string& semPath);
            ~SemanticsUtil();

        public:

            // detect the 2D semantic label of a given frame:
            // input: *cv::Mat* inputFrame, a normal RGB image in cv::Mat format
            // output: cv::Mat 16U contains label.
            void detectHardLabel(const cv::Mat& inputFrame, cv::Mat& labels, std::vector<cv::Rect>& bbox, bool visualize = false);
            #if 0
            std::shared_ptr<torch::Tensor> detectProbabilityMap(const cv::Mat& inputFrame, bool visualize = false);
            #endif 

#if 0
            void run();
            void insertKeyFrame(KeyFrame* kf);
            //void searchObject3DMatchByProjection(KeyFrame* kf);
            void setReconstruction(Reconstruction* pRecons) { mpReconstruction = pRecons; }
            //void SetLocalMapper(LocalMapping* pLocalMapper) {mpLocalMapper = pLocalMapper; }        
#endif
        protected:
            // We use one kf instead of a queue because semantic has to be kept up-to-date for instant tracking.
            // aka. we drop all old but not detected kfs.
#if 0
            std::vector<KeyFrame*> mpPendingKFQueue;
            std::mutex mMutexPendingKF;

            KeyFrame* mpCurrentKF;
            Reconstruction* mpReconstruction;

            //for bundle adjustment
            //LocalMapping* mpLocalMapper;

            // Download map, perform CRF and upload it...
            void performCRF();
            void performSVSeg();
            cv::Mat refineLabel(const cv::Mat& inputLabel, const cv::Mat& inputDepth, const KeyFrame* kf, bool visualize);
#endif
        public:
            cv::Mat visualizeSemantic(const cv::Mat& hardLabel);

    };
}

#endif