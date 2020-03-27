
#include <iostream>
#include <fstream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.hpp>
#include "SemanticsUtil.h"

#if 0
#include "Reconstruction.h"
#include "Thirdparty/SuperVoxel/codelibrary/geometry/io/xyz_io.h"
#include "Thirdparty/SuperVoxel/codelibrary/geometry/point_cloud/supervoxel_segmentation.h"
#include "Thirdparty/SuperVoxel/codelibrary/geometry/util/distance_3d.h"

#include "Thirdparty/SuperVoxel/vccs_supervoxel.h"
#include "Thirdparty/SuperVoxel/vccs_knn_supervoxel.h"
#endif 

#include "../python_env/numpy/ndarraytypes.h"
#include "../python_env/numpy/arrayobject.h"


static const short SCANNET_SUN38_CONVERT[] = {
        0, 1, 5, 2, 7, 8, 6, 3, 15, 14, 5, 4, 0, 18, 34, 11, 9, 33, 10, 0, 0, 16, 23, 5, 7, 0, 29, 24, 35, 3, 0, 27, 21, 25, 32, 12, 17, 0, 0, 18, 0, 22, 36, 0, 7, 7, 0, 37, 0, 0, 0, 0, 30, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 5, 3, 0, 0, 0, 26, 31, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 8, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 8, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 11, 0, 0, 4, 0, 7, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 8, 0, 0, 0, 0, 23, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 37, 0, 37, 0, 0, 0, 0, 0, 37, 0, 0, 0, 0, 1, 0, 29, 0, 0, 0, 0, 0, 0, 37, 29, 31, 0, 0, 33, 0, 0, 0, 0, 0, 0, 37, 0, 0, 20, 0, 0, 0, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 37, 0, 0, 5, 0, 0, 0, 0, 0, 37, 0, 29, 0, 21, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 15, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 5, 0, 0, 0, 21, 0, 0, 8, 0, 2, 0, 4, 0, 0, 0, 0, 0, 7, 37, 0
};

namespace DynaSeg
{
    
    SemanticsUtil::SemanticsUtil(){
        // mpCurrentKF = nullptr;
        // mpPendingKFQueue.clear();
        mbFileBased = false;

#if 0
        // Neural Net API.
        mNeuralModel = torch::jit::load(nnModelPath);
        //MaskNet = new DeepVoxel::SegmentObject();
        assert(mNeuralModel != nullptr);
#endif        
        initializeMaskRCNN();
        std::cout << "* Initialised MaskRCNN" << std::endl;

        for (int i = 0; i < 50; ++i) {
            mClsColor.push_back(cv::Vec3b(rand() % 255, rand() % 255, rand() % 255));
        }
    }

    void* SemanticsUtil::initializeMaskRCNN() {
        Py_SetProgramName((wchar_t*)L"MaskRCNN");
        Py_Initialize();
        wchar_t const * argv2[] = { L"MaskRCNN.py" };
        PySys_SetArgv(1, const_cast<wchar_t**>(argv2));
        // Load Module
        mpPythonModule = PyImport_ImportModule("MaskRCNN");
        if (mpPythonModule == nullptr) {
            if(PyErr_Occurred()) {
                std::cout << "Python error indicator is set:" << std::endl;
                PyErr_Print();
            }
            throw std::runtime_error("Could not open MaskRCNN module.");
        }
        import_array();
        // Get function
        mpPythonExecute = PyObject_GetAttrString(mpPythonModule, "execute");
        if(mpPythonExecute == nullptr || !PyCallable_Check(mpPythonExecute)) {
            if(PyErr_Occurred()) {
                std::cout << "Python error indicator is set:" << std::endl;
                PyErr_Print();
            }
            throw std::runtime_error("Could not load function 'execute' from MaskRCNN module.");
        }
        return 0;
    }

    SemanticsUtil::SemanticsUtil(const std::string& assocPath, const std::string& semPath) {
        // mpCurrentKF = nullptr;
        // mpPendingKFQueue.clear();
        mbFileBased = true;

        // Load association file.
        std::ifstream fAssociation(assocPath);
        while (fAssociation) {
            string s; getline(fAssociation, s);
            if (!s.empty()) {
                stringstream ss;
                ss << s;
                double t;
                ss >> t;
                mFileTimestamps.push_back(t);
            }
        }
        mFileBaseSemFolder = semPath;
        #if 0
        mFileBaseInstFolder = semPath + "/../instance/";

        // Check if the two folders exist.
        {
            struct stat info;
            if (stat(mFileBaseSemFolder.c_str(), &info) != 0 ||
                stat(mFileBaseInstFolder.c_str(), &info) != 0) {
                LOG(ERROR) << "Groundtruth semantic folder not exist!";
                exit(2);
            }
        }
        #endif
       // LOG(INFO) << "Finish .." << std::endl;
    }

    SemanticsUtil::~SemanticsUtil(){

    }

        // Creating arguments for MaskRCNN to use.
    static inline PyObject* createArguments(cv::Mat rgbImage) {
        assert(rgbImage.channels() == 3);
        npy_intp dims[3] = { rgbImage.rows, rgbImage.cols, 3 };
        return PyArray_SimpleNewFromData(3, dims, NPY_UINT8, rgbImage.data); // TODO Release?
    }
    
    PyObject *SemanticsUtil::getPyObject(const char* name){
        PyObject* obj = PyObject_GetAttrString(mpPythonModule, name);
        if(!obj || obj == Py_None) throw std::runtime_error(std::string("Failed to get python object: ") + name);
        return obj;
    }

    cv::Mat SemanticsUtil::extractImage(){
        PyObject* pImage = getPyObject("current_segmentation");
        PyArrayObject *pImageArray = (PyArrayObject*)(pImage);
        //assert(pImageArray->flags & NPY_ARRAY_C_CONTIGUOUS);

        unsigned char* pData = (unsigned char*)PyArray_GETPTR1(pImageArray,0);
        npy_intp h = PyArray_DIM(pImageArray,0);
        npy_intp w = PyArray_DIM(pImageArray,1);

        cv::Mat result;
        cv::Mat(h,w, CV_8UC1, pData).copyTo(result);
        Py_DECREF(pImage);
        return result;
    }

    void SemanticsUtil::extractClassIDs(std::vector<int>* result){
        assert(result->size() == 0);
        PyObject* pClassList = getPyObject("current_class_ids");
        if(!PySequence_Check(pClassList)) throw std::runtime_error("pClassList is not a sequence.");
        Py_ssize_t n = PySequence_Length(pClassList);
        result->reserve(n+1);
        result->push_back(0); // Background
        for (int i = 0; i < n; ++i) {
            PyObject* o = PySequence_GetItem(pClassList, i);
            assert(PyLong_Check(o));
            result->push_back(PyLong_AsLong(o));
            Py_DECREF(o);
        }
        Py_DECREF(pClassList);
    }

    void SemanticsUtil::extractBoundingBoxes(std::vector<cv::Rect> *result){
        assert(result->size() == 0);
        PyObject* pRoiList = getPyObject("current_bounding_boxes");
        if(!PySequence_Check(pRoiList)) throw std::runtime_error("pRoiList is not a sequence.");
        Py_ssize_t n = PySequence_Length(pRoiList);
        result->reserve(n);
        for (int i = 0; i < n; ++i) {
            PyObject* pRoi = PySequence_GetItem(pRoiList, i);
            assert(PySequence_Check(pRoi));
            Py_ssize_t ncoords = PySequence_Length(pRoi);
            assert(ncoords==4);

            PyObject* c0 = PySequence_GetItem(pRoi, 0);
            PyObject* c1 = PySequence_GetItem(pRoi, 1);
            PyObject* c2 = PySequence_GetItem(pRoi, 2);
            PyObject* c3 = PySequence_GetItem(pRoi, 3);
            assert(PyLong_Check(c0) && PyLong_Check(c1) && PyLong_Check(c2) && PyLong_Check(c3));

            int a = PyLong_AsLong(c0);
            int b = PyLong_AsLong(c1);
            int c = PyLong_AsLong(c2);
            int d = PyLong_AsLong(c3);
            Py_DECREF(c0);
            Py_DECREF(c1);
            Py_DECREF(c2);
            Py_DECREF(c3);

            result->push_back(cv::Rect(b,a,d-b,c-a));
            Py_DECREF(pRoi);
        }
        Py_DECREF(pRoiList);
    }

    void SemanticsUtil::detectHardLabel(const cv::Mat& inputFrame, cv::Mat& labels, std::vector<cv::Rect>& bbox, bool visualize) {
//        const int cropSize = 513;
        int inputW = inputFrame.cols;
        int inputH = inputFrame.rows;

        cv::Mat img;
//        cv::resize(inputFrame, img, cv::Size(cropSize, cropSize));
//        img = inputFrame.clone();
        img = inputFrame.clone();

        Py_XDECREF(PyObject_CallFunctionObjArgs(mpPythonExecute, createArguments(img), NULL));
        extractBoundingBoxes(&bbox);
        std::vector<int> instanceSemantic;
        extractClassIDs(&instanceSemantic);
        cv::Mat instance = extractImage();
        instance.convertTo(instance, CV_16U);
        labels = cv::Mat(img.size(), CV_16U);
        for (int i = 0; i < labels.rows; ++i) {
            for (int j = 0; j < labels.cols; ++j) {
                labels.at<short>(i, j) = instanceSemantic[instance.at<short>(i, j)];
            }
        }

#if 0
        torch::Tensor tensor_image = torch::from_blob(img.data,
                                                        {1, img.rows, img.cols, 3}, torch::kByte);
        tensor_image = tensor_image.permute({0, 3, 1, 2});
        tensor_image = tensor_image.toType(torch::kFloat);
        tensor_image.div_(255);
        float mean_data[] = {0.485, 0.456, 0.406};
        torch::Tensor mPreMean = torch::from_blob(mean_data, {1, 3, 1, 1}, torch::kFloat);
        float std_data[] = {0.229, 0.224, 0.225};
        torch::Tensor mPreStd = torch::from_blob(std_data, {1, 3, 1, 1}, torch::kFloat);
        tensor_image.sub_(mPreMean);
        tensor_image.div_(mPreStd);
        tensor_image = tensor_image.to(torch::kCUDA);
        torch::Tensor result = mNeuralModel->forward({tensor_image}).toTensor();
        torch::Tensor maxResult = result.argmax(1).toType(torch::kShort);
        Vector2i noDims(maxResult.size(2), maxResult.size(1));
        spair.predRes = std::make_shared<ITMShortImage>(noDims, true, true);
        spair.predRes->SetFromRaw(maxResult.data<short>(), noDims,
                                    ITMShortImage::CPU_TO_CUDA);
        spair.predRes->UpdateHostFromDevice();
        // add the keyframe to the localMapper
        short *l_data = maxResult.data<short>();
        cv::Mat labels = cv::Mat(noDims.y , noDims.x , CV_16SC1);
        for(int i = 0 ;i < noDims.y ; i ++){
            for(int j = 0 ;j < noDims.x ; j ++){
                int local_index = j + i * noDims.x;
                short label_i = l_data[local_index];
                labels.at<short>(i,j) = label_i;
            }
        }
#endif        
    }

#if 0
    std::shared_ptr<torch::Tensor> SemanticsUtil::detectProbabilityMap(const cv::Mat &inputFrame, bool visualize) {
        torch::Tensor tensor_image = torch::from_blob(inputFrame.data,
                                                      {1, inputFrame.rows, inputFrame.cols, 3}, torch::kByte);
        tensor_image = tensor_image.permute({0, 3, 1, 2});
        tensor_image = tensor_image.toType(torch::kFloat);
        tensor_image.div_(255);
        float mean_data[] = {0.485, 0.456, 0.406};
        torch::Tensor mPreMean = torch::from_blob(mean_data, {1, 3, 1, 1}, torch::kFloat);
        float std_data[] = {0.229, 0.224, 0.225};
        torch::Tensor mPreStd = torch::from_blob(std_data, {1, 3, 1, 1}, torch::kFloat);
        tensor_image.sub_(mPreMean);
        tensor_image.div_(mPreStd);
        tensor_image = tensor_image.to(torch::kCUDA);
        torch::Tensor result = mNeuralModel->forward({tensor_image}).toTensor();
        return nullptr;
//        return std::make_shared<torch::Tensor>(&result);
    }


    void SemanticsUtil::run() {
//        std::cout << ">>>>>>>>> SemDetector Thread Started" << std::endl;
//        int counter = 0;

        while (true) {
            {
                unique_lock<mutex> lock(mMutexPendingKF);
                if (!mpPendingKFQueue.empty()) {
                    mpCurrentKF = mpPendingKFQueue.back();
                    mpPendingKFQueue.pop_back();
                }
            }
            if (mpCurrentKF != nullptr) {
                // Detect this and trigger integration in Recon.

                cv::Mat img = mpCurrentKF->mat_rgb;   // RGB transformed by System::GrabImageRGBD.
//                cv::imwrite("/home/huangjh/syncCheck/" + std::to_string(counter++) + ".png", img);

                if (!mbFileBased) {
                                              
                    //////////////////////////////////////////////////////////////////////////////////////////
#if 0                    
                    torch::Tensor tensor_image = torch::from_blob(img.data,
                                                                  {1, img.rows, img.cols, 3}, torch::kByte);
                    tensor_image = tensor_image.permute({0, 3, 1, 2});
                    tensor_image = tensor_image.toType(torch::kFloat);
                    tensor_image.div_(255);
                    float mean_data[] = {0.485, 0.456, 0.406};
                    torch::Tensor mPreMean = torch::from_blob(mean_data, {1, 3, 1, 1}, torch::kFloat);
                    float std_data[] = {0.229, 0.224, 0.225};
                    torch::Tensor mPreStd = torch::from_blob(std_data, {1, 3, 1, 1}, torch::kFloat);
                    tensor_image.sub_(mPreMean);
                    tensor_image.div_(mPreStd);
                    tensor_image = tensor_image.to(torch::kCUDA);
                    torch::Tensor result = mNeuralModel->forward({tensor_image}).toTensor();
                    torch::Tensor maxResult = result.argmax(1).toType(torch::kShort);
                    Vector2i noDims(maxResult.size(2), maxResult.size(1));
                    spair.predRes = std::make_shared<ITMShortImage>(noDims, true, true);
                    spair.predRes->SetFromRaw(maxResult.data<short>(), noDims,
                                              ITMShortImage::CPU_TO_CUDA);
                    spair.predRes->UpdateHostFromDevice();
                    // add the keyframe to the localMapper
                    short *l_data = maxResult.data<short>();
                    cv::Mat labels(noDims.y , noDims.x , CV_16SC1);
                    for(int i = 0 ;i < noDims.y ; i ++){
                        for(int j = 0 ;j < noDims.x ; j ++){
                            int local_index = j + i * noDims.x;
                            short label_i = l_data[local_index];
                            labels.at<short>(i,j) = label_i;
                        }
                    }
#endif
                    Py_XDECREF(PyObject_CallFunctionObjArgs(mpPythonExecute, createArguments(img), NULL));
//                extractBoundingBoxes(&frameData->rois);
                    std::vector<int> instanceSemantic;
                    extractClassIDs(&instanceSemantic);
                    cv::Mat instance = extractImage();
                    instance.convertTo(instance, CV_16U);
                    cv::Mat labels(img.size(), CV_16U);
                    for (int i = 0; i < labels.rows; ++i) {
                        for (int j = 0; j < labels.cols; ++j) {
                            labels.at<short>(i, j) = instanceSemantic[instance.at<short>(i, j)];
                        }
                    }

                    mpCurrentKF->mat_label = labels.clone();
                    if(mpCurrentKF == NULL){
                        std::cout << "mpCurrentKF is NULL!!!" << std::endl;
                    }

                    SemPair spair;
                    spair.kf = mpCurrentKF;
                    spair.pose = mpCurrentKF->GetPose();
                    //spair.old_pose = mpCurrentKF->lastIntoScene;//mpCurrentKF->GetPrvPose();
                    spair.score = 0.0f;
                    mpReconstruction->addToSemCastQueue(spair);

                    //////////////////////////////////////////////////////////////////////////////////////////
                    //////////////////////////////////////////////////////////////////////////////////////////
                } else {
                    auto tit = std::find(mFileTimestamps.begin(), mFileTimestamps.end(), mpCurrentKF->mTimeStamp);
                    int img_id = std::distance(mFileTimestamps.begin(), tit);
                    cv::Mat label_img = cv::imread(mFileBaseSemFolder + "/" + std::to_string(img_id) + ".png",
                                                   CV_LOAD_IMAGE_UNCHANGED);
                    cv::Mat instance_img = cv::imread(mFileBaseInstFolder + "/" + std::to_string(img_id) + ".png",
                            CV_LOAD_IMAGE_UNCHANGED);
                    cv::resize(label_img, label_img, img.size(), 0, 0, INTER_NEAREST);
                    label_img.convertTo(label_img , CV_16S);
                    #if 0
                    cv::resize(instance_img, instance_img, img.size(), 0, 0, INTER_NEAREST);
                    assert(label_img.type() == CV_16U);
                    assert(instance_img.type() == CV_8UC1);
                    instance_img.convertTo(instance_img, CV_16U);
                    #endif
//                    cout << instance_
                    // Map Labels.
                    for (int i = 0; i < label_img.rows; ++i) {
                        for (int j = 0; j < label_img.cols; ++j) {
                            short& c = label_img.at<short>(i, j);
                            c = SCANNET_SUN38_CONVERT[c];
                        }
                    }
                    #if 0
                    // Reassign instance id & Find instance id to semantic id mapping.
                    // Instance id starts from 1, with 0 being void.
                    std::vector<short> instanceSemantics;
                    instanceSemantics.push_back(0);
                    {
                        std::map<short, short> instIdReassign;
                        instIdReassign[0] = 0;
                        for (int i = 0; i < instance_img.rows; ++i) {
                            for (int j = 0; j < instance_img.cols; ++j) {
                                short& c = instance_img.at<short>(i, j);
                                if (instIdReassign.count(c) == 0) {
                                    short semLabel = label_img.at<short>(i, j);
                                    // wall, floor and ceiling are excluded.
                                    if (semLabel <= 2 || semLabel == 22 || semLabel == 8 || semLabel == 11) {
                                        instIdReassign[c] = 0;
                                    } else {
                                        instIdReassign[c] = instanceSemantics.size();
                                        instanceSemantics.push_back(semLabel);
                                    }
                                }
                                c = instIdReassign[c];
                            }
                        }
                    }
                    instance_img = refineLabel(instance_img, mpCurrentKF->mat_depth, mpCurrentKF, false);
                    // After the refinement Map semantics back to label_img.
                    for (int i = 0; i < label_img.rows; ++i) {
                        for (int j = 0; j < label_img.cols; ++j) {
                            label_img.at<short>(i, j) = instanceSemantics[instance_img.at<short>(i, j)];
                        }
                    }
                    #endif
                    //////////////////////////////////////////////////////////////////////////////////////////
                    //////////////////////////////////////////////////////////////////////////////////////////
                    // add the keyframe to the localMapper
                    mpCurrentKF->mat_label = label_img.clone();//new_label_image.clone();
                    if(mpCurrentKF == NULL){
                        std::cout << "mpCurrentKF is NULL!!!" << std::endl;
                    } 

                    SemPair spair;
                    spair.kf = mpCurrentKF;
                    spair.pose = mpCurrentKF->GetPose();
                    //spair.old_pose = mpCurrentKF->lastIntoScene;//mpCurrentKF->GetPrvPose();
                    spair.score = 0.0f;

                    mpReconstruction->addToSemCastQueue(spair);
                                   
                }

                mpCurrentKF = nullptr;
                // performCRF();
                performSVSeg();
            }
            usleep(5000);
        }
    }

    void SemanticsUtil::insertKeyFrame(KeyFrame *kf) {
        unique_lock<mutex> lock(mMutexPendingKF);
        mpPendingKFQueue.push_back(kf);
    }

    void SemanticsUtil::performCRF() {
        // Download map.
        // TODO: Possible Synchronization Problem
        mpReconstruction->getAllVoxels();
    }

    void SemanticsUtil::performSVSeg() {
        // Download map.
        // TODO: Possible Synchronization Problem
        //mpReconstruction->performSVSeg();
    }

    cv::Mat SemanticsUtil::refineLabel(const cv::Mat& inputLabel, const cv::Mat& inputDepth, const KeyFrame* kf,
            bool visualize = false) {
        // Refine Label using VCSS SuperVoxel
        const float fx = kf->mK.at<float>(0,0);
        const float fy = kf->mK.at<float>(1,1);
        const float cx = kf->mK.at<float>(0,2);
        const float cy = kf->mK.at<float>(1,2);
        cl::Array<cl::RPoint3D> points;
        cl::Array<cl::IPoint2D> indexes;
        for (int y = 0; y < inputDepth.rows; ++y) {
            for (int x = 0; x < inputDepth.cols; ++x) {
                float tDepth = inputDepth.at<float>(y, x);
                if (tDepth <= 1e-6) continue;
                float posX = tDepth * (x - cx) / fx;
                float posY = tDepth * (y - cy) / fy;
                points.emplace_back(posX, posY, tDepth);
                //std::cout << posX << " " << posY << " " << tDepth << " ";
                indexes.emplace_back(x, y);
            }
        }
        cout << "Start VCCS supervoxel segmentation... with points.size() = " << points.size()  << endl;
        // Note that, you may need to change the resolution of voxel.
        const double voxel_resolution = 0.03;
        const double resolution = 0.5;  // The larger, the bigger super voxel is.
        cl::VCCSSupervoxel vccs(points.begin(), points.end(),
                                voxel_resolution,
                                resolution);
        cl::Array<int> vccs_labels;
        cl::Array<cl::VCCSSupervoxel::Supervoxel> vccs_supervoxels;
        vccs.Segment(&vccs_labels, &vccs_supervoxels);
        int n_supervoxels = vccs_supervoxels.size();
        cout << n_supervoxels << " supervoxels computed." << endl;
//        WritePoints("out_vccs.xyz", n_supervoxels, points, vccs_labels);

        // Semantic Refinement.
        struct SemanticCounter {
            std::map<int, int> counter;
            inline void add(int idx) {
                counter[idx] ++;
            }
            inline int getMax() const {
                using pair_type = decltype(counter)::value_type;
                return std::max_element(counter.begin(), counter.end(), [] (const pair_type & p1, const pair_type & p2) {
                    return p1.second < p2.second;
                })->first;
            }
        };
        std::vector<SemanticCounter> counters(n_supervoxels);
        for (int i = 0; i < (int) indexes.size(); ++i) {
            int tLabel = vccs_labels[i];
            if (tLabel == -1) continue;
            int sLabel = inputLabel.at<short>(indexes[i][1], indexes[i][0]);
            counters[tLabel].add(sLabel);
        }
        std::vector<int> voxelSemanticAssignment(n_supervoxels);
        for (int i = 0; i < n_supervoxels; ++i) {
            voxelSemanticAssignment[i] = counters[i].getMax();
        }
        // Backprojection of labels.
        cv::Mat newSem(inputLabel.size(), CV_16U);
        // Invalid areas of VCSS is set to label 0, meaning void
        // void area are assigned little weight during semCasting
        newSem.setTo(0);
        for (int i = 0; i < (int) indexes.size(); ++i) {
            if (vccs_labels[i] == -1) continue;
            newSem.at<short>(indexes[i][1], indexes[i][0]) = voxelSemanticAssignment[vccs_labels[i]];
        }

        if (visualize) {
            cl::Array<cl::RGB32Color> colors(points.size());
            std::mt19937 random;
            cl::Array<cl::RGB32Color> supervoxel_colors(n_supervoxels);
            for (int i = 0; i < n_supervoxels; ++i) {
                supervoxel_colors[i] = cl::RGB32Color(random());
            }
            cv::Mat backProjected(inputLabel.size(), CV_8UC3);
            backProjected.setTo(cv::Vec3b());
            for (int i = 0; i < (int) indexes.size(); ++i) {
                cl::RGB32Color tColor = supervoxel_colors[vccs_labels[i]];
                backProjected.at<cv::Vec3b>(indexes[i][1], indexes[i][0]) = cv::Vec3b(tColor.red(),
                                                                                      tColor.green(), tColor.blue());
            }
            cv::Mat oldSemanticVis = visualizeSemantic(inputLabel);
            cv::Mat newSemanticVis = visualizeSemantic(newSem);

            oldSemanticVis = oldSemanticVis * 0.5 + backProjected * 0.5;

            cv::resize(backProjected, backProjected, cv::Size(320, 240), 0, 0, INTER_NEAREST);
            cv::resize(oldSemanticVis, oldSemanticVis, cv::Size(320, 240), 0, 0, INTER_NEAREST);
            cv::resize(newSemanticVis, newSemanticVis, cv::Size(320, 240), 0, 0, INTER_NEAREST);
            cv::hconcat(backProjected, oldSemanticVis, backProjected);
            cv::hconcat(backProjected, newSemanticVis, backProjected);

            cv::imshow("Semantic Refinement visualization", backProjected);
            cv::waitKey(1);
        }

        return newSem;
    }
#endif
    cv::Mat SemanticsUtil::visualizeSemantic(const cv::Mat& hardLabel) {
        cv::Mat vis(hardLabel.size(), CV_8UC3);
        int width = hardLabel.cols;
        int height = hardLabel.rows;
        //cout << "Here with "<< hardLabel.rows << " " << hardLabel.cols << std::endl;
        for (int ii = 0; ii < height; ii++) {
            for (int jj = 0; jj < width; jj++) {
                //cout << "here: i = " << ii << ", j = " << jj << " ";
                short c = hardLabel.at<short>(ii, jj);
                //cout <<" c = "<< c << " "; 
                if(c > 0){
                    vis.at<cv::Vec3b>(ii, jj) =   mClsColor[(int)c % mClsColor.size()];  
                }
                //vis.at<cv::Vec3b>(i, j) = (c > 0) ? mClsColor[c % mClsColor.size()] : cv::Vec3b(0,0,0);
            }
        }
        return vis;
    }

} // namespace ORB_SLAM2

