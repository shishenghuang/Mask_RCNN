#pragma once

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


/**
 * Helper class for reading a semantic specifier file.
 */
namespace DynaSeg {
    class SemfileReader {
    public:
        short mSemCount;  // This includes background
        std::string mSemModelPath;
        std::vector<std::string> mClassNames;
        std::vector<cv::Vec3b> mColorTable;
        std::vector<int> mClassType;
    public:
        SemfileReader(const std::string& path);
        ~SemfileReader(){};
    };
}
