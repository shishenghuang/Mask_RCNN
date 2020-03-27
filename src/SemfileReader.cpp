#include "SemfileReader.h"
#include <fstream>
#include <iostream>
#include <cassert>
#include <sstream>

namespace DynaSeg {

SemfileReader::SemfileReader(const std::string &path) {
    std::ifstream fin(path);
    if (!fin) {
        std::cerr << "Sem file " << path << " not exist!" << std::endl;
        assert(false);
    }

    std::string fline;
    // Line 1: model path
    std::getline(fin, fline); mSemModelPath = fline;
    // Line 2: semCount.
    std::getline(fin, fline); std::stringstream ss(fline); ss >> mSemCount;
    // Line 3~: Sem defines.
    for (int i = 0; i < mSemCount; ++i) {
        std::getline(fin, fline);
        ss.clear(); ss.str(fline);
        // Class name. Color (RGB). isObject.
        std::string semName; cv::Vec3b semColor; int semClassType;
        int r, g, b;
        ss >> semName >> r >> g >> b >> semClassType;
        semColor[0] = (uchar)r;
        semColor[1] = (uchar)g;
        semColor[2] = (uchar)b;
        mClassNames.push_back(semName);
        mClassType.push_back(semClassType);
        mColorTable.push_back(semColor);
    }

}

}
