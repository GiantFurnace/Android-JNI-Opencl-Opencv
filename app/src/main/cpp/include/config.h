
#ifndef ANDROIDFABRICDEFECTDETECT_CONFIG_H
#define ANDROIDFABRICDEFECTDETECT_CONFIG_H

namespace config
{
     static const int IMAGE_ROWS = 640;
     static const int IMAGE_COLS = 640;
     static const int IMAGE_PIXELS = IMAGE_ROWS * IMAGE_COLS;
     static const int REDUCE_BLOCKS = 40;
     static const int CL_REDUCTION_GLOBAL_WORK_SIZE = 640;
     static const int CL_REDUCTION_LOCAL_WORK_SIZE = 64;
     static const int CL_GROUPS = 640 / 64;
}

#endif