//
// Created by Backer on 2019/1/15.
//

/*
@author:chenzhengqiang
@date:2018-12-10
@email:642346572@qq.com
@desc:buffer native header
*/

#ifndef ANDROIDFABRICDEFECTDETECT_BUFFER_H
#define ANDROIDFABRICDEFECTDETECT_BUFFER_H
#include "opencl_common.h"
#include "config.h"

namespace buffer
{

    struct CLBuffer
    {
        cl_device_id        device;
        cl_context context;
        cl_command_queue queue;
        unsigned char  bimg_buffer[config::IMAGE_PIXELS];
        unsigned char  gimg_buffer[config::IMAGE_PIXELS];
        unsigned char  rimg_buffer[config::IMAGE_PIXELS];
        unsigned int reduce_l[config::REDUCE_BLOCKS];
        unsigned int reduce_a[config::REDUCE_BLOCKS];
        unsigned int reduce_b[config::REDUCE_BLOCKS];
        cl_image_format climg_format;
        cl_image_desc climg_desc;
        cl_mem cl_bimg;
        cl_mem cl_gimg;
        cl_mem cl_rimg;
        cl_mem cl_Limg;
        cl_mem cl_Aimg;
        cl_mem cl_Bimg;

        cl_mem loutput_buffer;
        cl_mem aoutput_buffer;
        cl_mem boutput_buffer;

        cl_sampler cl_sampler_;
        cl_program cl_bgr2lab_program;
        cl_kernel cl_bgr2lab_kernel;
        cl_program cl_reduction_program;
        cl_kernel cl_reduction_kernel;
        size_t cl_bgr2lab_origin[3];
        size_t cl_bgr2lab_region[3];
        size_t cl_bgr2lab_global_work_size[2];
        size_t cl_reduction_global_work_size[1];
        size_t cl_reduction_local_work_size[1];
        size_t cl_groups;
    };
}
#endif //ANDROIDFABRICDEFECTDETECT_BUFFER_H
