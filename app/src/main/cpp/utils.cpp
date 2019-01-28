//
// Created by Backer on 2019/1/15.
//

#include "include/buffer.h"
#include "include/utils.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>


const char * BGR2LAB_KERNEL[] = {
"__kernel void bgr2lab(__read_only image2d_t Bimg_, __read_only image2d_t Gimg_, __read_only image2d_t Rimg_,\n"
                       "__write_only image2d_t Limg, __write_only image2d_t Aimg, __write_only image2d_t Bimg,\n"
                       "sampler_t sampler)\n"
"{\n"
    "const int x = get_global_id(0);\n"
    "const int y = get_global_id(1);\n"
    "float4 sB = read_imagef(Bimg_, sampler, (int2)(x, y));\n"
    "float4 sG = read_imagef(Gimg_, sampler, (int2)(x, y));\n"
    "float4 sR = read_imagef(Rimg_, sampler, (int2)(x, y));\n"
    "float B = sB.x/255.0f;\n"
    "float G = sG.x/255.0f;\n"
    "float R = sR.x/255.0f;\n"
    "float b = 0.0f;\n"
    "float g = 0.0f;\n"
    "float r = 0.0f;\n"
    "if(R <= 0.04045f) r = R/12.92f;\n"
    "else              r = pow((R+0.055f)/1.055f, 2.4f);\n"
    "if(G <= 0.04045f) g = G/12.92f;\n"
    "else              g = pow((G+0.055f)/1.055f, 2.4f);\n"
    "if(B <= 0.04045f) b = B/12.92f;\n"
    "else	      b = pow((B+0.055f)/1.055f, 2.4f);\n"
    "float X = r*0.4124564f + g*0.3575761f + b*0.1804375f;\n"
    "float Y = r*0.2126729f + g*0.7151522f + b*0.0721750f;\n"
    "float Z = r*0.0193339f + g*0.1191920f + b*0.9503041f;\n"
    "float epsilon = 0.008856f;\n"
    "float kappa   = 903.3f;\n"
    "float Xr = 0.950456f;\n"
    "float Yr = 1.0f;\n"
    "float Zr = 1.088754f;\n"
    "float xr = X/Xr;\n"
    "float yr = Y/Yr;\n"
    "float zr = Z/Zr;\n"
    "float fx = 0.0f;\n"
    "float fy = 0.0f;\n"
    "float fz = 0.0f;\n"
    "if(xr > epsilon) fx = pow(xr, 1.0f/3.0f);\n"
    "else fx = (kappa*xr + 16.0f)/116.0f;\n"
    "if(yr > epsilon) fy = pow(yr, 1.0f/3.0f);\n"
    "else fy = (kappa*yr + 16.0f)/116.0f;\n"
    "if(zr > epsilon) fz = pow(zr, 1.0f/3.0f);\n"
    "else fz = (kappa*zr + 16.0f)/116.0f;\n"
    "sR.x =  yr > epsilon ? (116.0*fy-16.0):(yr*kappa);\n"
    "sG.x =  500.0f * (fx-fy);\n"
    "sB.x =  200.0f * (fy-fz);\n"
    "write_imagef(Limg, (int2)(x, y), sR);\n"
    "write_imagef(Aimg, (int2)(x, y), sG);\n"
    "write_imagef(Bimg, (int2)(x, y), sB);\n"
"}\n"
};

const char * REDUCTION_KERNEL[] = {
"__kernel void reduction(__global uchar4* Limg, __global uint4* L_reduce,\n"
                     "__global uchar4* Aimg, __global uint4* A_reduce,\n"
                     "__global uchar4* Bimg, __global uint4* B_reduce,\n"
                     "int image_pixels)\n"
"{\n"
    "image_pixels = image_pixels / 4;\n"
    "unsigned int tid = get_local_id(0);\n"
    "unsigned int local_size = get_local_size(0);\n"
    "unsigned int global_size = get_global_size(0);\n"
    "uint4 L_pixels = (uint4) { 0, 0, 0, 0 };\n"
    "uint4 A_pixels = (uint4) { 0, 0, 0, 0 };\n"
    "uint4 B_pixels = (uint4) { 0, 0, 0, 0 };\n"
    "__local uint4 L_cache[64];\n"
    "__local uint4 A_cache[64];\n"
    "__local uint4 B_cache[64];\n"
    "unsigned int i = get_global_id(0);\n"
    "while (i < image_pixels)\n"
    "{\n"
        "L_pixels += convert_uint4(Limg[i]);\n"
        "A_pixels += convert_uint4(Aimg[i]);\n"
        "B_pixels += convert_uint4(Bimg[i]);\n"
        "i += global_size;\n"
    "}\n"
    "L_cache[tid] = L_pixels;\n"
    "A_cache[tid] = A_pixels;\n"
    "B_cache[tid] = B_pixels;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"

    // do reduction in shared mem
    "for (unsigned int s = local_size >> 1; s > 0; s >>= 1)\n"
    "{\n"
        "if (tid < s)\n"
        "{\n"
            "L_cache[tid] += L_cache[tid + s];\n"
            "A_cache[tid] += A_cache[tid + s];\n"
            "B_cache[tid] += B_cache[tid + s];\n"
        "}\n"
        "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "if (tid == 0)\n"
    "{\n"
        "L_reduce[get_group_id(0)] = L_cache[0];\n"
        "A_reduce[get_group_id(0)] = A_cache[0];\n"
        "B_reduce[get_group_id(0)] = B_cache[0];\n"
    "}\n"
"}\n"
};


buffer::CLBuffer *  init_global_clbuffer()
{

    buffer::CLBuffer * clbuffer = (buffer::CLBuffer * ) malloc(sizeof(buffer::CLBuffer));
    cl_device_type      device_type = CL_DEVICE_TYPE_GPU;
    cl_platform_id platform = 0;
    cl_int error;

    error = clGetPlatformIDs(1, &platform, NULL);
    if (error != CL_SUCCESS)
    {
         return 0;
    }


    error = clGetDeviceIDs(platform, device_type, 1, &(clbuffer->device), 0);
    if (error != CL_SUCCESS)
    {
         return 0;
    }

    clbuffer->context = clCreateContext(0, 1, &(clbuffer->device), 0, 0, &error);
    if (! clbuffer->context || error != CL_SUCCESS)
    {
        return 0;
    }


    clbuffer->queue = clCreateCommandQueue(clbuffer->context, clbuffer->device, CL_QUEUE_PROFILING_ENABLE, &error);
    if (! clbuffer->queue || error != CL_SUCCESS)
    {
        return 0;
    }

    (clbuffer->climg_format).image_channel_order = CL_R;
    (clbuffer->climg_format).image_channel_data_type = CL_UNORM_INT8;
    (clbuffer->climg_desc).image_type = CL_MEM_OBJECT_IMAGE2D;
    (clbuffer->climg_desc).image_width = config::IMAGE_COLS;
    (clbuffer->climg_desc).image_height = config::IMAGE_ROWS;
    (clbuffer->climg_desc).image_depth = 0;
    (clbuffer->climg_desc).image_array_size = 0;
    (clbuffer->climg_desc).image_row_pitch = 0;
    (clbuffer->climg_desc).image_slice_pitch = 0;
    (clbuffer->climg_desc).num_mip_levels = 0;
    (clbuffer->climg_desc).num_samples = 0;


    clbuffer->cl_bimg = clCreateImage(clbuffer->context, CL_MEM_READ_ONLY, &(clbuffer->climg_format), &(clbuffer->climg_desc), 0, &error);
    clbuffer->cl_gimg = clCreateImage(clbuffer->context, CL_MEM_READ_ONLY, &(clbuffer->climg_format), &(clbuffer->climg_desc), 0, &error);
    clbuffer->cl_rimg = clCreateImage(clbuffer->context, CL_MEM_READ_ONLY, &(clbuffer->climg_format), &(clbuffer->climg_desc), 0, &error);

    clbuffer->cl_Limg = clCreateImage(clbuffer->context, CL_MEM_WRITE_ONLY, &(clbuffer->climg_format), &(clbuffer->climg_desc), 0, &error);
    clbuffer->cl_Aimg = clCreateImage(clbuffer->context, CL_MEM_WRITE_ONLY, &(clbuffer->climg_format), &(clbuffer->climg_desc), 0, &error);
    clbuffer->cl_Bimg = clCreateImage(clbuffer->context, CL_MEM_WRITE_ONLY, &(clbuffer->climg_format), &(clbuffer->climg_desc), 0, &error);

    clbuffer->cl_sampler_ = clCreateSampler(clbuffer->context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &error);
    if (error != CL_SUCCESS)
    {
        return 0;
    }


    clbuffer->cl_bgr2lab_program = clCreateProgramWithSource(clbuffer->context, sizeof(BGR2LAB_KERNEL)  / sizeof(BGR2LAB_KERNEL[0]),
    BGR2LAB_KERNEL, 0, &error);

    if (error != CL_SUCCESS)
    {
        return 0;
    }
    clBuildProgram(clbuffer->cl_bgr2lab_program, 1, &(clbuffer->device), 0, 0, 0);
    clbuffer->cl_bgr2lab_kernel = clCreateKernel(clbuffer->cl_bgr2lab_program, "bgr2lab", &error);
    if (error != CL_SUCCESS)
    {
        return 0;
    }

    clbuffer->cl_reduction_program =  clCreateProgramWithSource(clbuffer->context, sizeof(REDUCTION_KERNEL)  / sizeof(REDUCTION_KERNEL[0]),
                                                                                             REDUCTION_KERNEL, 0, &error);
    if (error != CL_SUCCESS)
    {
        return 0;
    }


    clBuildProgram(clbuffer->cl_reduction_program, 1, &(clbuffer->device), 0, 0, 0);
    clbuffer->cl_reduction_kernel = clCreateKernel(clbuffer->cl_reduction_program, "reduction", &error);
    if (error != CL_SUCCESS)
    {
        return 0;
    }

    clbuffer->cl_bgr2lab_origin[0] = 0;
    clbuffer->cl_bgr2lab_origin[1] = 0;
    clbuffer->cl_bgr2lab_origin[2] = 0;
    clbuffer->cl_bgr2lab_region[0] = config::IMAGE_COLS;
    clbuffer->cl_bgr2lab_region[1] = config::IMAGE_ROWS;
    clbuffer->cl_bgr2lab_region[2] = 1;
    clbuffer->cl_bgr2lab_global_work_size[0] = config::IMAGE_COLS;
    clbuffer->cl_bgr2lab_global_work_size[1] = config::IMAGE_ROWS;

    clbuffer->cl_reduction_global_work_size[0] = config::CL_REDUCTION_GLOBAL_WORK_SIZE;
    clbuffer->cl_reduction_local_work_size[0] = config::CL_REDUCTION_LOCAL_WORK_SIZE;
    clbuffer->cl_groups = config::CL_GROUPS;

    clbuffer->loutput_buffer = clCreateBuffer(clbuffer->context, CL_MEM_WRITE_ONLY, clbuffer->cl_groups * 4 * sizeof(unsigned int), 0, 0);
    clbuffer->aoutput_buffer = clCreateBuffer(clbuffer->context, CL_MEM_WRITE_ONLY, clbuffer->cl_groups * 4 * sizeof(unsigned int), 0, 0);
    clbuffer->boutput_buffer = clCreateBuffer(clbuffer->context, CL_MEM_WRITE_ONLY, clbuffer->cl_groups * 4 * sizeof(unsigned int), 0, 0);

    return clbuffer;

}