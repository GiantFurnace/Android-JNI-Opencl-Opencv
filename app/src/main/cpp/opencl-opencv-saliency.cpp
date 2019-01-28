#include "include/utils.h"
#include "include/opencv_common.h"
#include <jni.h>
#include <stdlib.h>
#include <string>

#include <android/log.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <sys/time.h>
#include <unistd.h>
#include<android/bitmap.h>

using namespace cv;
extern "C" JNIEXPORT jstring

JNICALL
Java_com_xiaying73_androidopenclopencvsaliency_OpenclOpencvSaliency_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello From C++";
    return env->NewStringUTF(hello.c_str());
}


extern "C" JNIEXPORT jlong JNICALL
Java_com_xiaying73_androidopenclopencvsaliency_OpenclOpencvSaliency_getCLBufferFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    buffer::CLBuffer * buffer = init_global_clbuffer();
    // int * data = new int(88);
    return (unsigned long) buffer;
}


extern "C" JNIEXPORT  jintArray JNICALL
Java_com_xiaying73_androidopenclopencvsaliency_OpenclOpencvSaliency_getFTSaliencyFromJNI(
        JNIEnv* env,
        jobject /* this */, jobject sample_bitmap,  jlong ptr) {

    buffer::CLBuffer * clbuffer = (buffer::CLBuffer *) ptr;

    void*  pixels;
    int    ret;
    ret = AndroidBitmap_lockPixels(env,  sample_bitmap, &pixels);
    jintArray saliency_map = env->NewIntArray(config::IMAGE_PIXELS);
    int SALIENCY_BUFFER[config::IMAGE_PIXELS];

    if (clbuffer != 0 && ret >=0)
    {

        cv::Mat tmp(config::IMAGE_ROWS, config::IMAGE_COLS, CV_8UC4, (char*)pixels);
        cv::Mat sample_image;
        cvtColor(tmp, sample_image, CV_RGBA2BGR);

        Mat channels[3];
        split(sample_image, channels);
        size_t reduce_buffer_bytes = clbuffer->cl_groups * 4 * sizeof(unsigned int);

        for (int index = 0; index < config::IMAGE_PIXELS; ++index)
        {
            clbuffer->bimg_buffer[index] = channels[0].data[index];
            clbuffer->gimg_buffer[index] = channels[1].data[index];
            clbuffer->rimg_buffer[index] = channels[2].data[index];
        }

        clSetKernelArg(clbuffer->cl_bgr2lab_kernel, 0, sizeof(cl_mem), &(clbuffer->cl_bimg));
        clSetKernelArg(clbuffer->cl_bgr2lab_kernel, 1, sizeof(cl_mem), &(clbuffer->cl_gimg));
        clSetKernelArg(clbuffer->cl_bgr2lab_kernel, 2, sizeof(cl_mem), &(clbuffer->cl_rimg));
        clSetKernelArg(clbuffer->cl_bgr2lab_kernel, 3, sizeof(cl_mem), &(clbuffer->cl_Limg));
        clSetKernelArg(clbuffer->cl_bgr2lab_kernel, 4, sizeof(cl_mem), &(clbuffer->cl_Aimg));
        clSetKernelArg(clbuffer->cl_bgr2lab_kernel, 5, sizeof(cl_mem), &(clbuffer->cl_Bimg));
        clSetKernelArg(clbuffer->cl_bgr2lab_kernel, 6, sizeof(cl_sampler), &(clbuffer->cl_sampler_));

        clEnqueueWriteImage(clbuffer->queue, clbuffer->cl_bimg, CL_TRUE, clbuffer->cl_bgr2lab_origin, clbuffer->cl_bgr2lab_region, 0, 0, clbuffer->bimg_buffer, 0, 0, 0);
        clEnqueueWriteImage(clbuffer->queue, clbuffer->cl_gimg, CL_TRUE, clbuffer->cl_bgr2lab_origin, clbuffer->cl_bgr2lab_region, 0, 0, clbuffer->gimg_buffer, 0, 0, 0);
        clEnqueueWriteImage(clbuffer->queue, clbuffer->cl_rimg, CL_TRUE,  clbuffer->cl_bgr2lab_origin, clbuffer->cl_bgr2lab_region, 0, 0, clbuffer->rimg_buffer, 0, 0, 0);
        clEnqueueNDRangeKernel(clbuffer->queue, clbuffer->cl_bgr2lab_kernel, 2, 0, clbuffer->cl_bgr2lab_global_work_size, 0, 0, 0, 0);
        clEnqueueReadImage(clbuffer->queue, clbuffer->cl_Limg, CL_TRUE, clbuffer->cl_bgr2lab_origin, clbuffer->cl_bgr2lab_region, 0, 0, clbuffer->bimg_buffer, 0, NULL, NULL);
        clEnqueueReadImage(clbuffer->queue, clbuffer->cl_Aimg, CL_TRUE, clbuffer->cl_bgr2lab_origin, clbuffer->cl_bgr2lab_region, 0, 0, clbuffer->gimg_buffer, 0, NULL, NULL);
        clEnqueueReadImage(clbuffer->queue, clbuffer->cl_Bimg, CL_TRUE, clbuffer->cl_bgr2lab_origin, clbuffer->cl_bgr2lab_region, 0, 0, clbuffer->rimg_buffer, 0, NULL, NULL);

        cl_mem linput_buffer = clCreateBuffer(clbuffer->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, config::IMAGE_PIXELS, clbuffer->bimg_buffer, NULL);
        cl_mem ainput_buffer = clCreateBuffer(clbuffer->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, config::IMAGE_PIXELS, clbuffer->gimg_buffer, NULL);
        cl_mem binput_buffer = clCreateBuffer(clbuffer->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, config::IMAGE_PIXELS,  clbuffer->rimg_buffer, NULL);

        clSetKernelArg(clbuffer->cl_reduction_kernel, 0, sizeof(cl_mem), (void *)&linput_buffer);
        clSetKernelArg(clbuffer->cl_reduction_kernel, 1, sizeof(cl_mem), (void *)&(clbuffer->loutput_buffer));
        clSetKernelArg(clbuffer->cl_reduction_kernel, 2, sizeof(cl_mem), (void *)&ainput_buffer);
        clSetKernelArg(clbuffer->cl_reduction_kernel, 3, sizeof(cl_mem), (void *)&(clbuffer->aoutput_buffer));
        clSetKernelArg(clbuffer->cl_reduction_kernel, 4, sizeof(cl_mem), (void *)&binput_buffer);
        clSetKernelArg(clbuffer->cl_reduction_kernel, 5, sizeof(cl_mem), (void *)&(clbuffer->boutput_buffer));
        clSetKernelArg(clbuffer->cl_reduction_kernel, 6, sizeof(int), &config::IMAGE_PIXELS);

        clEnqueueNDRangeKernel(clbuffer->queue, clbuffer->cl_reduction_kernel, 1, NULL, clbuffer->cl_reduction_global_work_size, clbuffer->cl_reduction_local_work_size, 0, 0, 0);
        clEnqueueReadBuffer(clbuffer->queue, clbuffer->loutput_buffer, CL_TRUE, 0, reduce_buffer_bytes, clbuffer->reduce_l, 0, NULL, NULL);
        clEnqueueReadBuffer(clbuffer->queue, clbuffer->aoutput_buffer, CL_TRUE, 0, reduce_buffer_bytes, clbuffer->reduce_a, 0, NULL, NULL);
        clEnqueueReadBuffer(clbuffer->queue, clbuffer->boutput_buffer, CL_TRUE, 0, reduce_buffer_bytes, clbuffer->reduce_b, 0, NULL, NULL);

        unsigned int lsum = 0;
        unsigned int asum = 0;
        unsigned int bsum = 0;

        for (int index = 0; index < config::REDUCE_BLOCKS; index++)
        {
            lsum += clbuffer->reduce_l[index];
            asum += clbuffer->reduce_a[index];
            bsum += clbuffer->reduce_b[index];
        }

        float L_mean = (float) lsum / config::IMAGE_PIXELS;
        float A_mean = (float) asum / config::IMAGE_PIXELS;
        float B_mean = (float) bsum / config::IMAGE_PIXELS;

        for (int i = 0; i < config::IMAGE_PIXELS; ++i)
        {
            float bdiff = (clbuffer->bimg_buffer[i] > L_mean) ? clbuffer->bimg_buffer[i] - L_mean : L_mean - clbuffer->bimg_buffer[i];
            float gdiff = (clbuffer->gimg_buffer[i] > A_mean) ? clbuffer->gimg_buffer[i] - A_mean : A_mean - clbuffer->gimg_buffer[i];
            float rdiff = (clbuffer->rimg_buffer[i] > B_mean) ? clbuffer->rimg_buffer[i] - B_mean : B_mean - clbuffer->rimg_buffer[i];
            clbuffer->bimg_buffer[i] = static_cast<unsigned char>(bdiff * bdiff + gdiff*gdiff + rdiff * rdiff);
        }

        memcpy(channels[0].data, clbuffer->bimg_buffer, config::IMAGE_PIXELS);
        cv::GaussianBlur(channels[0], channels[0], Size(7,7),0,0);


        for(int index=0; index<config::IMAGE_PIXELS; index++)
        {
            SALIENCY_BUFFER[index]=(int) channels[0].data[index];
        }

        env->SetIntArrayRegion(saliency_map, 0, config::IMAGE_PIXELS, SALIENCY_BUFFER);
        clReleaseMemObject(linput_buffer);
        clReleaseMemObject(ainput_buffer);
        clReleaseMemObject(binput_buffer);
        AndroidBitmap_unlockPixels(env, sample_bitmap);
    }

    return saliency_map;

}