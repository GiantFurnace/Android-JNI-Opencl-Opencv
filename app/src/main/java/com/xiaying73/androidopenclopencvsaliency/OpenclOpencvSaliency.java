package com.xiaying73.androidopenclopencvsaliency;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;

public class OpenclOpencvSaliency extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("opencl-opencv-saliency");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_opencl_opencv_saliency);

        // Example of a call to a native method
        long ptr = getCLBufferFromJNI();
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inScaled = false;
        Bitmap resource =  BitmapFactory.decodeResource(getResources(), R.drawable.flower, options);
        Bitmap sampleImage = Bitmap.createBitmap(resource, 0, 0, resource.getWidth(), resource.getHeight());

        int height = resource.getHeight();
        int width = resource.getWidth();


        int[] saliencyMap = getFTSaliencyFromJNI(sampleImage, ptr);
        Bitmap saliencyBitmapForDisplay = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);
        saliencyBitmapForDisplay.setPixels(saliencyMap,0, width,0,0, width, height);
        ImageView imageview = findViewById(R.id.sample_view);
        imageview.setImageBitmap(saliencyBitmapForDisplay);

    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
    public native long getCLBufferFromJNI();
    public native  int[] getFTSaliencyFromJNI(Bitmap img, long ptr);
}
