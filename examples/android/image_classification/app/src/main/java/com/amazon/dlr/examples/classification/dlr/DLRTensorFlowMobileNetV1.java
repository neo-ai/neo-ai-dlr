package com.amazon.dlr.examples.classification.dlr;

import android.app.Activity;

import java.io.IOException;

/** This is DLRTensorFlowMobileNetV1. */
public class DLRTensorFlowMobileNetV1 extends DLRModelBase {

    /** MobileNet requires additional normalization of the used input. */
    private static final float IMAGE_MEAN = 127.5f;
    private static final float IMAGE_STD = 127.5f;

    protected static long[] inShape = new long[] {1,224,224,3};

    public DLRTensorFlowMobileNetV1(Activity activity) throws IOException {
        super(activity);
    }

    @Override
    protected long[] getInShape() {
        return inShape;
    }

    @Override
    protected boolean isNCHW() {
        return false;
    }

    @Override
    protected String getModelPath() {
        return "dlr_tf_mobilenet_v1_100";
    }

    @Override
    protected void addPixelValue(int pixelValue) {
        imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
    }

    @Override
    protected String getLabelPath() {
        return "labels.txt";
    }
}
