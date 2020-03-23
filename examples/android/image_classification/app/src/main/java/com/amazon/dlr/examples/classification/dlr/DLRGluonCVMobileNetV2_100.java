package com.amazon.dlr.examples.classification.dlr;

import android.app.Activity;

import java.io.IOException;

/** This is DLRGluonCVMobileNetV2_100. */
public class DLRGluonCVMobileNetV2_100 extends DLRGluonCVBase {

    public DLRGluonCVMobileNetV2_100(Activity activity) throws IOException {
        super(activity);
    }

    @Override
    protected String getModelPath() {
        return "dlr_gluoncv_mobilenet_v2_100";
    }
}
