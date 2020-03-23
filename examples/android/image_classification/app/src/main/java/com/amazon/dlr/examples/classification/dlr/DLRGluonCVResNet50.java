package com.amazon.dlr.examples.classification.dlr;

import android.app.Activity;

import java.io.IOException;

/** This is DLRGluonCVResNet50. */
public class DLRGluonCVResNet50 extends DLRGluonCVBase {

    public DLRGluonCVResNet50(Activity activity) throws IOException {
        super(activity);
    }

    @Override
    protected String getModelPath() {
        return "dlr_gluoncv_resnet50_v2";
    }
}
