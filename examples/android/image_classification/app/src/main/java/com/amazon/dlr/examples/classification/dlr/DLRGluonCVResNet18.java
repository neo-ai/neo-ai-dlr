package com.amazon.dlr.examples.classification.dlr;

import android.app.Activity;

import java.io.IOException;

/** This is DLRGluonCVResNet18. */
public class DLRGluonCVResNet18 extends DLRGluonCVBase {

    public DLRGluonCVResNet18(Activity activity) throws IOException {
        super(activity);
    }

    @Override
    protected String getModelPath() {
        return "dlr_gluoncv_resnet18_v2";
    }
}
