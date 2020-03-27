package com.amazon.dlr.examples.classification.dlr;

import android.app.Activity;

import java.io.IOException;

/** This is DLRGluonCVBase. */
public abstract class DLRGluonCVBase extends DLRModelBase {

    protected static long[] inShape = new long[] {1,3,224,224};

    public DLRGluonCVBase(Activity activity) throws IOException {
        super(activity);
    }

    @Override
    protected long[] getInShape() {
        return inShape;
    }

    @Override
    protected boolean isNCHW() {
        return true;
    }

    @Override
    protected void addPixelValue(int pixelValue) {
        float rf = (((pixelValue >> 16) & 0xFF) - 123) / 58.395f;
        float gf = (((pixelValue >> 8) & 0xFF) - 117) / 57.12f;
        float bf = ((pixelValue & 0xFF) - 104) / 57.375f;
        //Log.i("DLR", "(" + rf + ","+ gf + ","+ bf + ")");
        imgData.putFloat(rf);
        imgData.putFloat(gf);
        imgData.putFloat(bf);
    }

    @Override
    protected String getLabelPath() {
        return "labels1000.txt";
    }
}
