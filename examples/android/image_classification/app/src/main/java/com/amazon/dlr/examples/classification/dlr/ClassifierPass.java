package com.amazon.dlr.examples.classification.dlr;

import android.app.Activity;

import com.amazon.dlr.examples.classification.Classifier;

import java.io.IOException;
import java.util.Arrays;


/** This is Pass classifier. */
public class ClassifierPass extends Classifier {

  /**
   * An array to hold inference results, to be feed into model outputs. This isn't part
   * of the super class, because we need a primitive array here.
   */
  private float[][] labelProbArray;

  /**
   * Initializes a {@code ClassifierPass}.
   *
   * @param activity
   */
  public ClassifierPass(Activity activity, Device device, int numThreads)
      throws IOException {
    super(activity);
    labelProbArray = new float[1][getNumLabels()];
  }

  @Override
  public int getImageSizeX() {
    return 224;
  }

  @Override
  public int getImageSizeY() {
    return 224;
  }

  @Override
  protected String getModelPath() {
    return "";
  }

  @Override
  protected String getLabelPath() {
    return "labels.txt";
  }

  @Override
  protected int getNumBytesPerChannel() {
    return 4; // Float.SIZE / Byte.SIZE;
  }

  @Override
  protected void addPixelValue(int pixelValue) {
    //pass
  }

  @Override
  protected float getProbability(int labelIndex) {
    return labelProbArray[0][labelIndex];
  }

  @Override
  protected void setProbability(int labelIndex, Number value) {
    labelProbArray[0][labelIndex] = value.floatValue();
  }

  @Override
  protected float getNormalizedProbability(int labelIndex) {
    return labelProbArray[0][labelIndex];
  }

  @Override
  protected void runInference() {
    double v = 0.5;
    Arrays.fill(labelProbArray[0], 0.0f);
    if (Math.random() > v) {
      labelProbArray[0][284] = 0.998f;
      labelProbArray[0][285] = 0.61f;
      labelProbArray[0][286] = 0.51f;
    } else {
      labelProbArray[0][153] = 0.998f;
      labelProbArray[0][154] = 0.61f;
      labelProbArray[0][155] = 0.51f;
    }
  }
}
