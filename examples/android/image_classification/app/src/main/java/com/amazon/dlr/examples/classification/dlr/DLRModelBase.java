package com.amazon.dlr.examples.classification.dlr;

import android.app.Activity;
import android.content.res.AssetManager;
import android.util.Log;

import com.amazon.dlr.examples.classification.Classifier;
import com.amazon.neo.dlr.DLR;
import com.amazon.dlr.examples.classification.env.ImageUtils;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

/** This is DLRModelBase. */
public abstract class DLRModelBase extends Classifier {

  /**
   * An array to hold inference results, to be feed into Tensorflow Lite as outputs. This isn't part
   * of the super class, because we need a primitive array here.
   */
  protected float[][] labelProbArray;
  protected long handle;
  protected String inName;
  protected int inSz;
  protected float[] input;
  protected float[] inputCHW;

  /**
   * Initializes a {@code ClassifierFloatMobileNetV1}.
   *
   * @param activity
   */
  public DLRModelBase(Activity activity)
      throws IOException {
    super(activity);
    this.inSz = 1;
    for (long v : getInShape()) {
      this.inSz *= (int) v;
    }
    input = new float[inSz];
    inputCHW = new float[inSz];
    labelProbArray = new float[1][getNumLabels()];

    AssetManager am = activity.getAssets();

    String modelPath = getModelPath();
    File dd = new File(activity.getApplicationContext().getApplicationInfo().dataDir, modelPath);
    String fullModelPath = dd.toString();
    dd.mkdir();

    ImageUtils.copyFromAssets(am, "model.so", modelPath, fullModelPath);
    ImageUtils.copyFromAssets(am, "model.json", modelPath, fullModelPath);
    ImageUtils.copyFromAssets(am, "model.params", modelPath, fullModelPath);

    //File f = new File(activity.getApplicationContext().getApplicationInfo().dataDir);

    //printDir(f);
    handle = DLR.CreateDLRModel(fullModelPath, 1, 0);
    Log.i("DLR", "CreateDLRModel: " + handle);
    if (handle == 0) {
      Log.i("DLR", "DLRGetLastError: " + DLR.DLRGetLastError());
      throw new RuntimeException("CreateDLRModel failed");
    }
    DLR.UseDLRCPUAffinity(handle,false);
    // DLR.SetDLRNumThreads(handle, 4);
    Log.i("DLR", "GetDLRBackend: " + DLR.GetDLRBackend(handle));
    Log.i("DLR", "GetDLRNumInputs: " + DLR.GetDLRNumInputs(handle));
    Log.i("DLR", "GetDLRNumWeights: " + DLR.GetDLRNumWeights(handle));
    Log.i("DLR", "GetDLRNumOutputs: " + DLR.GetDLRNumOutputs(handle));

    inName = DLR.GetDLRInputName(handle, 0);
    Log.i("DLR", "GetDLRInputName[0]: " + inName);
    Log.i("DLR", "GetDLRWeightName[4]: " + DLR.GetDLRWeightName(handle, 4));
    // GetDLROutputSize and Dim
    int outDim = DLR.GetDLROutputDim(handle, 0);
    long outSize = DLR.GetDLROutputSize(handle, 0);
    Log.i("DLR", "GetDLROutputSize[0]: " + outSize);
    Log.i("DLR", "GetDLROutputDim[0]: " + outDim);
    //GetDLROutputShape
    long[] out_shape = new long[outDim];
    if (DLR.GetDLROutputShape(handle, 0, out_shape) != 0) {
      Log.i("DLR", "DLRGetLastError: " + DLR.DLRGetLastError());
      throw new RuntimeException("GetDLROutputShape failed");
    }
    Log.i("DLR", "GetDLROutputShape[0]: " + Arrays.toString(out_shape));
  }

  protected abstract long[] getInShape();

  protected abstract boolean isNCHW();

  @Override
  public int getImageSizeX() {
    return 224;
  }

  @Override
  public int getImageSizeY() {
    return 224;
  }

  protected int getFrameSize() {
    return 50176;
  }

  @Override
  protected int getNumBytesPerChannel() {
    return 4; // Float.SIZE / Byte.SIZE;
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
    //Log.i("DLR", "img.HasArray: " + imgData.hasArray() + ", len: " + imgData.array().length);
    imgData.rewind();
    imgData.asFloatBuffer().get(input, 0, inSz);
    float[] theInput;
    if (isNCHW()) {
      for (int i = 0; i < getFrameSize(); i++) {
        inputCHW[i] = input[3 * i];
        inputCHW[getFrameSize() + i] = input[3 * i + 1];
        inputCHW[getFrameSize() * 2 + i] = input[3 * i + 2];
      }
      //Log.i("HWC", "(" + input[150525] + "," + input[150526] + "," + input[150527] + ")");
      //Log.i("CHW", "(" + inputCHW[50175] + "," + inputCHW[100351] + "," + inputCHW[150527] + ")");
      theInput = inputCHW;
    } else {
      theInput = input;
    }

    if (DLR.SetDLRInput(handle, inName, getInShape(), theInput, 4) != 0) {
      Log.i("DLR", "DLRGetLastError: " + DLR.DLRGetLastError());
      throw new RuntimeException("SetDLRInput failed");
    }
    //Log.i("DLR", "SetDLRInput: OK");

    if (DLR.RunDLRModel(handle) != 0) {
      Log.i("DLR", "DLRGetLastError: " + DLR.DLRGetLastError());
      throw new RuntimeException("RunDLRModel failed");
    }
    //Log.i("DLR", "RunDLRModel: OK");
    float[] output = labelProbArray[0];
    if (DLR.GetDLROutput(handle, 0, output) != 0) {
      Log.i("DLR", "DLRGetLastError: " + DLR.DLRGetLastError());
      throw new RuntimeException("GetDLROutput failed");
    }
  }

  @Override
  public void close() {
    super.close();
    if (handle > 0) {
      DLR.DeleteDLRModel(handle);
      handle = 0;
    }
  }
}
