from dlr import DLRModel
import numpy as np
import cv2
from pathlib import Path

MODEL_PATH = Path("../../../build/ssd_mobilenet_v2").resolve()
DATA_PATH = Path("../../../build/street_small.npy").resolve()

def test_ssd_mobilenet_v2_model():
  model = DLRModel(MODEL_PATH.as_posix())
  data = np.load(DATA_PATH)
  assert model.get_input_names() == ['image_tensor']
  assert model.get_output_names() == ['detection_scores:0', 'detection_classes:0', 'num_detections:0']
  assert model.get_input_dtypes() == ['uint8']
  assert model.get_output_dtypes() == ['float32', 'float32', 'float32']
  outputs = model.run({"image_tensor": data})
  assert outputs[0].shape == (1, 100, 4)
  assert outputs[1].shape == (1, 100)
  assert outputs[2].shape == (1, 100)
  detections = np.multiply(np.ceil(outputs[1]), outputs[2])
  expected = np.zeros(detections.shape)
  expected[:,:6] = np.array([[1., 1., 1., 2., 3., 1]])
  comparison = detections == expected
  assert comparison.all()

if __name__ == '__main__':
    test_ssd_mobilenet_v2_model()
    print('All tests passed!')
