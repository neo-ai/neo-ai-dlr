from dlr import DLRModel
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

SAVED_MODEL_PATH = "/tmp/saved_model"

signature_list = [tf.TensorSpec(shape=[2,2], dtype=tf.float32, name="input1"),
                  tf.TensorSpec(shape=[2,2], dtype=tf.float32, name="input2") ]

class TestModel(Model):
    @tf.function(input_signature = [signature_list])
    def call(self, inputs):
        a, b = inputs
        ab = tf.matmul(a, b)
        mm = tf.matmul(a, ab)
        output1 = tf.square(mm)
        mm_flat = tf.reshape(mm, shape=[-1])
        output2 = tf.argmax(mm_flat)
        return {"output1": output1, "output2" : output2}

def test_tf_model(dev_type=None, dev_id=None):
    model = TestModel()

    tf.saved_model.save(model, SAVED_MODEL_PATH)
    model = DLRModel(SAVED_MODEL_PATH, dev_type, dev_id)

    inp1 = tf.constant([[4., 1.], [3., 2.]])
    inp2 = tf.constant([[0., 1.], [1., 0.]])
    # list input
    inputs = [inp1, inp2]
    res = model.run(inputs)
    # dict input
    inputs = {"input1" : inp1, "input2" : inp2}
    res = model.run(inputs)

    inp_names = model.get_input_names()
    assert inp_names == ['input1', 'input2']

    out_names = model.get_output_names()
    assert out_names == ['output1', 'output2']

    assert res is not None
    assert len(res) == 2
    assert np.alltrue(res['output1'] == [[36., 361.], [49.,  324.]])
    assert res['output2'] == 1

    m_inp1 = model.get_input('input1')
    m_inp2 = model.get_input('input2')
    assert np.alltrue(m_inp1 == inp1)
    assert np.alltrue(m_inp2 == inp2)

def test_tf_model_on_cpu_0():
    test_tf_model("cpu", 0)

def test_tf_model_on_gpu_0():
    test_tf_model("gpu", 0)

if __name__ == '__main__':
    test_tf_model()
    print('All tests passed!')