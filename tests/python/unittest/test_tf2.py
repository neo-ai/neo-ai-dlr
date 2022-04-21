from dlr import DLRModel
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

SAVED_MODEL_PATH = "/tmp/saved_model"

signature_list = [
    tf.TensorSpec(shape=[2, 2], dtype=tf.float32, name="input1"),
    tf.TensorSpec(shape=[2, 2], dtype=tf.float32, name="input2"),
]


class TestTF2Model(Model):
    @tf.function(input_signature=[signature_list])
    def call(self, inputs):
        a, b = inputs
        ab = tf.matmul(a, b)
        mm = tf.matmul(a, ab)
        output1 = tf.square(mm)
        mm_flat = tf.reshape(mm, shape=[-1])
        output2 = tf.argmax(mm_flat)
        return {"output1": output1, "output2": output2}


def test_tf2_model(dev_type=None, dev_id=None):

    model = TestTF2Model()

    tf.saved_model.save(model, SAVED_MODEL_PATH)
    model = DLRModel(SAVED_MODEL_PATH, dev_type, dev_id)

    inp1 = tf.constant([[4.0, 1.0], [3.0, 2.0]])
    inp2 = tf.constant([[0.0, 1.0], [1.0, 0.0]])
    # list input
    inputs = [inp1, inp2]
    res = model.run(inputs)
    # dict input
    inputs = {"input1": inp1, "input2": inp2}
    res = model.run(inputs)

    inp_names = model.get_input_names()
    assert sorted(inp_names) == sorted(["input1", "input2"])

    out_names = model.get_output_names()
    assert sorted(out_names) == sorted(["output1", "output2"])

    input_name_0 = model.get_input_name(0)
    assert input_name_0 == inp_names[0]

    output_name_0 = model.get_output_name(0)
    assert output_name_0 == out_names[0]

    input_dtypes = model.get_input_dtypes()
    print("Model input types: ", input_dtypes)
    assert model.get_input_dtype(0) == input_dtypes[0]
    output_dtypes = model.get_output_dtypes()
    print("Model output types: ", output_dtypes)
    assert model.get_output_dtype(0) == output_dtypes[0]

    assert res is not None
    assert len(res) == 2
    assert np.alltrue(res["output1"] == [[36.0, 361.0], [49.0, 324.0]])
    assert res["output2"] == 1

    m_inp1 = model.get_input("input1")
    m_inp2 = model.get_input("input2")
    assert np.alltrue(m_inp1 == inp1)
    assert np.alltrue(m_inp2 == inp2)


if __name__ == "__main__":
    test_tf2_model()
    print("All tests passed!")
