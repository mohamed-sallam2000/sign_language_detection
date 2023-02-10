import tensorflow as tf
from tensorflow import keras
from tensorflow import lite

#this caode must work in colab or ubuntu with tensorflow installed
#colab link :https://colab.research.google.com/drive/1IUIn9ffk5ICKujqPyuGaHL2irQ9Wmtpm#scrollTo=QSLFKa8GfDMr
model = keras.models.load_model('action1.h5')
converter = lite.TFLiteConverter.from_keras_model( model ) 
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter=True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
tf.lite.OpsSet.SELECT_TF_OPS]

tfmodel = converter.convert()
open('model.tflite', 'wb').write(tfmodel)
model = converter.convert()

file = open( 'model.tflite' , 'wb' ) 
file.write( model )
