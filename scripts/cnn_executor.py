import tensorflow as tf
from keras.models import model_from_json
from keras.optimizers import Adam

def cargarModelo():
    json_file = open('../model/cnn_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("../model/cnn_model.h5")
    
    loaded_model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])
    graph = tf.get_default_graph()
    return loaded_model, graph  