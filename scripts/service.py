from flask import Flask, request
from keras.preprocessing import image
from cnn_executor import cargarModelo
import numpy as np

#Initialize the application service
app = Flask(__name__)
global loaded_model, graph
loaded_model, graph = cargarModelo()

#Define a route
@app.route('/')
def main_page():
	return 'Bienvenido a la URP - RNA!'

@app.route('/perrogato/', methods=['GET','POST'])
def perrogato():
	return 'Perros y gatos'

@app.route('/perrogato/default/', methods=['GET','POST'])
def default():
    print (request.args)
	# dimensions of our images.
    img_width, img_height = 50, 50
	# Show
    image_name = request.args.get("imagen")
    img_path='../prueba/'+image_name
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    
    #test_image_path = './prueba/hosh.jpg'
    #test_image = image.load_img(test_image_path)
    #test_image = image.load_img(test_image_path,target_size = (50, 50))
    #test_image = image.img_to_array(test_image) 
    #test_image = np.expand_dims(test_image, axis = 0)
    #result = loaded_model.predict(test_image)
    #if result[0][0] == 1:
    #    print(result[0][0], ' --> Es un perro')
    #else:
    #    print(result[0][0], ' --> Es un gato ')
	
    with graph.as_default():
        result = loaded_model.predict(img)
        if result[0][0] == 1:
            print(result[0][0], ' --> Es un perro')
        else:
            print(result[0][0], ' --> Es un gato ')
# Run de application
app.run(host='0.0.0.0',port=5000)

#--------------------------------------------------

