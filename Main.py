from DnnAdapter import DnnAdapter



print('Please enter paths using this format : '
      'C:/Users/User/Python-Projects/FaceDetection/needed_file.txt')
model_input = input('Please enter a path to the model you want to use: ')
weights_input = input('Please enter a path to weights you want to use: ')
task_input = input('Please enter a task type(face_detection or classification): ')
Model = DnnAdapter(model_input, weights_input, task_input)
image_input = input('Please enter a path to the image: ')
Model.process_image(image_input)
