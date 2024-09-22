import triton_python_backend_utils as pb_utils
import numpy as np
import urllib.request
import ssl
from PIL import Image
from io import BytesIO
import pickle 


model_file = '/models/sample/1/models/model_regression.pickle'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

def get_optimal_quality_(url):
    try:
        d = urllib.request.urlopen(url)
        image_bytes = d.read()
        image_io = BytesIO(image_bytes)
        pillow_image = Image.open(image_io)
        file_size_bits = image_io.getbuffer().nbytes * 8
        h, w = pillow_image.size
        bpp = (file_size_bits) / (h * w)
        if bpp > 4.15:
            return 50
        else:
            x_test = np.array([bpp]).reshape((-1, 1))
            score = round(model.predict(x_test)[0], 4)
            return score
        # print(image_io.tell())
    except:
        return 0.
    
class TritonPythonModel:
    def initialize(self, args):
        # Model initialization logic
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get input image path as a string (now handling the batch dimension)
            input_image_path = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE_PATH").as_numpy()

            # Handle the batch dimension (assuming batch size of 1)
            input_image_path = input_image_path[0].decode('utf-8')
            print(input_image_path)
            # Process the image and get the quality score (replace with your logic)
            quality = get_optimal_quality_(input_image_path)
            
            # Create output tensor with the result
            output_tensor = pb_utils.Tensor("OUTPUT_QUALITY", np.array([quality], dtype=np.float32))
            
            # Create inference response
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses

    def finalize(self):
        # Cleanup logic
        pass
