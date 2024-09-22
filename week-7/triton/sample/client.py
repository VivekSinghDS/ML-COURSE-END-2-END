import tritonclient.http as httpclient
import numpy as np

# Initialize the Triton client
client = httpclient.InferenceServerClient(url="localhost:8000")
# print(client)
# # Create the input data (image path as string)
inputs = []
inputs.append(httpclient.InferInput("INPUT_IMAGE_PATH", [1], "BYTES"))
inputs[0].set_data_from_numpy(np.array(["https://upload.wikimedia.org/wikipedia/commons/3/3f/JPEG_example_flower.jpg"], dtype=np.object_))

# # Define the output
outputs = []
outputs.append(httpclient.InferRequestedOutput("OUTPUT_QUALITY"))

# Make the inference request
response = client.infer(model_name="sample", inputs=inputs, outputs=outputs)
# print(response)
# # Get the output data (quality score)
quality = response.as_numpy("OUTPUT_QUALITY")
print(f"Predicted Quality: {quality[0]}")
