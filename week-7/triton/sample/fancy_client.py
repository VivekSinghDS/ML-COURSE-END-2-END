import tritonclient.http as httpclient
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Initialize the rich Console
console = Console()

# Initialize the Triton client
client = httpclient.InferenceServerClient(url="localhost:8000")

# Create the input data (image path as string)
inputs = []
inputs.append(httpclient.InferInput("INPUT_IMAGE_PATH", [1], "BYTES"))
inputs[0].set_data_from_numpy(np.array(["https://upload.wikimedia.org/wikipedia/commons/3/3f/JPEG_example_flower.jpg"], dtype=np.object_))

# Define the output
outputs = []
outputs.append(httpclient.InferRequestedOutput("OUTPUT_QUALITY"))

# Make the inference request
response = client.infer(model_name="sample", inputs=inputs, outputs=outputs)

# Get the output data (quality score)
quality = response.as_numpy("OUTPUT_QUALITY")

# Beautify the output using rich
table = Table(title="Model Inference Result")

table.add_column("Parameter", justify="center", style="cyan", no_wrap=True)
table.add_column("Value", justify="center", style="magenta")

# Add rows to the table with formatted quality value (2 decimal places)
table.add_row("Image URL", "https://upload.wikimedia.org/wikipedia/commons/3/3f/JPEG_example_flower.jpg")
table.add_row("Predicted Quality", f"{quality[0]:.2f}")

# Print the beautified result in a Panel
console.print(Panel(table, title="Triton Model Inference", subtitle="Prediction Output", border_style="green"))

