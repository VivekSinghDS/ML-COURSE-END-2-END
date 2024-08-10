import numpy as np
import pinecone
import os 

class PineconeClient:
    def __init__(self, api_key: str, environment: str, index_name: str, dimension: int) -> None:
        pinecone.init(api_key=api_key, environment=environment)
        self.index_name = index_name
        self.dimension = dimension
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name=self.index_name, dimension=self.dimension)
        self.index = pinecone.Index(self.index_name)

    def create_vector(self, vector_id: str) -> None:
        vector = np.random.rand(self.dimension).tolist()  
        self.index.upsert([(vector_id, vector)])
        print(f"Vector with ID '{vector_id}' created.")

    def read_vector(self, vector_id: str) -> None:
        response = self.index.fetch([vector_id])
        if vector_id in response['vectors']:
            vector_data = response['vectors'][vector_id]
            print(f"Vector with ID '{vector_id}' found: {vector_data}")
        else:
            print(f"Vector with ID '{vector_id}' not found.")

    def update_vector(self, vector_id: str) -> None:
        vector = np.random.rand(self.dimension).tolist()
        self.index.upsert([(vector_id, vector)])
        print(f"Vector with ID '{vector_id}' updated.")

    def delete_vector(self, vector_id: str) -> None:
        self.index.delete(ids=[vector_id])
        print(f"Vector with ID '{vector_id}' deleted.")

    def list_vectors(self) -> None:
        print("Listing vectors in the index:")
        for vector in self.index.query('', top_k=10)['matches']:  
            print(f"ID: {vector['id']}, Vector: {vector['values']}")

    def __del__(self):
        self.index.close()

# Example usage:
api_key = os.getenv('api_key')
environment = os.getenv('env')
index_name = 'example-index'
dimension = 128  # Example dimension

client = PineconeClient(api_key, environment, index_name, dimension)

# Create a vector
client.create_vector("vector_1")

# Read a vector
client.read_vector("vector_1")

# Update the vector
client.update_vector("vector_1")

# List vectors
client.list_vectors()

# Delete the vector
client.delete_vector("vector_1")
