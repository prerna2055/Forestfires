import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
import uvicorn
# Initialize FastAPI app
app = FastAPI()


# Define the model class based on your saved model
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionModel, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)  # Hidden layer
        self.output = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        x = torch.relu(self.hidden(x))  # Apply ReLU activation
        x = self.output(x)  # Output layer
        return x


# Load the trained model (ensure correct path to your saved model)
input_size = 12  # Number of features
hidden_size = 10
output_size = 1  # Regression has 1 output
learning_rate = 0.01
epochs = 1000
model = RegressionModel(input_size, hidden_size, output_size)

# Load the model weights
model.load_state_dict(torch.load('model.pth'))  # Path to your saved model
model.eval()  # Set the model to evaluation mode


class PredictionRequest(BaseModel):
    features: list


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Convert features into a PyTorch tensor
        features = torch.tensor(request.features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Perform the prediction
        with torch.no_grad():  # No need to compute gradients during inference
            prediction = model(features)

        # Convert the prediction to a scalar value or list
        output = prediction.item()  # If the output is a single value, use .item()

        return {"prediction": output}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
   uvicorn.run(app, host="0.0.0.0", port=8005)