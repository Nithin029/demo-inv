# Use an official Python runtime as a parent image
FROM python:3.11.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Start the FastAPI app on port 7860, the default port expected by Spaces
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
