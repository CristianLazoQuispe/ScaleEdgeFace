FROM python:3.7.7-slim-stretch

#FROM tensorflow/tensorflow:2.4.0-gpu-jupyter

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Run Jupyter Notebook when the container launches
#CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]
