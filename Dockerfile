# Use the official Miniconda image as a base
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy the environment.yml file to the Docker image
COPY environment.yml .

# Create the Conda environment from the environment.yml file
RUN conda env create -f environment.yml

# Activate the environment and make it the default
RUN echo "conda activate imdb" >> ~/.bashrc
ENV PATH /opt/conda/envs/kaggle/bin:$PATH

# Optionally, set an entry point (e.g., Python shell)
ENTRYPOINT ["python"]
