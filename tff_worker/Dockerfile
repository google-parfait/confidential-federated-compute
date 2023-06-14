# Use a multi-stage build to reduce the final image size.
# TFF does not yet build on python 3.11.
FROM python:3.10.0-slim
RUN apt-get --yes update \
  && apt-get install --no-install-recommends --yes \
  g++
RUN python3 -m venv /opt/venv
# Enable venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install tensorflow-federated==0.59.0

# Final stage
# Use a base image that contains dependencies for running Python as well as a
# lightweight Debian Linux distribution.
# Alpine Linux is even smaller, but getting the hello_world.py example
# to work on Alpine will require more work to determine which
# dependencies are missing.
FROM python:3.10.0-slim
RUN apt-get --yes update \
  && apt-get install --no-install-recommends --yes \
  g++
# Copy the dependencies needed for running TFF from the virtual environment
# created in the previous build stage.
COPY --from=0 /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY hello_world.py app/hello_world.py
