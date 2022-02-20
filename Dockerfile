FROM nvcr.io/nvidia/deepstream:6.0-devel

RUN apt-get update \
    && apt-get install --no-install-recommends --no-install-suggests -y wget libgeos-dev build-essential \
    libpython3.6 python3-gi python-gst-1.0 python3-pip python3-dev python3-dotenv python3-setuptools git unzip \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/nvidia/deepstream/deepstream/sources/project
COPY . $WORKDIR

RUN /bin/bash setup_yolo.sh
RUN /bin/bash setup_opencv.sh
RUN /bin/bash setup_pyds.sh