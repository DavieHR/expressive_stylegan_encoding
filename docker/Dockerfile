FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN sed -i 's/http:\/\/archive.ubuntu.com/http:\/\/mirrors.tuna.tsinghua.edu.cn\/ubuntu/g' /etc/apt/sources.list
RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install \
    build-essential \
    wget \
    curl \
    git \
    make \
    gcc \
    graphviz \
    sudo\
    ffmpeg

RUN conda update conda
RUN conda update conda-build
RUN conda install pip
# RUN conda remove pytorch -y
# RUN conda remove numpy -y
RUN conda create -n pytorch1121 python=3.8
RUN /opt/conda/bin/conda env list
ENV PATH="/opt/conda/envs/pytorch1121/bin:$PATH"
RUN echo $PATH
# RUN /opt/conda/bin/conda activate pytorch1121
RUN python -m pip install --upgrade pip
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install opencv-python==4.8.1.78 loguru==0.7.2 matplotlib==3.5.3 tensorboard==2.11.2 ninja==1.11.1 lpips==0.1.4 awscli

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list
RUN chmod 1777 /tmp

RUN mkdir /app
RUN chmod 777 /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
USER user

# # All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

COPY requirements.txt /app
RUN pip install -r /app/requirements.txt

# sync model and zip file

ADD fan_python /app/fan_python
COPY fan_python/face_mesh.py /home/user/.local/lib/python3.8/site-packages/mediapipe/python/solutions/face_mesh.py
COPY fan_python/face_detection_full_range.tflite  /home/user/.local/lib/python3.8/site-packages/mediapipe/modules/face_detection/
COPY fan_python/face_landmark_front_cpu.binarypb  /home/user/.local/lib/python3.8/site-packages/mediapipe/modules/face_landmark/

RUN mkdir /home/user/.cache/torch/hub/checkpoints/ -p
RUN cp /app/fan_python/* /home/user/.cache/torch/hub/checkpoints/

# ADD tmp/packages /app/tmp/package
# RUN ls /app/tmp/package
# RUN ls /app/tmp/package | xargs -i pip install --no-cache-dir /app/tmp/package/{}
WORKDIR /app


# Set the default command to python3
CMD ["python3", "-m", "DataProcessing"]


