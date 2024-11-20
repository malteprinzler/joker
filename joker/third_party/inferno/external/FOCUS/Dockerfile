FROM python:3.7

ADD . ./FOCUS
WORKDIR ./FOCUS
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install libgl1
ENV PIP_ROOT_USER_ACTION=ignore
RUN echo "$PWD"
RUN pip install -r ./requirements.txt
ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install dlib

CMD ls
CMD python ./show_instructions.py
