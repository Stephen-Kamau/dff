FROM python:3.6
COPY . /app
WORKDIR /app
RUN pip install Flask

RUN pip install monai
RUN pip install torch
RUN pip install imageio
RUN pip install nibabel


EXPOSE 5000
CMD ["python", "application.py"]
