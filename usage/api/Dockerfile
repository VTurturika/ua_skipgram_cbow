FROM tensorflow/tensorflow:latest-py3
WORKDIR /api
RUN pip install Flask flask-cors
CMD ["flask", "run", "--host=0.0.0.0"]