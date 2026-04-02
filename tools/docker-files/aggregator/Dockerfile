FROM python:3.14-trixie

WORKDIR app
RUN pip install fastmcp && pip install --pre --upgrade fastmcp

ENTRYPOINT ["fastmcp", "run", "config.json", "-t", "http", "--host", "0.0.0.0"]