FROM unclecode/crawl4ai:0.8.0

COPY ./mcp.patch ./mcp.patch
RUN patch --batch -p3 < mcp.patch
#USER root
#RUN apt-get update; apt-get install vim -y
#USER appuser