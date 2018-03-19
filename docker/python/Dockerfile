FROM python:latest

RUN apt-get update && apt-get -y dist-upgrade
RUN apt-get -y install --no-install-recommends git

RUN useradd --create-home user

RUN git clone https://github.com/eleurent/highway-env.git /home/user/highway-env
RUN cd /home/user/highway-env && pip install .
RUN chown -R user: /home/user/highway-env
RUN chmod -R 777 /home/user/highway-env

USER user
WORKDIR /home/user/highway-env

CMD ["jupyter", "notebook", "--no-browser", "--ip='0.0.0.0'", "--allow-root"]

EXPOSE 8888
VOLUME /home/user/highway-env/scenarios/out
