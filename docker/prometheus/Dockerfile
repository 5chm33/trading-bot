<<<<<<< HEAD
FROM prom/prometheus:v2.47.0

COPY ../../config/prometheus /etc/prometheus/
VOLUME ["/prometheus"]
EXPOSE 9090

CMD ["--config.file=/etc/prometheus/prometheus.yml", \
     "--storage.tsdb.path=/prometheus", \
     "--web.console.libraries=/usr/share/prometheus/console_libraries", \
     "--web.console.templates=/usr/share/prometheus/consoles"]
=======
FROM prom/prometheus:v2.47.0

COPY ../../config/prometheus /etc/prometheus/
VOLUME ["/prometheus"]
EXPOSE 9090

CMD ["--config.file=/etc/prometheus/prometheus.yml", \
     "--storage.tsdb.path=/prometheus", \
     "--web.console.libraries=/usr/share/prometheus/console_libraries", \
     "--web.console.templates=/usr/share/prometheus/consoles"]
>>>>>>> 60870aec3b9ed2c2cb804ceb4f1eeb5c6af9d852
