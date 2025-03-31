FROM prom/prometheus:v2.47.0

COPY ../../config/prometheus /etc/prometheus/
VOLUME ["/prometheus"]
EXPOSE 9090

CMD ["--config.file=/etc/prometheus/prometheus.yml", \
     "--storage.tsdb.path=/prometheus", \
     "--web.console.libraries=/usr/share/prometheus/console_libraries", \
     "--web.console.templates=/usr/share/prometheus/consoles"]