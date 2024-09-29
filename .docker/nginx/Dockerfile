FROM nginx:latest

RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/templates/nginx.conf.template
COPY proxy_params /etc/nginx