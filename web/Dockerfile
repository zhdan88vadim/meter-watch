# Stage 1: Build Angular app
FROM node:22 as build

WORKDIR /meter-watch
COPY meter-watch/package*.json ./
RUN npm ci
COPY meter-watch/ .
RUN npm run build:prod

# Stage 2: Serve with nginx
FROM nginx:alpine
COPY --from=build /meter-watch/dist/meter-watch /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]