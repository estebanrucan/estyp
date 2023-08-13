# Utilizamos la imagen oficial de Python 3.9.12
FROM python:3.9.12

# Establecemos el directorio de trabajo en /app
WORKDIR /app

# Instalamos las herramientas necesarias para empaquetar y publicar en PyPI
RUN pip install --no-cache-dir setuptools build pytest==7.4.0

# Copiamos el archivo setup.py y cualquier otro archivo necesario para el paquete
COPY setup.py ./
COPY README.md ./
COPY estyp/ ./estyp/

RUN pip install --no-cache-dir -e .

# Ejecutamos los tests
COPY test/ ./test/
RUN pytest test/
RUN rm -rf test/

# Generamos los archivos de distribución en la carpeta dist
RUN python -m build

# Comandos para ejecutar:
# docker build -t my-package:latest . --no-cache
# docker run my-package:latest
# sudo docker cp $(docker ps -a --format "{{.Names}}" | head -1):/app/dist ./
# docker rm $(docker ps -a -q)