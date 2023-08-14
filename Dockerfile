# Utilizamos la imagen oficial de Python 3.9.12
FROM python:3.9.12

# Establecemos el directorio de trabajo en /app
WORKDIR /app

# Instalamos las herramientas necesarias para empaquetar y publicar en PyPI
RUN pip install --no-cache-dir setuptools build pytest==7.4.0

# Copiamos el archivo setup.py y cualquier otro archivo necesario para el paquete
COPY Makefile ./
COPY setup.py ./
COPY README.md ./
COPY estyp/ ./estyp/

RUN pip install --no-cache-dir -e .

# Ejecutamos los tests
COPY test/ ./test/
RUN make -B test
RUN rm -rf test/

# Generamos los archivos de distribución en la carpeta dist
RUN python -m build