# Run with make -B <name>
install:
	pip install --upgrade pip & 
		pip install --no-cache-dir -e .

test:
	pytest test/

dist:
	docker build -t my-package:latest . --no-cache &&\
		docker run my-package:latest &&\
		sudo docker cp $(docker ps -a --format "{{.Names}}" | head -1):/app/dist ./ &&\
		docker rm $(docker ps -a -q)

