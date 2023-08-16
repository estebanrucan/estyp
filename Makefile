# Run with make -B <name>
install:
	pip install --upgrade pip
	pip install --no-cache-dir -e .

test:
	pytest test/

dist:
	docker build -t my-package:latest . --no-cache
	docker run --name container my-package:latest
	sudo docker cp container:/app/dist ./
	docker rm container

upload:
	twine upload dist/*