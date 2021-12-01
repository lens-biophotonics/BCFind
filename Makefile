.PHONY: wheel
wheel:
	python3 setup.py bdist_wheel

.PHONY: clean
clean:
	rm -fr build dist *.egg-info

.PHONY: docker
docker: docker-image

.PHONY: docker-image
docker-image:
	docker build -t bcfind .
