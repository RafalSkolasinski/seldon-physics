KIND_NAME ?= seldon

LOCALHOST_PORT ?= 5000
AMBASSADOR_PORT ?= 8003


deploy:
	kubectl apply -f deploy.yaml

remove:
	kubectl delete -f deploy.yaml

image:
	# docker-build base first to make use of layer caching
	docker build . -t quantum-dot-base:latest

	# Do proper s2i build (add model)
	s2i build -E environment  . quantum-dot-base:latest quantum-dot:latest


kind-image: image
	kind load docker-image quantum-dot:latest --name ${KIND_NAME}

request-k8s:
	curl -X POST -H 'Content-Type: application/json' \
		-d '{"data": {"names": ["input"], "ndarray": [0, 0.1, 0.2]}}' \
		http://localhost:${AMBASSADOR_PORT}/seldon/seldon/seldon-quantum-dot/api/v0.1/predictions



docker-run:
	docker rm -f quantum-dot || echo "Already removed"
	docker run --name "quantum-dot" --rm -d -p 127.0.0.1:${LOCALHOST_PORT}:5000 quantum-dot:latest

docker-rm:
	docker rm -f quantum-dot || echo "Already removed"



local-run:
	PREDICTIVE_UNIT_SERVICE_PORT=${LOCALHOST_PORT} seldon-core-microservice Model REST --service-type MODEL

request-localhost:
	curl -X POST -H 'Content-Type: application/json' \
		-d '{"data": {"names": ["input"], "ndarray": [0, 0.1, 0.2]}}' \
		http://localhost:${LOCALHOST_PORT}/api/v0.1/predictions
