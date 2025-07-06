export tag="latest"
export game="racing_car"

docker buildx build --platform linux/amd64,linux/arm64 \
-t paiatech/${game}:${tag} \
-f ./Dockerfile . --push