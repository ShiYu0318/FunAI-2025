export tag=latest
export game="racing_car"

docker build \
-t ${game}:${tag} \
-f ./Dockerfile .
