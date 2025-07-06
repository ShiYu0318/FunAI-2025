export tag="latest"
export game="swimming_squid"

docker build \
-t ${game}:${tag} \
-f ./Dockerfile .
