cd docker-cuda
xhost +local:docker
cp ../environments/finetuneanon.yml env.yml
docker-compose run --service-ports koboldai bash -c "cd /content && python3 aiserver.py $*"
