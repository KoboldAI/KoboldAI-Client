cd docker-rocm
xhost +local:docker
docker-compose run --service-ports koboldai bash -c "cd /content && python3 aiserver.py $*"
