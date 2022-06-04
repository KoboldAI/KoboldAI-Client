cd docker-rocm
xhost +local:docker
cp ../environments/rocm.yml env.yml
docker-compose run --service-ports koboldai bash -c "cd /content && python3 aiserver.py $*"
