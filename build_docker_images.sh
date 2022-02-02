docker image build -f docker-cuda/Dockerfile_base . -t ebolam/koboldai_base:bare
docker image build -f docker-cuda/Dockerfile_base_huggingface . -t ebolam/koboldai_base
docker image build -f docker-cuda/Dockerfile_base_finetune . -t ebolam/koboldai_base:finetune
docker image build -f docker-cuda/Dockerfile . -t ebolam/koboldai
docker image build -f docker-cuda/Dockerfile_finetune . -t ebolam/koboldai:finetune