# alchemy-ner

SEE [alchemy](https://github.com/XiPotatonium/alchemy)

注意clone的时候需要`git clone xxx.git --recursive`

更新alchemy需要`git submodule foreach 'git pull'`

## 依赖

conda:

* rich
* loguru
* typer
* tomlkit
* pytorch
* numpy
* tensorboard (optional)

pip:

* fastapi (optional)
* uvicorn (optional)
* pynvml (如果用户自定义entry且不使用官方的alloc_cuda，那么不是必须的)
