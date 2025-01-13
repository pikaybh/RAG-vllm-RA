# VESSL Llama RAG Risk Assessment
This repository contains examples of how to use [VESSL](https://www.vessl.ai/). If you want to learn more about VESSL, please follow the [quick start documentation](https://docs.vessl.ai/guides/get-started/quickstart).

Each directory contains the examples of corresponding features, such as [VESSL Run](https://docs.vessl.ai/guides/run/overview), [VESSL Service](https://docs.vessl.ai/guides/serve/overview), and [VESSL Pipeline](https://docs.vessl.ai/guides/pipeline/overview). If you want to dive into them more, please refer to the docs.

## Try out VESSL quickstarts
- [Run RAG chatbot using LangChain with VESSL Run](runs/rag-chatbot/)
- [Fine-tune Meta Llama 3.1 using VESSL Run](runs/finetune-llms/)
- [Run FLUX.1 schnell model](runs/flux.1-schnell)
- [Deploy Llama 3 service with vLLM on VESSL Service](services/service-llama-3)

## Using CLI

Install the VESSL CLI this command:

```
pip install vessl
```

Set up the VESSL CLI with this command:

```
vessl configure
```

Create a run by sepcifying the YAML configuration file:

```
vessl run create -f run.yaml
```

For beginners, a simple [**“Hello World” example**](https://docs.vessl.ai/guides/get-started/quickstart) is recommended.

### RAG API

To run the server, execute the following command:
    
```bash
$ cd app
$ uvicorn main:app --reload
```

## Copyrights

Original code is forked from [repos: vess-ai/examples](https://github.com/vessl-ai/examples.git).