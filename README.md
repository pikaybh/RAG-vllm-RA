# VESSL Llama RAG Risk Assessment

<!--This repository contains examples of how to use [VESSL](https://www.vessl.ai/). If you want to learn more about VESSL, please follow the [quick start documentation](https://docs.vessl.ai/guides/get-started/quickstart).
<!--
<!--Each directory contains the examples of corresponding features, such as [VESSL Run](https://docs.vessl.ai/guides/run/overview), [VESSL Service](https://docs.vessl.ai/guides/serve/overview), and [VESSL Pipeline](https://docs.vessl.ai/guides/pipeline/overview). If you want to dive into them more, please refer to the docs.
<!--
<!--## Try out VESSL quickstarts
<!--- [Run RAG chatbot using LangChain with VESSL Run](runs/rag-chatbot/)
<!--- [Fine-tune Meta Llama 3.1 using VESSL Run](runs/finetune-llms/)
<!--- [Run FLUX.1 schnell model](runs/flux.1-schnell)
<!--- [Deploy Llama 3 service with vLLM on VESSL Service](services/service-llama-3)-->

This repository contains source code for a research project conducted by [SNUCEM](https://cem.snu.ac.kr/) focused on developing risk assessment models for LLM-based RAG (Retrieval-Augmented Generation) systems.

## Notice

⚠️ This is a private repository. 
Copying, distributing, or using this code without explicit permission from the authors is prohibited. 

For inquiries about this project, please contact [@pikaybh](mailto:pikaybh@snu.ac.kr) or [below](#contact).

## Todos

- [ ] Frontend 구현 (Select할 수 있는 Input List를 각 회사의 API로 구현)
- [ ] 모델 Troubleshooting 구현
- [ ] Loader 추상화
- [ ] 한국어 모델(EEVE) Resource 구현
- [ ] Query 처리 구현 (잘 안되면 raw json으로 처리)

## Getting Started

### Using Vessl CLI

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

### Running the server

To run the server, execute the following command:
    
```bash
$ cd app
$ uvicorn main:app --reload
```

## API

This app provides a REST API.

```
GET /health
POST /v1/openai/kras/invoke
```

### API Endpoints

```http
GET /health
    Returns the current status of the RAG system
    
    Response 200:
    {
        "status": "OK",
        "version": "1.0.0"
    }

POST /v1/openai/kras/invoke
    Performs risk assessment on the given input
    
    Request Header:
    {
        "Authorization": "X-API-KEY <API_KEY>"
    }

    Request Body:
    {
        "query": "string",      // 이거 아님! TODO: 수정 필요
        "context": "string",    // 이거 아님! TODO: 수정 필요
        "response": "string"    // 이거 아님! TODO: 수정 필요
    }

    Response 200:
    {
        ...
    }
```

## Architecture

The architecture of this project is as follows:

```
app
├── assets
│   ├── ...
├── models
│   ├── ...
아, 귀찮다... 직관적으로 이름 지어놨으니깐, 알잘딱깔쎈 하셈
```

이하 필요하다고 생각되는 내용 있으시면 알아서 추가해주세요...
어차피 예쁘게 정리해줘도 안 읽을거고, 결국 코딩 내가 다 하게 될 거잖아 ㅋ

## Contact

```yaml
Byunghee Yoo:
- Email: pikaybh@snu.ac.kr
- Page: https://pikaybh.github.io/
- Github: https://github.com/pikaybh
```

## Copyright

© 2025 [SNUCEM](https://cem.snu.ac.kr/). All Rights Reserved.
