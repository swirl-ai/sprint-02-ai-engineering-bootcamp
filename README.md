# 01-ai-engineering-bootcamp

Welcome to the second sprint of the End-to-End AI Engineering bootcamp! This sprint is dedicated to building your first RAG pipeline as well as introducing observability and evalation to this pipeline.

We strongly recomend you coding along the video available on Maven rather than just cloning the repository and running the code.

If you do need to run the code, this is how:

- Clone the repository.
- Run:
```bash
cp env.example .env
```

Edit `.env` and add your API keys:

```
OPENAI_API_KEY=your_google_api_key
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
```
Keep the remaining configuration as per ```.env.example```.


#### To run the project, execute:

```bash
make run-docker-compose
```

#### To run evals, execute:

```bash
make run-evals
```

Streamlit application: http://localhost:8501

FastAPI documentation: http://localhost:8000/docs

Qdrant UI: http://localhost:6333/dashboard


## This repository uses data provided by the authors of the following paper.
If you use this work, please cite:

```
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```
