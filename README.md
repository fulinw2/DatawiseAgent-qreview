<div align="center">

# DatawiseAgent: A Notebook-Centric LLM Agent Framework for Adaptive and Robust Data Science Automation

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.07044) [![Github](https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/zimingyou01/DatawiseAgent)
[![Huggingface](https://img.shields.io/badge/-HuggingFace-3B4252?style=for-the-badge&logo=huggingface)](https://huggingface.co/datasets/JasperYOU/DatawiseAgent-benchmarkdata)
</div>


📘 DatawiseAgent is a notebook-based LLM agent framework with FST-based multi-stage architecture designed to mimic human-like workflows for adaptive, robust, and end-to-end data science automation.


---



## 🗂️ Table of Contents
- [⚙️ Environment Setup](#️-environment-setup)
- [🚀 Start DatawiseAgent as a Backend Server](#-start-datawiseagent-as-a-backend-server)
  - [⛏️ Configuration](#️-configuration)
  - [▶️ Start the Server](#️-start-the-server)
- [📊 Evaluation](#-evaluation)
  - [📈 Experiment Results](#-experiment-results)
  - [📑 Benchmark Data](#-benchmark-data)
  - [🔍 Data Analysis (InfiAgentBench)](#-data-analysis-infiagentbench)
  - [🎨 Scientific Visualization (MatplotBench)](#-scientific-visualization-matplotbench)
  - [𝌭 DataModeling (DSBench)](#-datamodeling-dsbench)
- [📚 Citing](#-citing)


## ⚙️ Environment Setup
1. **Clone the  repository** 📥
   ```bash
   git clone git@github.com:zimingyou01/DatawiseAgent.git
   cd DatawiseAgent
   ```

2. **Set up Python environment** 🧑‍💻 
    We recommend Python ≥ 3.10 (tested with 3.10).
    ```bash
    conda create -n datawise python=3.10 -y
    conda activate datawise
    pip install -r requirements.txt
    ```

3. **Build Docker image for sandboxed code execution** 🐳

    DatawiseAgent supports a Docker-based sandbox for secure code execution. Please ensure [Docker](https://docs.docker.com/engine/install/) is installed before proceeding.
   
    *  **CPU-only environment**
        Build the default image and set `image_name`: `my-jupyter-image` in the configuration YAML file (see [Agent hyperparameters](#agent-config)):

        ```bash
        docker build -t my-jupyter-image -f datawiseagent/coding/jupyter/default_jupyter_server.dockerfile . --progress=plain
        ```
    * **GPU-enabled environment (e.g., DataModeling tasks)**
        Build the GPU-compatible image and set `image_name`: `my-jupyter-image-gpus` in the configuration YAML file (see [Agent hyperparameters](#agent-config)):

        ```bash
        docker build -t my-jupyter-image-gpus -f datawiseagent/coding/jupyter/cuda_jupyter_server.dockerfile . --progress=plain
        ```
4. **Optional: Deploy open-source LLMs with vLLM** 🤖
    DatawiseAgent is designed to work seamlessly with both commercial and open-source LLMs (e.g., **GPT-4o, GPT-4o-mini, Qwen2.5 7B/14B/32B/72B**).

    For open-source models, we recommend using [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) to serve them as an OpenAI-compatible API.
    Once deployed, simply configure in `.env` (see [Environment variables](#env-config)):
    ```bash
    export OPENAI_API_KEY=<your_key>
    export OPENAI_BASE_URL=<your_vllm_server_url>
    ```

##  🚀 Start DatawiseAgent as a Backend Server

We recommend running DatawiseAgent as a backend server.

### ⛏️ Configuration
DatawiseAgent requires **two levels of configuration**:
1. **Environment variables (`.env`)** <a name="env-config"></a>
    * Copy the template file and fill in required fields:
    ```bash
    cp .env.example .env
    ```
    * At a minimum, set the following variables:
      * `OPENAI_API_KEY` - your OpenAI (or OpenAI-compatible) API key
      * `OPENAI_BASE_URL` - the API base URL (can point to [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) or OpenAI's official endpoint)
    * By default, DatawiseAgent loads settings from `configs/default_config.yaml`.
    If you wish to use a custom config file, specify it in `.env`:
    ```bash
    export CUSTOM_CONFIG=configs/my_config.yaml
    ```



2. **Agent hyperparameters (configs/*.yaml)** <a name="agent-config"></a>
    * The YAML files in `configs/` define all hyperparameters of DatawiseAgent.
    * Each field is annotated with explanations for clarity.
    * We provide two reference configurations for reproducibility:
      * `default_config.yaml` → used in *InfiAgentBench* and *MatplotBench* experiments
      * `datamodeling.yaml` → used in Datamodeling tasks in DSBench
    * The main differences between them lie in:
      * The Docker image used for code execution
      * Minor adjustments to the finite-state transducer limitations in DatawiseAgent

### ▶️ Start the Server
Launch DatawiseAgent as a backend service:
```bash
python main.py
```
Once started, the server runs at:
👉 http://localhost:8000


## 📊 Evaluation

We evaluate DatawiseAgent on three representative data science scenarios:

* **Data Analysis** (*InfiAgentBench*)
* **Scientific Visualization** (*MatplotBench*)
* **Predictive Modelin**g (*DSBench*)

across both **proprietary models** (GPT-4o, GPT-4o-mini) and **open-source models** (Qwen2.5 series).

### 📈 Experiment results
We report the results on **effectiveness**, **adaptability** and **robustness**:
- **Effectiveness** and **Adaptability**

<p align="center">
  <img src="evaluation/experimental_results/effective_result.jpg" alt="Effectiveness Results" width="100%"/>
</p>

- **Robustness**

<p align="center">
  <img src="evaluation/experimental_results/robust_result.jpg" alt="Robustness Results" width="60%"/>
</p>

### 📑 Benchmark Data
To facilitate further research, we open-source the experimental results and agent trajectories under:
```
evaluation/experimental_results/
```

**💡 To reproduce results, follow these steps:**
1. **Download benchmark data** 
    All benchmarks (MatplotBench, InfiAgentBench, Datamodeling) are hosted on Hugging Face:  
    👉 [DatawiseAgent-benchmarkdata](https://huggingface.co/datasets/JasperYOU/DatawiseAgent-benchmarkdata)

    After downloading, the dataset contains three folders:  
    - `MatplotBench/`  
    - `InfiAgentBench/`  
    - `Datamodeling/`  

    Move their contents into the corresponding `evaluation/*/data` directories (create the `data/` folders if they don’t exist):  
    ```bash
    mkdir -p evaluation/MatplotBench/data evaluation/InfiAgentBench/data evaluation/DataModeling/data

    mv DatawiseAgent-benchmarkdata/MatplotBench/* evaluation/MatplotBench/data/
    mv DatawiseAgent-benchmarkdata/InfiAgentBench/* evaluation/InfiAgentBench/data/
    mv DatawiseAgent-benchmarkdata/Datamodeling/* evaluation/DataModeling/data/
    ```
    🥑 Alternative:
    If you prefer, you can also download the benchmark datasets individually from their original repositories and manually organize them under the `evaluation/*/data` folders:
    * [MatplotBench](https://github.com/thunlp/MatPlotAgent)
    * [InfiAgentBench](https://github.com/InfiAgent/InfiAgent)
    * datamodeling tasks from [DSBench](https://github.com/LiqiangJing/DSBench)

2. **Run all evaluations from the root of `evaluation/`**

    ```
    cd evaluation/
    ```
3. **Ensure the backend server address is consistent with `chat_test_asyncio.py`**
    ```python
    # set in chat_test_asyncio.py
    BASE_URL = "http://0.0.0.0:8000"
    BASE_WS_URL = "ws://localhost:8000/register_websocket"
    ```


### 🔍 Data Analysis ([InfiAgentBench](https://github.com/InfiAgent/InfiAgent))
<a name="Data-Analysis"></a>
1. Run evaluation:

    ```bash
    python ./eval_infiagent_bench.py --note "temperature-0_2-args-7-6-8"
    ```

2. Evaluate results (following [InfiAgentBench](https://github.com/InfiAgent/InfiAgent) protocol):
    * **Step 1: Configure API key and base URL**
        Set values in:
        * `evaluation/InfiAgentBench/scripts/api_key.txt`
        * `evaluation/InfiAgentBench/scripts/url.txt`

    * **Step 2: Reformat agent trajectories**

        ```bash
        python ./InfiAgentBench/scripts/reformat.py \
            --model "gpt-4o-mini" \
            --responses_file_path ./data/results_datawise-test.jsonl \
            --output_file_path ./experimental_results/InfiAgent-Bench/reformat/results_reformat_datawise-test.jsonl
        ```
    
    * **Step 3: Evaluate reformatted answers against ground truth**

        ```bash
        python ./InfiAgentBench/scripts/eval_closed_form.py \
                --questions_file_path ./InfiAgentBench/data/da-dev-questions.jsonl \
                --labels_file_path ./InfiAgentBench/data/da-dev-labels.jsonl \
                --responses_file_path ./experimental_results/InfiAgent-Bench/reformat/results_datawise-test.jsonl
        ```



### 🎨 Scientific Visualization ([MatplotBench](https://github.com/thunlp/MatPlotAgent))
<a name="Scientific-Visualization"></a>

1. Run MatplotBench tasks:
    ```bash
    python ./eval_infiagent_bench.py --note "temperature-0_2-args-7-6-8" [--with_tool]
    ```

    - `--with_tool` *(Optional)*: Enable the **visual tool** during evaluation.  
    - ⚠️ Before enabling `--with_tool`, make sure to set your API key and base URL in  
      `datawiseagent/tools/dsbench/vision_tool.py`:
      ```python
      # TODO: explicitly write in the api_key and base_url
      api_key = "<YOUR_API_KEY>"
      base_url = "<YOUR_BASE_URL>"
      ```

2. Evaluate results with model-based evaluation:
    In our experimental setting, we use `gpt-4o-2024-08-06` as the default judge VLM.
    * Set values in:
        * `evaluation/MatplotBench/scripts/api_key.txt`
        * `evaluation/MatplotBench/scripts/url.txt`
    
    * Run evaluation script:
    ```bash
    python ./MatplotBench/scripts/model_eval.py --dir ./experimental_results/MatplotBench/gpt-4o-mini
    ```
    

### 𝌭 DataModeling ([DSBench](https://github.com/LiqiangJing/DSBench))
<a name="DataModeling"></a>

1. Run DataModeling tasks:
    ```bash
    python eval_data_modeling.py \
        --user_name "DataModeling-gpt4o-mini-temperature=0-args=(7,6,8)-for-loop" \
        --result_path "./results/DataModeling/gpt-4o-mini/"
    ```

2. Compute evaluation scores:

    ```bash
    python ./DataModeling/scripts/score4each.py --model gpt-4o-mini
    ```

3. Aggregate and summarize results:
    ```bash
    python ./DataModeling/scripts/show_results.py --model gpt-4o-mini
    ```

👉 Detailed parameter usage is documented in the comments of each script.


## 📚 Citing

If you build upon this work, please cite the accompanying paper:

```bibtex
@article{you2025datawiseagent,
  title={DatawiseAgent: A Notebook-Centric LLM Agent Framework for Automated Data Science},
  author={You, Ziming and Zhang, Yumiao and Xu, Dexuan and Lou, Yiwei and Yan, Yandong and Wang, Wei and Zhang, Huaming and Huang, Yu},
  journal={arXiv preprint arXiv:2503.07044},
  year={2025}
}
```